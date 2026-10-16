"""
Benchmark of 100 Math Problems on Code-Neuron Network
with Staged Re-inputting (Beam Search + Residual Boosting)
"""
import time
import numpy as np

np.seterr(all="ignore")

# =====================================================================
# 1. Math Code Blocks Library
# =====================================================================
UNARY_BLOCKS = [
    ("id", lambda x: x, "x"),
    ("neg", lambda x: -x, "(-{x})"),
    ("sq", lambda x: x**2, "({x})**2"),
    ("cube", lambda x: x**3, "({x})**3"),
    ("sqrt", lambda x: np.where(x >= 0, np.sqrt(np.maximum(x, 0)), np.nan), "sqrt({x})"),
    ("inv", lambda x: np.where(np.abs(x) > 1e-7, 1.0 / x, np.nan), "(1/({x}))"),
    ("sin", np.sin, "sin({x})"),
    ("cos", np.cos, "cos({x})"),
    ("tan", lambda x: np.where(np.abs(np.cos(x)) > 1e-4, np.tan(x), np.nan), "tan({x})"),
    ("exp", lambda x: np.where(x < 25, np.exp(np.minimum(x, 25)), np.nan), "exp({x})"),
    ("log", lambda x: np.where(x > 1e-7, np.log(np.maximum(x, 1e-7)), np.nan), "log({x})"),
    ("log1p", lambda x: np.where(x > -0.999, np.log1p(np.maximum(x, -0.999)), np.nan), "log(1+{x})"),
    ("tanh", np.tanh, "tanh({x})"),
    ("abs", np.abs, "abs({x})"),
    ("gauss", lambda x: np.exp(-np.minimum(x**2, 30)), "exp(-({x})**2)"),
    ("inv_sq", lambda x: np.where(np.abs(x) > 1e-5, 1.0 / (x**2), np.nan), "(1/({x})**2)"),
    ("inv_1plus_sq", lambda x: 1.0 / (1.0 + x**2), "(1/(1+({x})**2))"),
]

# Dual target-transforms (to linearize non-linear outputs)
TARGET_TRANSFORMS = [
    ("identity", lambda y: y, lambda t: t, lambda y: True, "{t}"),
    ("log_y", lambda y: np.log(y), lambda t: np.exp(t), lambda y: np.all(y > 1e-7), "exp({t})"),
    ("sqrt_y", lambda y: np.sqrt(y), lambda t: t**2, lambda y: np.all(y >= 0), "({t})**2"),
    ("inv_y", lambda y: 1.0 / y, lambda t: 1.0 / t, lambda y: np.all(np.abs(y) > 1e-7), "(1/({t}))"),
    ("sq_y", lambda y: y**2, lambda t: np.sqrt(np.maximum(t, 0)), lambda y: True, "sqrt({t})"),
]

def fit_linear_readout(F, y_target):
    """
    Solves linear weights: y_hat = F @ w + b using ridge regression.
    Returns (weights, bias, predictions, residual, r2).
    """
    valid = np.all(np.isfinite(F), axis=0) & (np.std(F, axis=0) > 1e-8)
    if not np.any(valid):
        return None
    F_sub = F[:, valid]
    
    # Add bias column
    A = np.column_stack([F_sub, np.ones(len(F_sub))])
    # Ridge solve for stability
    ridge = 1e-9 * np.eye(A.shape[1])
    ridge[-1, -1] = 0.0 # Don't penalize bias
    try:
        w = np.linalg.solve(A.T @ A + ridge, A.T @ y_target)
    except np.linalg.LinAlgError:
        return None
    
    y_pred = A @ w
    res = y_target - y_pred
    var_total = np.var(y_target)
    r2 = 1.0 - (np.var(res) / (var_total + 1e-12))
    return w, valid, y_pred, res, r2

# =====================================================================
# 2. Staged Re-inputting Routing Engine (Beam Search + Boosting)
# =====================================================================
class CodeNeuronNetwork:
    def __init__(self, beam_width=5, max_depth=3, n_boost_stages=2):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.n_boost_stages = n_boost_stages

    def _expand_features(self, X_curr, names_curr):
        """Generates new candidate features by applying unary blocks and binary interactions."""
        N, p = X_curr.shape
        new_cols = []
        new_names = []
        
        # 1. Unary blocks on existing features
        for j in range(p):
            col = X_curr[:, j]
            col_name = names_curr[j]
            for op_name, fn, fmt in UNARY_BLOCKS:
                res = fn(col)
                if np.all(np.isfinite(res)) and np.std(res) > 1e-7:
                    new_cols.append(res)
                    new_names.append(fmt.format(x=col_name))
        
        # 2. Pairwise products among top features
        p_sub = min(p, 6)
        for i in range(p_sub):
            for j in range(i, p_sub):
                prod = X_curr[:, i] * X_curr[:, j]
                if np.all(np.isfinite(prod)) and np.std(prod) > 1e-7:
                    new_cols.append(prod)
                    new_names.append(f"({names_curr[i]} * {names_curr[j]})")
                    
        if not new_cols:
            return X_curr, names_curr
            
        all_cols = np.column_stack([X_curr] + new_cols)
        all_names = names_curr + new_names
        
        # Deduplicate features with identical correlations
        selected_idx = [0]
        for idx in range(1, len(all_names)):
            # Check against already selected
            is_dup = False
            for prev in selected_idx[-20:]:
                if np.allclose(all_cols[:, idx], all_cols[:, prev], rtol=1e-5, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                selected_idx.append(idx)
                
        return all_cols[:, selected_idx], [all_names[i] for i in selected_idx]

    def solve(self, X_train, y_train):
        """
        Learns the math program using staged beam re-inputting and residual boosting.
        """
        N, d = X_train.shape
        init_names = [f"x{i+1}" if d > 1 else "x" for i in range(d)]
        
        y_remaining = y_train.copy()
        stages = []
        
        for stage in range(self.n_boost_stages):
            if np.std(y_remaining) < 1e-6 or np.mean(y_remaining**2) < 1e-10:
                break
                
            best_stage_model = None
            best_stage_r2 = -1e9
            
            # Try each dual target-transform
            for t_name, t_fwd, t_inv, t_chk, t_fmt in TARGET_TRANSFORMS:
                if not t_chk(y_remaining):
                    continue
                y_target = t_fwd(y_remaining)
                if not np.all(np.isfinite(y_target)):
                    continue
                    
                # Beam of candidate feature sets: (features, names, depth)
                beam = [(X_train.copy(), list(init_names))]
                
                for depth in range(1, self.max_depth + 1):
                    next_beam = []
                    for feat_mat, feat_names in beam:
                        # Re-input features and expand
                        exp_feat, exp_names = self._expand_features(feat_mat, feat_names)
                        
                        # Evaluate fit on y_target
                        fit_res = fit_linear_readout(exp_feat, y_target)
                        if fit_res is not None:
                            w, valid, y_pred_t, res_t, r2 = fit_res
                            # Reconstruct prediction in original y space
                            y_pred = t_inv(y_pred_t)
                            if np.all(np.isfinite(y_pred)):
                                raw_res = y_remaining - y_pred
                                raw_r2 = 1.0 - (np.var(raw_res) / (np.var(y_remaining) + 1e-12))
                                
                                if raw_r2 > best_stage_r2:
                                    best_stage_r2 = raw_r2
                                    best_stage_model = {
                                        "t_inv": t_inv,
                                        "w": w,
                                        "valid": valid,
                                        "names": exp_names,
                                        "feat_eval": exp_feat,
                                        "raw_r2": raw_r2,
                                        "t_name": t_name,
                                    }
                        
                        # Select top features to keep beam manageable
                        # Score features by correlation with y_target
                        scores = []
                        for col_idx in range(exp_feat.shape[1]):
                            c = exp_feat[:, col_idx]
                            std = np.std(c)
                            if std > 1e-7:
                                corr = np.abs(np.corrcoef(c, y_target)[0, 1])
                                scores.append((corr if np.isfinite(corr) else 0.0, col_idx))
                            else:
                                scores.append((0.0, col_idx))
                        scores.sort(reverse=True)
                        top_sub = [idx for _, idx in scores[:15]]
                        
                        next_beam.append((exp_feat[:, top_sub], [exp_names[i] for i in top_sub]))
                        
                    beam = next_beam[:self.beam_width]
                    if best_stage_r2 > 0.999999: # Exact match found
                        break
                        
            if best_stage_model is None or best_stage_model["raw_r2"] < 0.05:
                # No progress made in this stage
                break
                
            stages.append(best_stage_model)
            # Subtract explained part from target for residual boosting
            # Compute stage prediction
            val_t = np.column_stack([best_stage_model["feat_eval"][:, best_stage_model["valid"]], np.ones(N)]) @ best_stage_model["w"]
            pred = best_stage_model["t_inv"](val_t)
            y_remaining = y_remaining - pred
            
            if np.mean(y_remaining**2) < 1e-10 or best_stage_model["raw_r2"] > 0.999999:
                break
                
        return stages

# =====================================================================
# 3. 100 Math Problems Benchmark Generator
# =====================================================================
def get_100_problems():
    """Generates 100 diverse mathematical problems across 5 categories."""
    probs = []
    
    # Helper to add problem
    def add(name, category, fn, x_train_range=(0.2, 2.5), x_test_range=(2.5, 4.0), dim=1):
        probs.append({
            "name": name,
            "cat": category,
            "fn": fn,
            "train_range": x_train_range,
            "test_range": x_test_range,
            "dim": dim,
        })

    # --- Category 1: Polynomials & Powers (20 problems) ---
    add("p01: 2*x + 1", "Polynomial", lambda x: 2*x + 1)
    add("p02: -3*x + 4", "Polynomial", lambda x: -3*x + 4)
    add("p03: x^2", "Polynomial", lambda x: x**2)
    add("p04: 3*x^2 - 5*x + 2", "Polynomial", lambda x: 3*x**2 - 5*x + 2)
    add("p05: -x^2 + 4*x - 3", "Polynomial", lambda x: -x**2 + 4*x - 3)
    add("p06: x^3", "Polynomial", lambda x: x**3)
    add("p07: x^3 - 2*x^2 + x", "Polynomial", lambda x: x**3 - 2*x**2 + x)
    add("p08: 2*x^3 + 3*x - 1", "Polynomial", lambda x: 2*x**3 + 3*x - 1)
    add("p09: x^4 - 2*x^2", "Polynomial", lambda x: x**4 - 2*x**2)
    add("p10: (x-1)^2", "Polynomial", lambda x: (x-1)**2)
    add("p11: (x+2)^3", "Polynomial", lambda x: (x+2)**3)
    add("p12: 0.5*x^2 - x + 4", "Polynomial", lambda x: 0.5*x**2 - x + 4)
    add("p13: x^4", "Polynomial", lambda x: x**4)
    add("p14: x^5 - x", "Polynomial", lambda x: x**5 - x)
    add("p15: 4*x^2 + 2", "Polynomial", lambda x: 4*x**2 + 2)
    add("p16: (2*x - 1)^2", "Polynomial", lambda x: (2*x - 1)**2)
    add("p17: 5*x - 10", "Polynomial", lambda x: 5*x - 10)
    add("p18: -0.2*x^3 + x^2", "Polynomial", lambda x: -0.2*x**3 + x**2)
    add("p19: x^2 + 3*x + 5", "Polynomial", lambda x: x**2 + 3*x + 5)
    add("p20: (x-2)*(x-1)*x", "Polynomial", lambda x: (x-2)*(x-1)*x)

    # --- Category 2: Trigonometric & Oscillatory (20 problems) ---
    add("t21: sin(x)", "Trigonometric", lambda x: np.sin(x))
    add("t22: cos(x)", "Trigonometric", lambda x: np.cos(x))
    add("t23: 2*sin(x) - 3*cos(x)", "Trigonometric", lambda x: 2*np.sin(x) - 3*np.cos(x))
    add("t24: sin(x)^2", "Trigonometric", lambda x: np.sin(x)**2)
    add("t25: cos(x)^2", "Trigonometric", lambda x: np.cos(x)**2)
    add("t26: sin(x)*cos(x)", "Trigonometric", lambda x: np.sin(x)*np.cos(x))
    add("t27: sin(2*x)", "Trigonometric", lambda x: np.sin(2*x))
    add("t28: cos(2*x)", "Trigonometric", lambda x: np.cos(2*x))
    add("t29: 3*sin(x) + 1", "Trigonometric", lambda x: 3*np.sin(x) + 1)
    add("t30: sin(x^2)", "Trigonometric", lambda x: np.sin(x**2))
    add("t31: cos(x^2)", "Trigonometric", lambda x: np.cos(x**2))
    add("t32: sin(x) + x", "Trigonometric", lambda x: np.sin(x) + x)
    add("t33: x*cos(x)", "Trigonometric", lambda x: x*np.cos(x))
    add("t34: sin(x)^3", "Trigonometric", lambda x: np.sin(x)**3)
    add("t35: cos(x) - sin(x)", "Trigonometric", lambda x: np.cos(x) - np.sin(x))
    add("t36: tan(x)", "Trigonometric", lambda x: np.tan(x), x_train_range=(0.1, 1.3), x_test_range=(1.35, 1.5))
    add("t37: 1 + sin(x)^2", "Trigonometric", lambda x: 1 + np.sin(x)**2)
    add("t38: 2*cos(x) + x^2", "Trigonometric", lambda x: 2*np.cos(x) + x**2)
    add("t39: sin(x)*cos(x)^2", "Trigonometric", lambda x: np.sin(x)*(np.cos(x)**2))
    add("t40: cos(3*x)", "Trigonometric", lambda x: np.cos(3*x))

    # --- Category 3: Exponential, Gaussian & Logarithmic (20 problems) ---
    add("e41: exp(x)", "Exponential/Log", lambda x: np.exp(x), x_train_range=(-2.0, 1.0), x_test_range=(1.0, 2.0))
    add("e42: exp(-x)", "Exponential/Log", lambda x: np.exp(-x))
    add("e43: 2*exp(x) - 1", "Exponential/Log", lambda x: 2*np.exp(x) - 1, x_train_range=(-1.0, 1.0), x_test_range=(1.0, 1.8))
    add("e44: exp(-x^2) [Gaussian]", "Exponential/Log", lambda x: np.exp(-x**2))
    add("e45: x*exp(-x)", "Exponential/Log", lambda x: x*np.exp(-x))
    add("e46: exp(sin(x))", "Exponential/Log", lambda x: np.exp(np.sin(x)))
    add("e47: exp(-0.5*x^2)", "Exponential/Log", lambda x: np.exp(-0.5*x**2))
    add("e48: log(x)", "Exponential/Log", lambda x: np.log(x), x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("e49: log(1 + x^2)", "Exponential/Log", lambda x: np.log(1 + x**2))
    add("e50: 2*log(x) + 3", "Exponential/Log", lambda x: 2*np.log(x) + 3, x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("e51: log(x)^2", "Exponential/Log", lambda x: np.log(x)**2, x_train_range=(1.0, 4.0), x_test_range=(4.0, 6.0))
    add("e52: x*log(x)", "Exponential/Log", lambda x: x*np.log(x), x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("e53: exp(x) + exp(-x)", "Exponential/Log", lambda x: np.exp(x) + np.exp(-x), x_train_range=(-1.0, 1.0), x_test_range=(1.0, 2.0))
    add("e54: tanh(x)", "Exponential/Log", lambda x: np.tanh(x))
    add("e55: 1 - exp(-x)", "Exponential/Log", lambda x: 1 - np.exp(-x))
    add("e56: exp(-2*x)", "Exponential/Log", lambda x: np.exp(-2*x))
    add("e57: log(1 + x)", "Exponential/Log", lambda x: np.log1p(x))
    add("e58: exp(cos(x))", "Exponential/Log", lambda x: np.exp(np.cos(x)))
    add("e59: 3*exp(-x^2) + 2", "Exponential/Log", lambda x: 3*np.exp(-x**2) + 2)
    add("e60: sqrt(x)*log(x)", "Exponential/Log", lambda x: np.sqrt(x)*np.log(x), x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))

    # --- Category 4: Rational & Root functions (20 problems) ---
    add("r61: 1/x", "Rational/Root", lambda x: 1.0/x, x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("r62: 1/(x + 1)", "Rational/Root", lambda x: 1.0/(x + 1.0))
    add("r63: 1/(1 + x^2)", "Rational/Root", lambda x: 1.0/(1.0 + x**2))
    add("r64: x/(1 + x^2)", "Rational/Root", lambda x: x/(1.0 + x**2))
    add("r65: 1/x^2", "Rational/Root", lambda x: 1.0/(x**2), x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("r66: sqrt(x)", "Rational/Root", lambda x: np.sqrt(x), x_train_range=(0.2, 3.0), x_test_range=(3.0, 6.0))
    add("r67: 2*sqrt(x) - 1", "Rational/Root", lambda x: 2*np.sqrt(x) - 1, x_train_range=(0.2, 3.0), x_test_range=(3.0, 6.0))
    add("r68: sqrt(1 + x^2)", "Rational/Root", lambda x: np.sqrt(1 + x**2))
    add("r69: (x - 1)/(x + 1)", "Rational/Root", lambda x: (x - 1.0)/(x + 1.0))
    add("r70: x^2 / (1 + x^2)", "Rational/Root", lambda x: (x**2)/(1.0 + x**2))
    add("r71: 1 / sqrt(x)", "Rational/Root", lambda x: 1.0/np.sqrt(x), x_train_range=(0.5, 3.0), x_test_range=(3.0, 6.0))
    add("r72: (x + 2)/x", "Rational/Root", lambda x: (x + 2.0)/x, x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("r73: 3/(2 + x)", "Rational/Root", lambda x: 3.0/(2.0 + x))
    add("r74: x*sqrt(x)", "Rational/Root", lambda x: x*np.sqrt(x), x_train_range=(0.2, 3.0), x_test_range=(3.0, 5.0))
    add("r75: 1/(x^2 + 4)", "Rational/Root", lambda x: 1.0/(x**2 + 4.0))
    add("r76: sqrt(x) + 1/x", "Rational/Root", lambda x: np.sqrt(x) + 1.0/x, x_train_range=(0.5, 3.0), x_test_range=(3.0, 5.0))
    add("r77: (2*x + 1)/(x + 2)", "Rational/Root", lambda x: (2*x + 1)/(x + 2))
    add("r78: 1/(1 + exp(-x)) [sigmoid]", "Rational/Root", lambda x: 1.0/(1.0 + np.exp(-x)))
    add("r79: x^3 / (1 + x^2)", "Rational/Root", lambda x: (x**3)/(1.0 + x**2))
    add("r80: 2/(x^3 + 1)", "Rational/Root", lambda x: 2.0/(x**3 + 1.0))

    # --- Category 5: Multi-variable & Deep Composites (20 problems) ---
    add("c81: x1 * x2", "Multi/Composite", lambda X: X[:, 0] * X[:, 1], dim=2)
    add("c82: x1^2 + x2^2", "Multi/Composite", lambda X: X[:, 0]**2 + X[:, 1]**2, dim=2)
    add("c83: 2*x1 - 3*x2 + 1", "Multi/Composite", lambda X: 2*X[:, 0] - 3*X[:, 1] + 1, dim=2)
    add("c84: sin(x1) * cos(x2)", "Multi/Composite", lambda X: np.sin(X[:, 0]) * np.cos(X[:, 1]), dim=2)
    add("c85: x1 / (1 + x2^2)", "Multi/Composite", lambda X: X[:, 0] / (1.0 + X[:, 1]**2), dim=2)
    add("c86: exp(-(x1^2 + x2^2))", "Multi/Composite", lambda X: np.exp(-(X[:, 0]**2 + X[:, 1]**2)), dim=2)
    add("c87: x1^2 * x2", "Multi/Composite", lambda X: (X[:, 0]**2) * X[:, 1], dim=2)
    add("c88: (x1 + x2)^2", "Multi/Composite", lambda X: (X[:, 0] + X[:, 1])**2, dim=2)
    add("c89: sin(x1 + x2)", "Multi/Composite", lambda X: np.sin(X[:, 0] + X[:, 1]), dim=2)
    add("c90: x1 * exp(-x2)", "Multi/Composite", lambda X: X[:, 0] * np.exp(-X[:, 1]), dim=2)
    add("c91: exp(-cos(x^2))", "Multi/Composite", lambda x: np.exp(-np.cos(x**2)))
    add("c92: sin(log(x))", "Multi/Composite", lambda x: np.sin(np.log(x)), x_train_range=(1.0, 5.0), x_test_range=(5.0, 8.0))
    add("c93: log(1 + sin(x)^2)", "Multi/Composite", lambda x: np.log(1 + np.sin(x)**2))
    add("c94: sqrt(1 + exp(-x))", "Multi/Composite", lambda x: np.sqrt(1 + np.exp(-x)))
    add("c95: cos(exp(-x))", "Multi/Composite", lambda x: np.cos(np.exp(-x)))
    add("c96: sin(x)^2 + cos(x)^2 [=1]", "Multi/Composite", lambda x: np.sin(x)**2 + np.cos(x)**2)
    add("c97: (x1*x2)^2", "Multi/Composite", lambda X: (X[:, 0]*X[:, 1])**2, dim=2)
    add("c98: exp(-x1) + sin(x2)", "Multi/Composite", lambda X: np.exp(-X[:, 0]) + np.sin(X[:, 1]), dim=2)
    add("c99: (x1^2 - x2^2)", "Multi/Composite", lambda X: X[:, 0]**2 - X[:, 1]**2, dim=2)
    add("c100: x1 / (x2 + 1)", "Multi/Composite", lambda X: X[:, 0] / (X[:, 1] + 1.0), dim=2)

    return probs

# =====================================================================
# 4. Evaluation Runner
# =====================================================================
def run_benchmark():
    problems = get_100_problems()
    print(f"Loaded {len(problems)} benchmark problems across 5 categories.")
    
    net = CodeNeuronNetwork(beam_width=4, max_depth=2, n_boost_stages=2)
    
    results = []
    t_start_total = time.time()
    
    exact_count = 0
    high_acc_count = 0
    mod_acc_count = 0
    fail_count = 0
    
    print("\n" + "="*85)
    print(f"{'Problem Name':<32} | {'Category':<15} | {'Extrap NRMSE':<14} | {'Status':<15} | {'Time(s)'}")
    print("="*85)
    
    for idx, prob in enumerate(problems, 1):
        t0 = time.time()
        dim = prob["dim"]
        
        # 1. Sample Train and Disjoint Extrapolation Test sets
        rng = np.random.default_rng(42 + idx)
        if dim == 1:
            X_tr = rng.uniform(prob["train_range"][0], prob["train_range"][1], size=(80, 1))
            X_te = rng.uniform(prob["test_range"][0], prob["test_range"][1], size=(80, 1))
            y_tr = prob["fn"](X_tr[:, 0])
            y_te = prob["fn"](X_te[:, 0])
        else:
            X_tr = rng.uniform(0.5, 2.5, size=(100, dim))
            X_te = rng.uniform(2.5, 4.0, size=(100, dim))
            y_tr = prob["fn"](X_tr)
            y_te = prob["fn"](X_te)
            
        # 2. Fit Code-Neuron Network
        stages = net.solve(X_tr, y_tr)
        elapsed = time.time() - t0
        
        # 3. Evaluate Extrapolation
        if not stages:
            status = "FAILED"
            extrap_nrmse = 999.0
            fail_count += 1
        else:
            # Reconstruct prediction on test set
            # Build features for test set
            curr_names = [f"x{i+1}" if dim > 1 else "x" for i in range(dim)]
            # We evaluate stages
            y_pred_te_total = np.zeros(len(X_te))
            
            # Helper to compute features on X_te
            def compute_stage_features(X_eval, names_eval, target_names):
                # Generates requested features by name
                pool = {names_eval[i]: X_eval[:, i] for i in range(len(names_eval))}
                # Compute up to max_depth
                for d in range(net.max_depth):
                    new_entries = {}
                    for nm, col in list(pool.items()):
                        for op_name, fn, fmt in UNARY_BLOCKS:
                            res_name = fmt.format(x=nm)
                            if res_name in target_names and res_name not in pool:
                                r = fn(col)
                                new_entries[res_name] = r
                    # Pairwise
                    items = list(pool.items())[:6]
                    for i in range(len(items)):
                        for j in range(i, len(items)):
                            p_name = f"({items[i][0]} * {items[j][0]})"
                            if p_name in target_names and p_name not in pool:
                                new_entries[p_name] = items[i][1] * items[j][1]
                    pool.update(new_entries)
                # Stack in exact order
                cols = []
                for tn in target_names:
                    if tn in pool and np.all(np.isfinite(pool[tn])):
                        cols.append(pool[tn])
                    else:
                        cols.append(np.zeros(len(X_eval)))
                return np.column_stack(cols)
                
            for st in stages:
                F_te = compute_stage_features(X_te, curr_names, st["names"])
                F_te_valid = F_te[:, st["valid"]]
                A_te = np.column_stack([F_te_valid, np.ones(len(F_te_valid))])
                y_pred_t = A_te @ st["w"]
                y_pred_te = st["t_inv"](y_pred_t)
                y_pred_te_total += np.where(np.isfinite(y_pred_te), y_pred_te, 0.0)
                
            # NRMSE on extrapolation
            denom = np.std(y_te)
            if denom < 1e-9:
                denom = np.mean(np.abs(y_te)) + 1e-6
            extrap_nrmse = np.sqrt(np.mean((y_te - y_pred_te_total)**2)) / denom
            
            if extrap_nrmse < 1e-3:
                status = "EXACT (0-error)"
                exact_count += 1
            elif extrap_nrmse < 0.05:
                status = "HIGH ACC (<5%)"
                high_acc_count += 1
            elif extrap_nrmse < 0.20:
                status = "MODERATE (<20%)"
                mod_acc_count += 1
            else:
                status = "POOR/FAILED"
                fail_count += 1
                
        results.append((prob["name"], prob["cat"], extrap_nrmse, status, elapsed))
        print(f"{prob['name']:<32} | {prob['cat']:<15} | {extrap_nrmse:<14.2e} | {status:<15} | {elapsed:.2f}s")
        
    total_time = time.time() - t_start_total
    
    print("\n" + "="*85)
    print("BENCHMARK SUMMARY (100 Mathematical Problems)")
    print("="*85)
    print(f"Total Problems:              100")
    print(f"Exact Analytical (NRMSE<1e-3): {exact_count} / 100 ({exact_count}%)")
    print(f"High Accuracy   (NRMSE<0.05): {high_acc_count} / 100 ({high_acc_count}%)")
    print(f"Moderate Acc    (NRMSE<0.20): {mod_acc_count} / 100 ({mod_acc_count}%)")
    print(f"Failed / Poor   (NRMSE>=0.20): {fail_count} / 100 ({fail_count}%)")
    print(f"Total Benchmark Runtime:     {total_time:.1f} seconds (Avg {total_time/100:.3f}s / problem)")
    print("="*85)

if __name__ == "__main__":
    run_benchmark()
