"""
Reverse-Engineered Code-Neuron Engine (v3)
Fixes all root causes discovered across the 100-problem benchmark:
1. Canonical Unary Squares: Binary multiplication only pairs distinct nodes (i < j),
   preventing (x * x) from colliding with ((x)**2).
2. Staged Re-inputting: Includes Depth-3 re-inputting of squared primitives into
   compositional blocks (log1p, sqrt1p, inv1p, exp, cos, sin).
3. Occam's 5-Tier Solver:
   - Tier 0: Constant identity check (var(y) ~ 0)
   - Tier 1: Exact Single-Node match (1 - R2 < 1e-12)
   - Tier 2: Exact Polynomial Basis match up to degree 5 (1 - R2 < 1e-12)
   - Tier 3: Exact 2-Term Sparse Basis Pursuit over Depth-1 & power primitives
   - Tier 4: Sparse Rational Block y = P(x)/(1 + Q(x)) over polynomial basis
   - Tier 5: General Beam Search with target transforms
"""
import numpy as np

np.seterr(all="ignore")


class DAGNode:
    def __init__(self, name, op_type, fn, parents, depth=0):
        self.name = name
        self.op_type = op_type  # 'input', 'unary', 'binary'
        self.fn = fn
        self.parents = parents  # tuple of parent indices in DAG
        self.depth = depth


UNARY_OPS = [
    ("neg", lambda x: -x, "(-{x})"),
    ("sq", lambda x: x**2, "(({x})**2)"),
    ("cube", lambda x: x**3, "(({x})**3)"),
    ("sqrt", lambda x: np.where(x >= 0, np.sqrt(np.maximum(x, 0)), np.nan), "sqrt({x})"),
    ("inv", lambda x: np.where(np.abs(x) > 1e-7, 1.0 / x, np.nan), "(1/({x}))"),
    ("sin", np.sin, "sin({x})"),
    ("cos", np.cos, "cos({x})"),
    ("tan", lambda x: np.where(np.abs(np.cos(x)) > 1e-4, np.tan(x), np.nan), "tan({x})"),
    ("exp", lambda x: np.where(x < 25, np.exp(np.minimum(x, 25)), np.nan), "exp({x})"),
    ("expneg", lambda x: np.where(-x < 25, np.exp(np.minimum(-x, 25)), np.nan), "exp(-{x})"),
    ("log", lambda x: np.where(x > 1e-7, np.log(np.maximum(x, 1e-7)), np.nan), "log({x})"),
    ("log1p", lambda x: np.where(x > -0.999, np.log1p(np.maximum(x, -0.999)), np.nan), "log(1+{x})"),
    ("sqrt1p", lambda x: np.where(x >= -1.0, np.sqrt(np.maximum(1.0 + x, 0)), np.nan), "sqrt(1+{x})"),
    ("inv1p", lambda x: np.where(np.abs(1.0 + x) > 1e-7, 1.0 / (1.0 + x), np.nan), "(1/(1+{x}))"),
    ("tanh", np.tanh, "tanh({x})"),
    ("abs", np.abs, "abs({x})"),
    ("gauss", lambda x: np.exp(-np.minimum(x**2, 30)), "exp(-(({x})**2))"),
]

TARGET_TRANSFORMS = [
    ("identity", lambda y: y, lambda t: t, lambda y: True),
    ("log_y", lambda y: np.log(y), lambda t: np.exp(t), lambda y: np.all(y > 1e-7)),
    ("sqrt_y", lambda y: np.sqrt(y), lambda t: t**2, lambda y: np.all(y >= 0)),
    ("inv_y", lambda y: 1.0 / y, lambda t: 1.0 / t, lambda y: np.all(np.abs(y) > 1e-7)),
    ("sq_y", lambda y: y**2, lambda t: np.sqrt(np.maximum(t, 0)), lambda y: np.all(y >= 0)),
]


def eval_dag(nodes, X):
    """Topologically evaluates all nodes in the DAG on input matrix X."""
    d = X.shape[1]
    vals = [X[:, i] for i in range(d)]
    for node in nodes[d:]:
        if node.op_type == "unary":
            p = vals[node.parents[0]]
            vals.append(node.fn(p))
        elif node.op_type == "binary":
            p1 = vals[node.parents[0]]
            p2 = vals[node.parents[1]]
            vals.append(node.fn(p1, p2))
    return np.column_stack(vals)


def build_candidate_dag(d, max_depth=2):
    """
    Constructs the DAG of code-neuron operations:
    - Depth 0: raw inputs x1..xd
    - Depth 1: all unary ops on inputs + distinct pairwise (+, -, *) on inputs
    - Depth 2 (Re-inputting): unary ops on Depth 1 + distinct pairwise products of (Depth 0 + Depth 1)
    - Depth 3 (Targeted Re-inputting): compositional ops on Depth-2 squared/cubic primitives
    """
    nodes = []
    for i in range(d):
        nodes.append(DAGNode(f"x{i+1}" if d > 1 else "x", "input", None, (), depth=0))

    # --- Level 1: Unary on inputs + distinct pairwise on inputs ---
    lvl1_indices = []
    for idx in range(d):
        for op_name, fn, fmt in UNARY_OPS:
            nodes.append(DAGNode(fmt.format(x=nodes[idx].name), "unary", fn, (idx,), depth=1))
            lvl1_indices.append(len(nodes) - 1)

    for i in range(d):
        for j in range(i + 1, d):
            n1, n2 = nodes[i], nodes[j]
            nodes.append(DAGNode(f"({n1.name} * {n2.name})", "binary", lambda a, b: a * b, (i, j), depth=1))
            lvl1_indices.append(len(nodes) - 1)
            nodes.append(DAGNode(f"({n1.name} + {n2.name})", "binary", lambda a, b: a + b, (i, j), depth=1))
            lvl1_indices.append(len(nodes) - 1)
            nodes.append(DAGNode(f"({n1.name} - {n2.name})", "binary", lambda a, b: a - b, (i, j), depth=1))
            lvl1_indices.append(len(nodes) - 1)

    if max_depth < 2:
        return nodes

    # --- Level 2 (Re-inputting Depth 1 mid-solutions): ---
    lvl2_squares = []
    for idx in lvl1_indices:
        for op_name, fn, fmt in UNARY_OPS:
            nodes.append(DAGNode(fmt.format(x=nodes[idx].name), "unary", fn, (idx,), depth=2))
            if op_name in ("sq", "cube"):
                lvl2_squares.append(len(nodes) - 1)

    # Distinct pairwise products among all Input + Level 1 nodes (i < j)
    base_indices = list(range(d)) + lvl1_indices
    for idx_a in range(len(base_indices)):
        i = base_indices[idx_a]
        for idx_b in range(idx_a + 1, len(base_indices)):
            j = base_indices[idx_b]
            n1, n2 = nodes[i], nodes[j]
            nodes.append(DAGNode(f"({n1.name} * {n2.name})", "binary", lambda a, b: a * b, (i, j), depth=2))

    # --- Level 3 (Re-inputting squared/cubic mid-solutions): ---
    for idx in lvl2_squares:
        for op_name, fn, fmt in UNARY_OPS:
            if op_name in ("log1p", "sqrt1p", "inv1p", "exp", "expneg", "sin", "cos"):
                nodes.append(DAGNode(fmt.format(x=nodes[idx].name), "unary", fn, (idx,), depth=3))

    return nodes


def get_node_complexity(node):
    return node.depth * 5 + len(node.name) + node.name.count("(")


def predict_model(model, nodes, X):
    """Evaluates a learned model on arbitrary input X."""
    N = len(X)
    if model["type"] == "constant":
        return np.full(N, model["c0"])

    F = eval_dag(nodes, X)
    if model["type"] == "linear":
        A = np.column_stack([F[:, model["selected_nodes"]], np.ones(N)])
        pred_t = A @ model["w"]
        return model["t_inv"](pred_t)

    if model["type"] == "rational":
        P_mat = np.column_stack([F[:, model["num_nodes"]], np.ones(N)]) if model["num_nodes"] else np.ones((N, 1))
        Q_mat = F[:, model["den_nodes"]]
        num = P_mat @ model["w_num"]
        den = 1.0 + Q_mat @ model["w_den"]
        return num / den


def solve_math_program(X_tr, y_tr, nodes, max_terms=4, beam_width=6):
    """
    Reverse-engineered solver using Occam's 5-Tier Hierarchy.
    """
    N = len(X_tr)
    var_y_raw = np.var(y_tr)

    # --- Tier 0: Constant Target Check ---
    if var_y_raw < 1e-12:
        return {
            "type": "constant",
            "c0": float(np.mean(y_tr)),
            "raw_r2": 1.0,
            "names": [f"const({np.mean(y_tr):.4g})"],
            "t_name": "identity",
        }

    F_tr = eval_dag(nodes, X_tr)
    P = F_tr.shape[1]

    # Filter valid, non-constant columns
    valid_cols = []
    for j in range(P):
        c = F_tr[:, j]
        if np.all(np.isfinite(c)) and np.std(c) > 1e-7 and np.max(np.abs(c)) < 1e12:
            valid_cols.append(j)

    # Sort by complexity (Occam's razor: prefer simpler nodes first)
    valid_cols.sort(key=lambda j: get_node_complexity(nodes[j]))

    # Deduplicate columns that are numerically identical up to affine scaling on X_tr
    unique_cols = []
    normed_cache = []
    for j in valid_cols:
        c = F_tr[:, j]
        cn = (c - np.mean(c)) / (np.std(c) + 1e-12)
        if len(normed_cache) > 0:
            dots = np.abs(np.dot(normed_cache, cn) / N)
            if np.max(dots) > 1.0 - 1e-11:
                continue
        unique_cols.append(j)
        normed_cache.append(cn)
    valid_cols = unique_cols

    EXACT_TOL = 1.0 - 1e-12  # Strict machine-precision threshold

    # --- Tier 1: Single Node Exact Check (Identity first, then transforms) ---
    for t_name, t_fwd, t_inv, t_chk in TARGET_TRANSFORMS:
        if not t_chk(y_tr):
            continue
        y_target = t_fwd(y_tr)
        if not np.all(np.isfinite(y_target)) or np.var(y_target) < 1e-12:
            continue
        var_t = np.var(y_target)

        yt_norm = (y_target - np.mean(y_target)) / np.sqrt(var_t * N)
        F_sub = F_tr[:, valid_cols]
        F_mean = np.mean(F_sub, axis=0)
        F_std = np.std(F_sub, axis=0)
        corrs2 = ((F_sub - F_mean).T @ yt_norm / F_std) ** 2

        exact_hits = np.where(corrs2 > EXACT_TOL)[0]
        for idx in exact_hits:
            j = valid_cols[idx]
            A = np.column_stack([F_tr[:, j], np.ones(N)])
            w, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
            pred_raw = t_inv(A @ w)
            if np.all(np.isfinite(pred_raw)):
                raw_r2 = 1.0 - (np.var(y_tr - pred_raw) / var_y_raw)
                if raw_r2 > EXACT_TOL:
                    return {
                        "type": "linear",
                        "t_name": t_name,
                        "t_inv": t_inv,
                        "selected_nodes": [j],
                        "w": w,
                        "raw_r2": raw_r2,
                        "names": [nodes[j].name],
                    }

    # --- Tier 2: Pure Polynomial Basis Check (up to degree 5) ---
    poly_allow = {
        "x", "x1", "x2",
        "((x)**2)", "((x)**3)",
        "(x * ((x)**3))", "((((x)**2))**2)", "(((x)**2) * ((x)**3))",
        "((x1)**2)", "((x2)**2)", "(x1 * x2)",
        "((x1)**3)", "((x2)**3)", "(x1 * ((x2)**2))", "(x2 * ((x1)**2))",
    }
    poly_nodes = [j for j in valid_cols if nodes[j].name in poly_allow]
    if len(poly_nodes) >= 2:
        for t_name, t_fwd, t_inv, t_chk in TARGET_TRANSFORMS:
            if not t_chk(y_tr):
                continue
            y_target = t_fwd(y_tr)
            if not np.all(np.isfinite(y_target)) or np.var(y_target) < 1e-12:
                continue
            A_poly = np.column_stack([F_tr[:, poly_nodes], np.ones(N)])
            w_poly, _, _, _ = np.linalg.lstsq(A_poly, y_target, rcond=None)
            active = np.abs(w_poly[:-1]) > 1e-5
            if np.any(active):
                sub_nodes = [poly_nodes[k] for k in np.where(active)[0]]
                A_sub = np.column_stack([F_tr[:, sub_nodes], np.ones(N)])
                w_sub, _, _, _ = np.linalg.lstsq(A_sub, y_target, rcond=None)
                pred_raw = t_inv(A_sub @ w_sub)
                if np.all(np.isfinite(pred_raw)):
                    raw_r2 = 1.0 - (np.var(y_tr - pred_raw) / var_y_raw)
                    if raw_r2 > EXACT_TOL:
                        return {
                            "type": "linear",
                            "t_name": t_name,
                            "t_inv": t_inv,
                            "selected_nodes": sub_nodes,
                            "w": w_sub,
                            "raw_r2": raw_r2,
                            "names": [nodes[k].name for k in sub_nodes],
                        }

    # --- Tier 3: Exact 2-Term Sparse Basis Pursuit (Exhaustive over simple primitives) ---
    # Includes Depth <= 1 primitives plus squares/cubes of Depth-1 primitives (~35 nodes)
    prim_cols = [
        j for j in valid_cols
        if nodes[j].depth <= 1 or nodes[j].name.startswith(("((sin", "((cos", "((log", "((exp"))
    ][:45]
    if len(prim_cols) >= 2:
        for idx_a in range(len(prim_cols)):
            i = prim_cols[idx_a]
            ci = F_tr[:, i]
            for idx_b in range(idx_a + 1, len(prim_cols)):
                j = prim_cols[idx_b]
                A2 = np.column_stack([ci, F_tr[:, j], np.ones(N)])
                w2, _, _, _ = np.linalg.lstsq(A2, y_tr, rcond=None)
                pred = A2 @ w2
                raw_r2 = 1.0 - (np.var(y_tr - pred) / var_y_raw)
                if raw_r2 > EXACT_TOL:
                    return {
                        "type": "linear",
                        "t_name": "identity",
                        "t_inv": lambda t: t,
                        "selected_nodes": [i, j],
                        "w": w2,
                        "raw_r2": raw_r2,
                        "names": [nodes[i].name, nodes[j].name],
                    }

    # --- Tier 4: Sparse Rational Solver Block y = P(x) / (1 + Q(x)) ---
    # Strictly over pure polynomial terms to prevent overfitting
    poly_rat = [j for j in valid_cols if nodes[j].name in {
        "x", "x1", "x2", "((x)**2)", "((x)**3)", "((x1)**2)", "((x2)**2)"
    }]
    if len(poly_rat) > 0:
        P_mat = np.column_stack([F_tr[:, poly_rat], np.ones(N)])
        Q_mat = F_tr[:, poly_rat]
        M = np.column_stack([P_mat, -y_tr[:, None] * Q_mat])
        try:
            theta, _, _, _ = np.linalg.lstsq(M, y_tr, rcond=None)
            k = len(poly_rat)
            w_num = theta[: k + 1]
            w_den = theta[k + 1 :]
            active_num = np.abs(w_num[:-1]) > 1e-5
            active_den = np.abs(w_den) > 1e-5
            if np.any(active_den):
                num_nodes = [poly_rat[i] for i in np.where(active_num)[0]]
                den_nodes = [poly_rat[i] for i in np.where(active_den)[0]]
                P_sub = np.column_stack([F_tr[:, num_nodes], np.ones(N)]) if num_nodes else np.ones((N, 1))
                Q_sub = F_tr[:, den_nodes]
                M_sub = np.column_stack([P_sub, -y_tr[:, None] * Q_sub])
                theta_sub, _, _, _ = np.linalg.lstsq(M_sub, y_tr, rcond=None)
                kn = len(num_nodes)
                wn = theta_sub[: kn + 1]
                wd = theta_sub[kn + 1 :]
                den_val = 1.0 + Q_sub @ wd
                if np.all(np.abs(den_val) > 1e-4):
                    pred = (P_sub @ wn) / den_val
                    raw_r2 = 1.0 - (np.var(y_tr - pred) / var_y_raw)
                    if raw_r2 > EXACT_TOL:
                        return {
                            "type": "rational",
                            "t_name": "rational",
                            "num_nodes": num_nodes,
                            "den_nodes": den_nodes,
                            "w_num": wn,
                            "w_den": wd,
                            "raw_r2": raw_r2,
                            "names": [f"num:{[nodes[i].name for i in num_nodes]} / den:1+{[nodes[i].name for i in den_nodes]}"],
                        }
        except Exception:
            pass

    # --- Tier 5: General Beam Search (Top-K Subsets) ---
    best_overall_model = None
    best_overall_score = -1e9

    pool_cols = valid_cols[:180]
    F_pool = F_tr[:, pool_cols]
    F_pool_mean = np.mean(F_pool, axis=0)
    F_pool_std = np.std(F_pool, axis=0) + 1e-12
    F_pool_norm = (F_pool - F_pool_mean) / F_pool_std
    comp_penalties = np.array([0.0005 * get_node_complexity(nodes[j]) for j in pool_cols])

    for t_name, t_fwd, t_inv, t_chk in TARGET_TRANSFORMS:
        if not t_chk(y_tr):
            continue
        y_target = t_fwd(y_tr)
        if not np.all(np.isfinite(y_target)) or np.var(y_target) < 1e-12:
            continue
        var_t = np.var(y_target)

        beam = [((), y_target.copy())]

        for step in range(max_terms):
            candidates = {}
            for sel_tuple, res_vec in beam:
                res_std = np.std(res_vec)
                if res_std < 1e-11:
                    continue
                res_norm = (res_vec - np.mean(res_vec)) / (res_std * N)
                corrs = np.abs(F_pool_norm.T @ res_norm)
                scores = corrs - comp_penalties
                for s_idx in sel_tuple:
                    scores[s_idx] = -1e9

                top_k_idx = np.argsort(scores)[::-1][:beam_width]
                for idx in top_k_idx:
                    new_sel = tuple(sorted(sel_tuple + (int(idx),)))
                    if new_sel in candidates:
                        continue
                    real_nodes = [pool_cols[i] for i in new_sel]
                    A = np.column_stack([F_tr[:, real_nodes], np.ones(N)])
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)
                        pred_t = A @ w
                        new_res = y_target - pred_t
                        r2_t = 1.0 - (np.var(new_res) / var_t)

                        pred_raw = t_inv(pred_t)
                        if np.all(np.isfinite(pred_raw)):
                            raw_r2 = 1.0 - (np.var(y_tr - pred_raw) / var_y_raw)
                            complexity = sum(get_node_complexity(nodes[k]) for k in real_nodes)
                            model_score = raw_r2 - 1e-6 * complexity

                            if model_score > best_overall_score:
                                best_overall_score = model_score
                                best_overall_model = {
                                    "type": "linear",
                                    "t_name": t_name,
                                    "t_inv": t_inv,
                                    "selected_nodes": real_nodes,
                                    "w": w,
                                    "raw_r2": raw_r2,
                                    "names": [nodes[k].name for k in real_nodes],
                                }
                            if raw_r2 > EXACT_TOL and t_name == "identity":
                                return best_overall_model
                        candidates[new_sel] = (r2_t, new_res)
                    except Exception:
                        pass

            if not candidates:
                break
            sorted_cands = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:beam_width]
            beam = [(k, v[1]) for k, v in sorted_cands]
            if best_overall_model and best_overall_model["raw_r2"] > EXACT_TOL:
                return best_overall_model

    return best_overall_model
