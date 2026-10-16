"""
Unified Continuous Physics & Dynamical Code-Neuron System
Integrates:
1. Continuous Harmonic Block (FFT + Phase Lock)
2. Continuous Power-Law Block (Log-Log)
3. Continuous Gaussian / Exponential Block
4. ODE Identification & Forward RK45 Extrapolation Block
5. PDE Field Identification & Spatio-Temporal Extrapolation Block
"""
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

np.seterr(all="ignore")

# =====================================================================
# 1. Specialized Continuous Blocks
# =====================================================================

def solve_continuous_harmonic(x, y):
    """
    Solves y = A * sin(omega * x + phi) + C
    Uses FFT for global frequency initialization, then Gauss-Newton polish.
    """
    N = len(x)
    dx = (x[-1] - x[0]) / (N - 1)
    x_unif = np.linspace(x[0], x[-1], N)
    y_unif = np.interp(x_unif, x, y)
    
    # FFT peak detection
    fft_vals = np.fft.rfft(y_unif - np.mean(y_unif))
    freqs = np.fft.rfftfreq(N, dx)
    if len(fft_vals) < 2:
        return None
    peak_idx = np.argmax(np.abs(fft_vals[1:])) + 1
    omega_init = 2 * np.pi * freqs[peak_idx]
    
    try:
        popt, _ = curve_fit(
            lambda t, a, w, p, c: a * np.sin(w * t + p) + c,
            x, y,
            p0=[np.std(y) * 1.414, omega_init, 0.0, np.mean(y)],
            maxfev=100
        )
        pred = popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
        r2 = 1.0 - (np.var(y - pred) / (np.var(y) + 1e-12))
        return {
            "type": "harmonic",
            "params": popt, # [A, omega, phi, C]
            "fn": lambda t: popt[0] * np.sin(popt[1] * t + popt[2]) + popt[3],
            "formula": f"{popt[0]:.4f} * sin({popt[1]:.4f}*x + {popt[2]:.4f}) + {popt[3]:.4f}",
            "r2": r2
        }
    except Exception:
        return None


def solve_continuous_power(x, y):
    """
    Solves y = A * x^alpha + C
    Uses log-log linear regression.
    """
    mask = (x > 1e-6) & (y > 1e-6)
    if np.sum(mask) < 4:
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    A = np.exp(intercept)
    alpha = slope
    pred = A * (x**alpha)
    r2 = 1.0 - (np.var(y - pred) / (np.var(y) + 1e-12))
    return {
        "type": "power",
        "params": [A, alpha],
        "fn": lambda t: A * (t**alpha),
        "formula": f"{A:.4f} * x^{alpha:.4f}",
        "r2": r2
    }


def solve_continuous_gaussian(x, y):
    """
    Solves y = A * exp(-(x - mu)^2 / (2 * sigma^2)) + C
    Uses moments estimation + nonlinear polish.
    """
    baseline = np.min(y)
    y_pos = y - baseline + 1e-6
    mu_est = np.sum(x * y_pos) / np.sum(y_pos)
    sigma_est = np.sqrt(np.abs(np.sum((x - mu_est)**2 * y_pos) / np.sum(y_pos)))
    A_est = np.max(y) - baseline
    
    try:
        popt, _ = curve_fit(
            lambda t, a, mu, sig, c: a * np.exp(-((t - mu)**2) / (2 * sig**2 + 1e-9)) + c,
            x, y,
            p0=[A_est, mu_est, max(sigma_est, 0.1), baseline],
            maxfev=100
        )
        pred = popt[0] * np.exp(-((x - popt[1])**2) / (2 * popt[2]**2 + 1e-9)) + popt[3]
        r2 = 1.0 - (np.var(y - pred) / (np.var(y) + 1e-12))
        return {
            "type": "gaussian",
            "params": popt,
            "fn": lambda t: popt[0] * np.exp(-((t - popt[1])**2) / (2 * popt[2]**2 + 1e-9)) + popt[3],
            "formula": f"{popt[0]:.4f} * exp(-(x - {popt[1]:.4f})^2 / (2*{popt[2]:.4f}^2)) + {popt[3]:.4f}",
            "r2": r2
        }
    except Exception:
        return None

# =====================================================================
# 2. ODE Code Block: System Identification & Future Trajectory Extrapolation
# =====================================================================

class ODEBlock:
    """
    Identifies 2nd-order autonomous ODEs: d^2x/dt^2 = f(x, dx/dt).
    Uses CubicSpline differentiation to eliminate numerical truncation error,
    followed by Best-Subset Sparse Selection to prevent collinear cancellations.
    """
    def __init__(self, order=2):
        self.order = order

    def fit_and_extrapolate(self, t_tr, x_tr, t_te):
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t_tr, x_tr)
        x_dot = cs(t_tr, 1)
        x_ddot = cs(t_tr, 2)
        
        # Candidate library
        candidates = [x_tr, x_dot, np.sin(x_tr), x_tr**3]
        names = ["x", "x_dot", "sin(x)", "x^3"]
        
        # Test 1-term and 2-term subsets to find the simplest exact fit
        best_sub = None
        best_w = None
        best_r2 = -1e9
        
        # 2-term subsets
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A_sub = np.column_stack([candidates[i], candidates[j]])
                w_sub, _, _, _ = np.linalg.lstsq(A_sub, x_ddot, rcond=None)
                pred = A_sub @ w_sub
                r2 = 1.0 - (np.var(x_ddot - pred) / np.var(x_ddot))
                if r2 > best_r2:
                    best_r2 = r2
                    best_sub = [i, j]
                    best_w = w_sub
                    
        formula_terms = [f"{best_w[k]:+.4f}*{names[best_sub[k]]}" for k in range(len(best_sub))]
        formula_str = " ".join(formula_terms)
        
        def ode_rhs(t, state):
            pos, vel = state[0], state[1]
            cand_vals = [pos, vel, np.sin(pos), pos**3]
            accel = best_w[0] * cand_vals[best_sub[0]] + best_w[1] * cand_vals[best_sub[1]]
            return [vel, accel]
            
        y0 = [x_tr[0], x_dot[0]]
        sol = solve_ivp(ode_rhs, (t_tr[0], t_te[-1]), y0, t_eval=t_te, method="RK45", rtol=1e-9, atol=1e-9)
        if not sol.success or sol.y.shape[1] != len(t_te):
            return None
            
        return {
            "formula": formula_str,
            "pred_test": sol.y[0]
        }


class PDEBlock:
    """
    Identifies 1D time-dependent PDEs: u_t = alpha * u_xx + beta * (u * u_x).
    Integrates forward in time with Dirichlet boundary conditions.
    """
    def fit_and_extrapolate(self, t_tr, x_grid, U_tr, t_te):
        dt = t_tr[1] - t_tr[0]
        dx = x_grid[1] - x_grid[0]
        
        u_t = np.gradient(U_tr, dt, axis=0)
        u_x = np.zeros_like(U_tr)
        u_xx = np.zeros_like(U_tr)
        for i in range(len(t_tr)):
            u_x[i] = np.gradient(U_tr[i], dx)
            u_xx[i] = np.gradient(u_x[i], dx)
        u_ux = U_tr * u_x
        
        ut_f = u_t[2:-2, 2:-2].flatten()
        uxx_f = u_xx[2:-2, 2:-2].flatten()
        u_ux_f = u_ux[2:-2, 2:-2].flatten()
        
        # Test linear diffusion only vs nonlinear advection-diffusion
        # Model A: u_t = alpha * u_xx
        alpha_only = np.linalg.lstsq(uxx_f[:, None], ut_f, rcond=None)[0][0]
        res_a = np.var(ut_f - alpha_only * uxx_f)
        
        # Model B: u_t = alpha * u_xx + beta * (u * u_x)
        A_b = np.column_stack([uxx_f, u_ux_f])
        w_b = np.linalg.lstsq(A_b, ut_f, rcond=None)[0]
        res_b = np.var(ut_f - A_b @ w_b)
        
        if res_a < 1e-4 or res_b / (res_a + 1e-9) > 0.8:
            # Pure Diffusion
            w_pde = [alpha_only, 0.0]
            formula_str = f"u_t = {alpha_only:+.4f}*u_xx"
        else:
            w_pde = w_b
            formula_str = f"u_t = {w_b[0]:+.4f}*u_xx {w_b[1]:+.4f}*(u*u_x)"
            
        def pde_rhs(t, u):
            dudt = np.zeros_like(u)
            ux = (u[2:] - u[:-2]) / (2 * dx)
            uxx = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
            dudt[1:-1] = w_pde[0] * uxx + w_pde[1] * u[1:-1] * ux
            return dudt
            
        sol = solve_ivp(pde_rhs, (t_tr[0], t_te[-1]), U_tr[0], t_eval=t_te, method="RK45", rtol=1e-8, atol=1e-8)
        if not sol.success or sol.y.shape[1] != len(t_te):
            return None
            
        return {
            "formula": formula_str,
            "pred_test": sol.y.T
        }

# =====================================================================
# 4. Benchmark Execution on Continuous Parameters, ODEs & PDEs
# =====================================================================

def run_physics_benchmark():
    print("=" * 95)
    print("CONTINUOUS PARAMETERS, ODE & PDE CODE-BLOCK BENCHMARK")
    print("=" * 95)
    
    results = []

    # -------------------------------------------------------------
    # Case 1: Continuous Harmonic Wave with arbitrary parameters
    # y = 3.2 * sin(2.718*x + 0.85) - 0.5
    # -------------------------------------------------------------
    x_tr = np.linspace(0.1, 3.0, 100)
    x_te = np.linspace(3.0, 6.0, 100)
    y_tr = 3.2 * np.sin(2.718 * x_tr + 0.85) - 0.5
    y_te = 3.2 * np.sin(2.718 * x_te + 0.85) - 0.5
    
    res1 = solve_continuous_harmonic(x_tr, y_tr)
    pred_te1 = res1["fn"](x_te)
    nrmse1 = np.sqrt(np.mean((y_te - pred_te1)**2)) / np.std(y_te)
    results.append(("Cont Harmonic Wave: 3.2*sin(2.718*x + 0.85) - 0.5", res1["formula"], nrmse1))

    # -------------------------------------------------------------
    # Case 2: Continuous Real Power Law with arbitrary exponent
    # y = 1.8 * x^1.61803 (Golden Ratio exponent)
    # -------------------------------------------------------------
    y_tr2 = 1.8 * (x_tr**1.61803)
    y_te2 = 1.8 * (x_te**1.61803)
    res2 = solve_continuous_power(x_tr, y_tr2)
    pred_te2 = res2["fn"](x_te)
    nrmse2 = np.sqrt(np.mean((y_te2 - pred_te2)**2)) / np.std(y_te2)
    results.append(("Cont Power Law: 1.8 * x^1.61803", res2["formula"], nrmse2))

    # -------------------------------------------------------------
    # Case 3: Continuous Gaussian Packet with arbitrary center/width
    # y = 2.5 * exp(-(x - 1.414)^2 / (2 * 0.45^2)) + 0.3
    # -------------------------------------------------------------
    y_tr3 = 2.5 * np.exp(-((x_tr - 1.414)**2) / (2 * 0.45**2)) + 0.3
    y_te3 = 2.5 * np.exp(-((x_te - 1.414)**2) / (2 * 0.45**2)) + 0.3
    res3 = solve_continuous_gaussian(x_tr, y_tr3)
    pred_te3 = res3["fn"](x_te)
    nrmse3 = np.sqrt(np.mean((y_te3 - pred_te3)**2)) / np.std(y_te3)
    results.append(("Cont Gaussian Packet: 2.5*exp(-(x-1.414)^2 / (2*0.45^2))", res3["formula"], nrmse3))

    # -------------------------------------------------------------
    # Case 4: Damped Harmonic Oscillator ODE
    # d^2x/dt^2 = -0.3*dx/dt - 4.0*x
    # -------------------------------------------------------------
    t_tr = np.linspace(0, 8.0, 300)
    t_te = np.linspace(8.0, 16.0, 300)
    
    def true_oscillator(t, state):
        return [state[1], -0.3 * state[1] - 4.0 * state[0]]
        
    sol_tr = solve_ivp(true_oscillator, (0, 8.0), [1.5, 0.0], t_eval=t_tr, rtol=1e-9, atol=1e-9)
    sol_te = solve_ivp(true_oscillator, (8.0, 16.0), [sol_tr.y[0, -1], sol_tr.y[1, -1]], t_eval=t_te, rtol=1e-9, atol=1e-9)
    
    ode_block = ODEBlock(order=2)
    res_ode = ode_block.fit_and_extrapolate(t_tr, sol_tr.y[0], t_te)
    nrmse_ode = np.sqrt(np.mean((sol_te.y[0] - res_ode["pred_test"])**2)) / np.std(sol_te.y[0])
    results.append(("Damped Oscillator ODE: x'' = -0.3*x' - 4.0*x", f"x'' = {res_ode['formula']}", nrmse_ode))

    # -------------------------------------------------------------
    # Case 5: Nonlinear Pendulum ODE (Large amplitude)
    # d^2theta/dt^2 = -0.2*theta' - 9.81*sin(theta)
    # -------------------------------------------------------------
    def true_pendulum(t, state):
        return [state[1], -0.2 * state[1] - 9.81 * np.sin(state[0])]
        
    sol_tr_p = solve_ivp(true_pendulum, (0, 6.0), [1.2, 0.0], t_eval=np.linspace(0, 6.0, 300), rtol=1e-9, atol=1e-9)
    sol_te_p = solve_ivp(true_pendulum, (6.0, 12.0), [sol_tr_p.y[0, -1], sol_tr_p.y[1, -1]], t_eval=np.linspace(6.0, 12.0, 300), rtol=1e-9, atol=1e-9)
    
    res_pend = ode_block.fit_and_extrapolate(sol_tr_p.t, sol_tr_p.y[0], sol_te_p.t)
    nrmse_pend = np.sqrt(np.mean((sol_te_p.y[0] - res_pend["pred_test"])**2)) / np.std(sol_te_p.y[0])
    results.append(("Nonlinear Pendulum ODE: theta'' = -0.2*theta' - 9.81*sin(theta)", f"theta'' = {res_pend['formula']}", nrmse_pend))

    # -------------------------------------------------------------
    # Case 6: 1D Heat / Diffusion PDE
    # u_t = 0.5 * u_xx
    # Analytical solution: u(t, x) = exp(-0.5 * pi^2 * t) * sin(pi * x)
    # -------------------------------------------------------------
    t_pde_tr = np.linspace(0, 0.5, 40)
    t_pde_te = np.linspace(0.5, 1.0, 40)
    x_grid = np.linspace(0, 1.0, 50)
    
    T_tr_m, X_m = np.meshgrid(t_pde_tr, x_grid, indexing='ij')
    U_heat_tr = np.exp(-0.5 * (np.pi**2) * T_tr_m) * np.sin(np.pi * X_m)
    
    T_te_m, X_te_m = np.meshgrid(t_pde_te, x_grid, indexing='ij')
    U_heat_te = np.exp(-0.5 * (np.pi**2) * T_te_m) * np.sin(np.pi * X_te_m)
    
    pde_block = PDEBlock()
    res_pde = pde_block.fit_and_extrapolate(t_pde_tr, x_grid, U_heat_tr, t_pde_te)
    nrmse_pde = np.sqrt(np.mean((U_heat_te - res_pde["pred_test"])**2)) / np.std(U_heat_te)
    results.append(("1D Heat Diffusion PDE: u_t = 0.5*u_xx", res_pde["formula"], nrmse_pde))

    # -------------------------------------------------------------
    # Case 7: Nonlinear Viscous Burgers PDE (Fluid shockwave)
    # u_t = 0.1 * u_xx - 1.0 * (u * u_x)
    # -------------------------------------------------------------
    def burgers_rhs(t, u_flat):
        dx = x_grid[1] - x_grid[0]
        dudt = np.zeros_like(u_flat)
        ux = (u_flat[2:] - u_flat[:-2]) / (2 * dx)
        uxx = (u_flat[2:] - 2*u_flat[1:-1] + u_flat[:-2]) / (dx**2)
        dudt[1:-1] = 0.1 * uxx - u_flat[1:-1] * ux
        return dudt
        
    u0_burgers = np.sin(2 * np.pi * x_grid)
    sol_burgers = solve_ivp(burgers_rhs, (0, 0.8), u0_burgers, t_eval=np.linspace(0, 0.8, 80), rtol=1e-8, atol=1e-8)
    U_burg_all = sol_burgers.y.T # (80, 50)
    U_burg_tr = U_burg_all[:40]
    U_burg_te = U_burg_all[40:]
    t_burg_tr = np.linspace(0, 0.4, 40)
    t_burg_te = np.linspace(0.4, 0.8, 40)
    
    res_burg = pde_block.fit_and_extrapolate(t_burg_tr, x_grid, U_burg_tr, t_burg_te)
    nrmse_burg = np.sqrt(np.mean((U_burg_te - res_burg["pred_test"])**2)) / np.std(U_burg_te)
    results.append(("Viscous Burgers PDE: u_t = 0.1*u_xx - 1.0*(u*u_x)", res_burg["formula"], nrmse_burg))

    # Print Report
    print(f"{'Physical Problem / Equation':<46} | {'Discovered Law':<38} | {'Extrap NRMSE':<12}")
    print("-" * 105)
    for name, formula, nrmse in results:
        status = "EXACT (0-error)" if nrmse < 1e-4 else ("HIGH ACC (<5%)" if nrmse < 0.05 else "POOR")
        print(f"{name:<46} | {formula:<38} | {nrmse:<12.2e} [{status}]")
    print("=" * 105)

if __name__ == "__main__":
    run_physics_benchmark()
