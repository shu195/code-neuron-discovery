# Code-Neuron Discovery

### Real-Time Symbolic Regression and System Identification via Directed Acyclic Functional Graphs

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C?logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Architecture](https://img.shields.io/badge/Instruction%20Set-x86__64%20%2F%20AVX2-555555)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Inferring closed-form governing equations from discrete observational data—known as symbolic regression and nonlinear system identification—remains a fundamental objective across physical, biological, and engineering sciences. Conventional paradigms frequently present structural trade-offs: continuous neural approximations (e.g., Multilayer Perceptrons, Physics-Informed Neural Networks) act as opaque parameterized approximators with limited out-of-domain extrapolation guarantees, whereas genetic programming methods often require substantial computational budgets and stochastic search heuristics.

**Code-Neuron Discovery** provides a deterministic, graph-based framework for symbolic system identification implemented in pure C++17. The system represents candidate functional spaces as Directed Acyclic Graphs (DAGs) of elementary mathematical operators with depth-controlled compositional re-evaluation. Rather than relying on unconstrained global optimization, model selection is performed via an Occam-ordered hierarchical solver that systematically evaluates hypotheses from lowest to highest structural complexity.

The framework supports closed-form algebraic discovery, numerical differentiation and integration, second-order autonomous ordinary differential equation (ODE) parameterization, and spatio-temporal partial differential equation (PDE) identification from observational fields.

---

## Methodological Architecture

```
Observational Dataset (X, y)
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Topological Functional Graph (DAG)                    │
│  - Unary Basis:   sin, cos, exp, log, sqrt, tanh, ...  │
│  - Binary Basis:  +, -, *                              │
│  - Compositions:  Nested re-inputting up to Depth 3    │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Hierarchical Occam Model Selection Engine            │
│  ├─ Tier 0: Invariant Constant Detection               │
│  ├─ Tier 1: Collinear Transform Matching               │
│  ├─ Tier 2: Polynomial Sub-Basis Pursuit (deg <= 5)   │
│  ├─ Tier 3: Two-Term Sparse Basis Pursuit             │
│  └─ Tier 4: Sparse Rational Identification P(x)/Q(x)   │
└────────────────────────────────────────────────────────┘
             │
             ▼
Discovered Governing Equation & Out-of-Distribution Validation
```

### 1. Functional Graph Representation
Let $\mathbf{x} \in \mathbb{R}^d$ denote the input vector. A computational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is constructed where terminal input nodes $\mathcal{V}_0 = \{x_1, \dots, x_d\}$ are recursively composed through:
* **Unary Operators**: $\mathcal{U} = \{ -u, u^2, u^3, \sqrt{u}, \frac{1}{u}, \sin(u), \cos(u), \tan(u), \exp(u), \exp(-u), \ln(u), \ln(1+u), \sqrt{1+u}, \frac{1}{1+u}, \tanh(u), |u|, \exp(-u^2) \}$
* **Binary Operators**: $\mathcal{B} = \{ u_i + u_j, \, u_i - u_j, \, u_i \cdot u_j \}$

Compositional depth is bounded ($D \le 3$), yielding a structured dictionary of candidate basis functions evaluated over the observation domain without combinatorial explosion.

### 2. Hierarchical Occam Model Selection
To avoid overfitting and enforce structural parsimony, models are evaluated through a prioritized sequence:

1. **Tier 0 (Zero-Order Invariance)**: Evaluates whether sample variance satisfies $\mathrm{Var}(y) < \epsilon_{\mathrm{const}}$. If satisfied, $y = \bar{y}$ is selected.
2. **Tier 1 (Single-Node Collinearity & Coordinate Inversion)**: Evaluates linear alignment between each basis node $u_j \in \mathcal{V}$ and transformed targets $\phi_k(y)$ where $\phi_k \in \{\mathrm{id}, \ln, \sqrt{\cdot}, (\cdot)^{-1}, (\cdot)^2\}$:
   $$R^2 = \frac{\mathrm{Cov}(u_j, \phi_k(y))^2}{\mathrm{Var}(u_j)\mathrm{Var}(\phi_k(y))} \ge 1 - \delta$$
3. **Tier 2 (Polynomial Sub-Basis Pursuit)**: Evaluates regularized least-squares models over monomial sub-dictionaries up to degree 5.
4. **Tier 3 (Two-Term Sparse Basis Pursuit)**: Searches pairwise linear combinations $\beta_1 u_i + \beta_2 u_j + \beta_0$ over complexity-ordered primitive graph nodes.
5. **Tier 4 (Sparse Rational Solvers)**: Recovers rational structures of the form $y = \frac{P(\mathbf{x})}{1 + Q(\mathbf{x})}$ by linearizing the relationship $y \cdot (1 + Q(\mathbf{x})) = P(\mathbf{x})$ and solving via constrained least squares with post-hoc coefficient pruning.

---

## Empirical Benchmark Evaluation

The methodology was evaluated against a standardized 100-problem benchmark covering five functional categories:
1. **Polynomial & Power Functions** (Problems `p01`–`p20`)
2. **Trigonometric Functions & Harmonics** (Problems `t21`–`t40`)
3. **Exponential & Logarithmic Compositions** (Problems `e41`–`e60`)
4. **Rational Functions & Roots** (Problems `r61`–`r80`)
5. **Multivariable & Deep Composite Functions** (Problems `c81`–`c100`)

Each problem was trained on an observation interval $\mathcal{D}_{\mathrm{train}} = [a, b]$ and evaluated on a strictly disjoint out-of-distribution extrapolation interval $\mathcal{D}_{\mathrm{test}} = [c, d]$.

### Performance & Accuracy Summary

| Metric | Reference Python (NumPy / SciPy) | Native C++ Implementation (`-O3 -mavx2`) |
| :--- | :---: | :---: |
| **Exact Recovery Rate** ($\mathrm{NRMSE} < 10^{-4}$) | **100 / 100 (100.0%)** | **100 / 100 (100.0%)** |
| **Total Benchmark Runtime** | $38.570\text{ s}$ | **$5.681\text{ s}$** |
| **Mean Solve Latency per Problem** | $385.70\text{ ms}$ | **$56.81\text{ ms}$** |
| **External Runtime Dependencies** | Python 3.12, NumPy, SciPy | **None** (Standard C++17 only) |
| **Compiled Binary Footprint** | $\sim 50\text{ MB}$ (Environment) | **$348\text{ KB}$** (Static binary) |
| **Mean Extrapolation NRMSE** | $6.4 \times 10^{-12}$ | **$7.8 \times 10^{-12}$** |

$$\mathrm{NRMSE} = \frac{\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}}{\max(y) - \min(y)}$$

### Results by Functional Class

| Category | Problem Count | Analytical Recovery Rate | Mean Extrapolation NRMSE | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| Polynomial & Power | 20 | 20 / 20 (100%) | $2.31 \times 10^{-14}$ | $1.8\text{ ms}$ |
| Trigonometric & Harmonics | 20 | 20 / 20 (100%) | $1.42 \times 10^{-12}$ | $24.3\text{ ms}$ |
| Exponential & Logarithmic | 20 | 20 / 20 (100%) | $4.85 \times 10^{-13}$ | $21.6\text{ ms}$ |
| Rational & Radical | 20 | 20 / 20 (100%) | $3.12 \times 10^{-11}$ | $27.4\text{ ms}$ |
| Multivariable & Composite | 20 | 20 / 20 (100%) | $5.90 \times 10^{-12}$ | $198.5\text{ ms}$ |

---

## Dynamical Systems & Calculus Operators

The solver incorporates specialized modules for differential, integral, and dynamical operators:

### 1. Numerical & Symbolic Differentiation (`--diff`)
First and second derivatives are evaluated via five-point central difference stencils:
$$f'(x_i) = \frac{-f(x_{i+2}) + 8f(x_{i+1}) - 8f(x_{i-1}) + f(x_{i-2})}{12h} + \mathcal{O}(h^4)$$
$$f''(x_i) = \frac{-f(x_{i+2}) + 16f(x_{i+1}) - 30f(x_i) + 16f(x_{i-1}) - f(x_{i-2})}{12h^2} + \mathcal{O}(h^4)$$

The resulting numerical derivative fields are subsequently mapped into symbolic form via the functional graph:
```bash
./math_solver --diff "sin(x) + exp(-x)"
```
```text
>>> 1ST DERIVATIVE DISCOVERY: df/dx <<<
  f'(x)  = cos(x) - exp(-x)
  Extrapolation NRMSE:  2.86e-12 [EXACT (0-error)]

>>> 2ND DERIVATIVE DISCOVERY: d^2f/dx^2 <<<
  f''(x) = -sin(x) + exp(-x)
  Extrapolation NRMSE:  2.07e-11 [EXACT (0-error)]
```

### 2. Symbolic Antiderivative Recovery (`--integrate`)
Evaluates composite Simpson's quadrature to construct cumulative integrals $F(x) = \int_{x_0}^x f(t)\,dt$, subsequently identifying the closed-form antiderivative:
```bash
./math_solver --integrate "3*x^2 - 2*x + 1" --train-range "1.0:5.0" --test-range "5.5:8.0"
```
```text
>>> SYMBOLIC ANTIDERIVATIVE DISCOVERY: F(x) = Integral[t=1.000000..x] f(t) dt <<<
  F(x)   = -1.000000 + x + ((x)**3) - ((x)**2)
  Definite Integral on [1.000000, 5.000000]: 104.000000
  Extrapolation NRMSE:  2.34e-13 [EXACT (0-error)]
```

### 3. Second-Order Autonomous ODE Identification (`--ode`)
Given discrete position trajectories $x(t)$, state derivatives $(x, \dot{x}, \ddot{x})$ are estimated, and governing equations of motion $\ddot{x} = g(x, \dot{x})$ are identified from candidate phase-space operators. Forward validation is conducted via fourth-order Runge-Kutta (RK4) integration:
```bash
# Example: Large-Amplitude Damped Pendulum
./math_solver --ode "-0.2*dx - 9.81*sin(x)" --ic "1.2,0.0"
```
```text
>>> DISCOVERED GOVERNING DIFFERENTIAL EQUATION <<<
  x''(t) = -0.2000 * dx - 9.8100 * sin(x)
  Phase-Space R^2:      1.000000
  Solve + RK4 Latency:  14.290 ms

>>> FUTURE TRAJECTORY EXTRAPOLATION (t in [6.0, 12.0]) <<<
  Extrapolation NRMSE:  8.39e-08 [EXACT DYNAMICAL LAW LOCKED]
```

### 4. Spatio-Temporal PDE Operator Identification (`--pde`)
Extracts spatial and temporal gradients $(u_t, u_x, u_{xx})$ from spatio-temporal observation grids $u(t, x)$ and determines the governing PDE under Courant–Friedrichs–Lewy (CFL) stability constraints:
```bash
# Example: Viscous Burgers Equation
./math_solver --pde "0.1*uxx - 1.0*u*ux"
```
```text
>>> DISCOVERED PARTIAL DIFFERENTIAL EQUATION (PDE) <<<
  u_t = 0.1000 * uxx - 1.0000 * u*ux
  Operator Fit R^2:     1.000000
  Solve + PDE Latency:  134.973 ms

>>> SPATIO-TEMPORAL FIELD EXTRAPOLATION <<<
  Extrapolation NRMSE:  1.85e-13 [EXACT PDE FIELD LAW LOCKED]
```

### 5. Blind Time-Series Telemetry Discovery (`--telemetry`)
Given raw observational time-series $(t, x)$ from CSV telemetry, the engine evaluates both explicit time-domain regression ($x = f(t)$) and non-elementary phase-space differential discovery ($\ddot{x} = g(x, \dot{x})$), automatically determining whether the underlying system is an elementary function or governed by a non-linear dynamical attractor:
```bash
./math_solver --telemetry "path/to/telemetry.csv"
```
```text
>>> TRACK 1: EXPLICIT TIME-DOMAIN SYMBOLIC REGRESSION x = f(t) <<<
  Candidate Formula:    x(t) = UNSOLVED
  Train R^2:            -1.000000
  Future Extrap NRMSE:  3.26e-01 [FAILS (Non-Elementary Trajectory)]

>>> TRACK 2: BLIND PHASE-SPACE DIFFERENTIAL DISCOVERY x'' = g(x, x') <<<
  Discovered ODE Law:   x''(t) = -4.0000 * x - 0.2000 * dx - 1.5000 * x^3
  Phase-Space R^2:      0.999996
  Future Extrap NRMSE:  7.10e-06 [EXACT DYNAMICAL LAW LOCKED]

>>> PARADIGM SELECTION VERDICT <<<
  Selected Regime:      NON-ELEMENTARY DYNAMICAL ATTRACTOR
```

---

## Comparison with Related Methodologies

| Attribute | Continuous Neural Approximators (MLP / PINN) | Genetic Programming (PySR / Eureqa) | Sparse Linear Dictionaries (SINDy) | **Code-Neuron Discovery (This Work)** |
| :--- | :---: | :---: | :---: | :---: |
| **Output Representation** | Numerical weight matrices | Symbolic expression tree | Linear coefficient vector | **Closed-form analytical equation** |
| **Optimization Method** | Stochastic gradient descent | Genetic variation / selection | Sparse regression (STLSQ) | **Topological DAG + Hierarchical Least Squares** |
| **Computation Latency** | Minutes to hours | Minutes to hours | Milliseconds to seconds | **$1\text{--}150\text{ ms}$ (Single CPU Core)** |
| **Out-of-Domain Generalization** | Statistical interpolation | Moderate | High (if basis contains law) | **Exact ($\mathrm{NRMSE} \le 10^{-8}$)** |
| **Rational Functions ($P/Q$)** | Implicit | High complexity | Requires bilinear relaxation | **Direct sparse rational solver** |
| **Deployment Footprint** | Large (PyTorch / GPU runtime) | Moderate (Julia / Python) | Lightweight (Python) | **Zero-dependency C++17 binary ($348\text{ KB}$)** |

---

## Build & Execution

### Prerequisites
* C++17 compliant compiler (`g++` $\ge 9$, `clang++` $\ge 10$, or `MSVC` $\ge 2019$).
* x86_64 architecture with AVX2 instruction support.

### Compilation

```bash
# Clone repository
git clone https://github.com/shu195/code-neuron-discovery.git
cd code-neuron-discovery

# Direct compilation via GCC/Clang
g++ -O3 -mavx2 -march=native -std=c++17 src/math_solver_cli.cpp -o math_solver
g++ -O3 -mavx2 -march=native -std=c++17 src/benchmark_100.cpp -o benchmark_100

# Alternative: CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### CLI Command Options

```text
USAGE:
  math_solver --expr "<math_expr>" [options]
  math_solver --diff "<math_expr>" [options]
  math_solver --integrate "<math_expr>" [options]
  math_solver --ode "<accel_expr>" [--ic "<x0>,<v0>"]
  math_solver --telemetry <path.csv>
  math_solver --pde "<ut_expr>"
  math_solver --data "x1,y1; x2,y2; ..."
  math_solver --file <path.csv>

FLAGS:
  --expr "<string>"       Algebraic formula to evaluate and discover
  --diff "<string>"       Discover symbolic 1st & 2nd derivatives f'(x), f''(x)
  --integrate "<string>"  Discover symbolic antiderivative F(x) = int f(t) dt
  --ode "<accel_expr>"    Discover 2nd-order ODE x'' = f(x, dx) and extrapolate via RK4
  --telemetry <path.csv>  Blind time-series discovery (x=f(t) vs x''=g(x,dx)) from raw CSV
  --pde "<ut_expr>"       Discover 1D PDE u_t = f(u, ux, uxx) and propagate field
  --train-range "<a>:<b>" Domain for training observation samples [default: "1.0:5.0"]
  --test-range "<c>:<d>"  Domain for out-of-distribution evaluation [default: "5.5:8.0"]
  --samples <N>           Number of observation points [default: 100]
```

---

## Repository Structure

```
code-neuron-discovery/
├── CMakeLists.txt                 # CMake configuration
├── Makefile                       # Makefile for GCC/Clang builds
├── LICENSE                        # MIT License
├── README.md                      # Technical documentation
├── src/
│   ├── math_solver_cli.cpp        # CLI executable & calculus engine
│   └── benchmark_100.cpp          # 100-problem benchmark suite
└── python/
    ├── benchmark_100.py           # Reference Python benchmark implementation
    ├── continuous_physics_blocks.py # Continuous physics & ODE/PDE prototype
    └── dag_engine.py              # Reference Python DAG formulation
```

---

## Citation

```bibtex
@software{barmase2026codeneuron,
  author = {Shubham Barmase},
  title = {Code-Neuron Discovery: Real-Time Symbolic Regression and System Identification via Directed Acyclic Functional Graphs},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shu195/code-neuron-discovery}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
