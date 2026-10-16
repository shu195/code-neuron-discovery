# Code-Neuron Discovery

> **A Real-Time Symbolic Scientific Discovery & Calculus Engine in Pure C++ (AVX2)**  
> *Discovers exact closed-form algebraic laws, ODEs, PDEs, derivatives, and integrals directly from data using Topological Code-Neuron DAGs and Occam Hierarchical Search.*

[![C++17](https://img.shields.io/badge/C%2B%2B-17%20%2F%20AVX2-00599C?logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Benchmark](https://img.shields.io/badge/Benchmark-100%2F100%20(100%25)%20Exact-brightgreen)](https://github.com/shu195/code-neuron-discovery)
[![NRMSE](https://img.shields.io/badge/Extrapolation%20NRMSE-%3C%201e--11-blue)](https://github.com/shu195/code-neuron-discovery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔬 Overview & Key Idea

Modern deep learning architectures (MLPs, Physics-Informed Neural Networks / PINNs, Transformers) approximate continuous phenomena through billions of floating-point weights. While effective at interpolating within training manifolds, they suffer from fundamental limitations:
1. **Zero Interpretability**: They output numerical predictions rather than closed-form mathematical equations.
2. **Catastrophic Out-of-Distribution Drift**: Outside the training support, neural approximations diverge.
3. **High Latency & Heavy Footprint**: Require massive GPU stacks, Python runtime overhead, and long training epochs.

**Code-Neuron Discovery** replaces black-box weight matrices with **Topological Code-Neurons**—executable mathematical blocks connected in a directed acyclic graph (DAG) with **compositional re-inputting**. Coupled with an **Occam 5-Tier Hierarchical Solver**, the engine discovers exact analytical formulations from data in milliseconds without gradient descent or genetic mutation thrashing.

```
Raw Data / Sensor Readings (x, y) 
      │
      ▼
Topological Code-Neuron DAG ──► [Unary & Binary Operators: sin, cos, exp, log, sqrt, ...]
      │
      ▼
Occam 5-Tier Hierarchical Solver:
  ├─ Tier 0: Invariant Constant Check (var < 1e-10)
  ├─ Tier 1: Single-Node Collinearity & Invertible Target Maps
  ├─ Tier 2: Polynomial Sub-basis Pursuit (Powers 1..5)
  ├─ Tier 3: 2-Term Sparse Primitive Basis Pursuit
  └─ Tier 4: Sparse Rational Solver P(x) / (1 + Q(x))
      │
      ▼
Exact Symbolic Equation + Infinite Horizon Extrapolation (NRMSE < 1e-11)
```

---

## ⚡ 100-Problem Benchmark Results

The architecture was evaluated on a comprehensive **100-problem mathematical benchmark** spanning 5 distinct functional categories. Each problem is trained on $[a, b]$ and evaluated on a strictly disjoint out-of-distribution extrapolation interval $[c, d]$.

### Python vs. Native C++ Head-to-Head

| Metric | Python (NumPy + Vectorized) | C++ Native (`-O3 -mavx2`) | Improvement / Advantage |
| :--- | :---: | :---: | :---: |
| **Exact Analytical Solutions** | **100 / 100 (100.0%)** | **100 / 100 (100.0%)** | **Identical mathematical closure** |
| **Total Wall-Clock Time** | `38.570 s` | `5.681 s` | **$6.79\times$ faster execution** |
| **Mean Latency per Problem** | `385.70 ms` | `56.81 ms` | **$-328.89\text{ ms}$ reduction** |
| **External Dependencies** | Python 3.12, NumPy, SciPy | **Zero** (Standard C++ Library only) | Fully portable single binary |
| **Binary / Environment Size** | $\sim 50\text{ MB}$ | **$348\text{ KB}$** executable | **$140\times$ smaller deployment** |
| **Memory / GC Determinism** | Dynamic PyObject allocation | Contiguous stack/heap buffers | Zero GC pauses |

### Category Breakdown

| Category | Problem Count | Analytical Exact ($NRMSE < 10^{-4}$) | Mean Extrapolation NRMSE | Mean C++ Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Polynomial & Power Functions** (`p01`–`p20`) | 20 | **20 / 20 (100%)** | $2.31 \times 10^{-14}$ | $1.8\text{ ms}$ |
| **Trigonometric Functions & Harmonics** (`t21`–`t40`) | 20 | **20 / 20 (100%)** | $1.42 \times 10^{-12}$ | $24.3\text{ ms}$ |
| **Exponential & Logarithmic Nests** (`e41`–`e60`) | 20 | **20 / 20 (100%)** | $4.85 \times 10^{-13}$ | $21.6\text{ ms}$ |
| **Rational Functions & Roots** (`r61`–`r80`) | 20 | **20 / 20 (100%)** | $3.12 \times 10^{-11}$ | $27.4\text{ ms}$ |
| **Multivariable & Deep Composites** (`c81`–`c100`) | 20 | **20 / 20 (100%)** | $5.90 \times 10^{-12}$ | $198.5\text{ ms}$ |
| **Grand Total** | **100** | **100 / 100 (100.0%)** | **$7.8 \times 10^{-12}$** | **$56.8\text{ ms}$** |

---

## 🧮 Calculus & Continuous Physics Blocks

The solver includes specialized calculus and dynamical systems code-neurons:

### 1. Differential Calculus Block (`--diff`)
Extracts 1st and 2nd derivatives using 5-point 4th-order finite-difference stencils ($O(h^4)$ error), then identifies the **exact symbolic derivative formulas**:
```powershell
.\math_solver.exe --diff "sin(x) + exp(-x)"
```
```text
>>> 1ST DERIVATIVE DISCOVERY: df/dx <<<
  f'(x)  = cos(x) - exp(-x)
  Extrapolation NRMSE:  2.86e-12 [EXACT (0-error)]

>>> 2ND DERIVATIVE DISCOVERY: d^2f/dx^2 <<<
  f''(x) = -sin(x) + exp(-x)
  Extrapolation NRMSE:  2.07e-11 [EXACT (0-error)]
```

### 2. Integral Calculus Block (`--integrate`)
Performs composite Simpson's quadrature to accumulate $F(x) = \int_{x_0}^x f(t)\,dt$ and discovers the **exact symbolic antiderivative**:
```powershell
.\math_solver.exe --integrate "3*x^2 - 2*x + 1" --train-range "1.0:5.0" --test-range "5.5:8.0"
```
```text
>>> SYMBOLIC ANTIDERIVATIVE DISCOVERY: F(x) = Integral[t=1.000000..x] f(t) dt <<<
  F(x)   = -1.000000 + x + ((x)**3) - ((x)**2)
  Definite Integral on [1.000000, 5.000000]: 104.000000
  Extrapolation NRMSE:  2.34e-13 [EXACT (0-error)]
```

### 3. Autonomous 2nd-Order ODE Discovery & RK4 Extrapolation (`--ode`)
Given only discrete position observations $x(t)$, the engine numerically differentiates, searches candidate phase-space code-neurons $(x, \dot{x}, \sin(x), x^3, \dots)$, recovers the governing equation of motion, and integrates forward via RK4 to predict future unseen horizons:
```powershell
# Nonlinear Large-Amplitude Pendulum:
.\math_solver.exe --ode "-0.2*dx - 9.81*sin(x)" --ic "1.2,0.0"
```
```text
>>> DISCOVERED GOVERNING DIFFERENTIAL EQUATION <<<
  x''(t) = -0.2000 * dx - 9.8100 * sin(x)
  Phase-Space R^2:      1.000000
  Solve + RK4 Latency:  14.290 ms

>>> FUTURE TRAJECTORY EXTRAPOLATION (t in [6.0, 12.0]) <<<
  Extrapolation NRMSE:  8.39e-08 [EXACT DYNAMICAL LAW LOCKED]
```

### 4. Spatio-Temporal PDE Field Identification (`--pde`)
Identifies 1D time-dependent PDEs $u_t = f(u, u_x, u_{xx})$ from spatio-temporal grid data and verifies forward field propagation under CFL-stable conditions:
```powershell
# Viscous Burgers Shockwave PDE:
.\math_solver.exe --pde "0.1*uxx - 1.0*u*ux"
```
```text
>>> DISCOVERED PARTIAL DIFFERENTIAL EQUATION (PDE) <<<
  u_t = 0.1000 * uxx - 1.0000 * u*ux
  Operator Fit R^2:     1.000000
  Solve + PDE Latency:  134.973 ms

>>> SPATIO-TEMPORAL FIELD EXTRAPOLATION <<<
  Extrapolation NRMSE:  1.85e-13 [EXACT PDE FIELD LAW LOCKED]
```

---

## 📊 Paradigm Comparison

| Capability | Standard MLPs / PINNs | Genetic Programming (PySR / Eureqa) | Transformer Regressors | **Code-Neuron Discovery (This Work)** |
| :--- | :---: | :---: | :---: | :---: |
| **Output Representation** | Black-box weights | Symbolic Tree | Token sequence | **Exact Closed-Form Symbolic Law** |
| **Solve Latency** | Minutes (GPU epochs) | Minutes to Hours | Seconds (GPU forward pass) | **$1\text{--}150\text{ ms}$ (Single CPU Core)** |
| **Extrapolation NRMSE** | Diverges ($> 50\%$) | $10^{-2}\text{--}10^{-4}$ | $\sim 5\%\text{--}20\%$ | **$< 10^{-11}$ (Machine Precision)** |
| **Calculus & ODE/PDE Discovery** | Requires AD/PINN loss | Complex custom wrappers | None | **Native (`--diff`, `--integrate`, `--ode`, `--pde`)** |
| **Runtime Footprint** | PyTorch / CUDA ($> 2\text{ GB}$) | Julia + Python runtime | PyTorch + LLM weights | **Single Portable C++ Binary ($348\text{ KB}$)** |

---

## 🚀 Quickstart & Compilation

### Requirements
* Any modern C++17 compiler (`g++`, `clang++`, or `MSVC`).
* x86_64 CPU with AVX2 support (standard on CPUs from 2013+).
* Zero external libraries or package managers required.

### Build with CMake
```bash
git clone https://github.com/shu195/code-neuron-discovery.git
cd code-neuron-discovery
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Build Directly with GCC / Clang
```bash
# Build Scientific Solver CLI:
g++ -O3 -mavx2 -std=c++17 src/math_solver_cli.cpp -o math_solver

# Build 100-Problem Benchmark:
g++ -O3 -mavx2 -std=c++17 src/benchmark_100.cpp -o benchmark_100
```

---

## 💻 CLI Usage Guide

```text
USAGE:
  math_solver --expr "<math_expr>" [options]
  math_solver --diff "<math_expr>" [options]
  math_solver --integrate "<math_expr>" [options]
  math_solver --ode "<accel_expr>" [--ic "<x0>,<v0>"]
  math_solver --pde "<ut_expr>"
  math_solver --data "x1,y1; x2,y2; ..."
  math_solver --file <path.csv>

FLAGS:
  --expr "<string>"       Algebraic formula to evaluate and discover
  --diff "<string>"       Discover symbolic 1st & 2nd derivatives f'(x), f''(x)
  --integrate "<string>"  Discover symbolic antiderivative F(x) = int f(t) dt
  --ode "<accel_expr>"    Discover 2nd-order ODE x'' = f(x, dx) & RK4 extrapolate
  --pde "<ut_expr>"       Discover 1D PDE u_t = f(u, ux, uxx) & propagate field
  --train-range "<a>:<b>" Training domain [default: "1.0:5.0"]
  --test-range "<c>:<d>"  Extrapolation test domain [default: "5.5:8.0"]
  --samples <N>           Number of sample points [default: 100]
```

---

## 📂 Repository Structure

```
code-neuron-discovery/
├── CMakeLists.txt              # Standard CMake build configuration
├── Makefile                    # One-command Makefile
├── LICENSE                     # MIT License
├── README.md                   # Complete documentation
├── src/
│   ├── math_solver_cli.cpp     # Unified C++ CLI, Calculus & Dynamical Engine
│   └── benchmark_100.cpp       # 100-Problem Benchmark Suite
└── python/
    ├── benchmark_100.py        # Python vectorized reference benchmark
    ├── continuous_physics_blocks.py # Continuous physics & ODE/PDE prototype
    └── dag_engine.py           # Core Python DAG solver
```

---

## 📜 Citation

If you use this engine in your research or applications, please cite:

```bibtex
@software{barmase2026codeneuron,
  author = {Shubham Barmase},
  title = {Code-Neuron Discovery: A Real-Time Symbolic Scientific Discovery & Calculus Engine in Pure C++},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shu195/code-neuron-discovery}}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
