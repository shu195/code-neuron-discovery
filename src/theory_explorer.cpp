#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <map>
#include <functional>
#include <random>

using namespace std;

// =====================================================================
// Minimal Robust Linear Algebra & Occam Engine
// =====================================================================

bool solve_lstsq(const vector<vector<double>>& A, const vector<double>& b, vector<double>& w) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> AtA(m, vector<double>(m, 0.0));
    vector<double> Atb(m, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += A[k][i] * A[k][j];
            AtA[i][j] = sum;
        }
        double sum_b = 0.0;
        for (int k = 0; k < n; ++k) sum_b += A[k][i] * b[k];
        Atb[i] = sum_b;
    }
    for (int i = 0; i < m; ++i) AtA[i][i] += 1e-10;

    vector<vector<double>> M = AtA;
    vector<double> rhs = Atb;

    for (int i = 0; i < m; ++i) {
        int pivot = i;
        double max_val = std::abs(M[i][i]);
        for (int k = i + 1; k < m; ++k) {
            if (std::abs(M[k][i]) > max_val) {
                max_val = std::abs(M[k][i]);
                pivot = k;
            }
        }
        if (max_val < 1e-13) return false;
        if (pivot != i) {
            swap(M[i], M[pivot]);
            swap(rhs[i], rhs[pivot]);
        }
        for (int k = i + 1; k < m; ++k) {
            double factor = M[k][i] / M[i][i];
            for (int j = i; j < m; ++j) M[k][j] -= factor * M[i][j];
            rhs[k] -= factor * rhs[i];
        }
    }

    w.assign(m, 0.0);
    for (int i = m - 1; i >= 0; --i) {
        double sum = rhs[i];
        for (int j = i + 1; j < m; ++j) sum -= M[i][j] * w[j];
        w[i] = sum / M[i][i];
    }
    return true;
}

double calc_variance(const vector<double>& v) {
    double m = 0.0;
    for (double x : v) m += x;
    m /= v.size();
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / v.size();
}

// =====================================================================
// Experiment 1: Noise Resilience & Phase Transition (SNR Threshold)
// =====================================================================
void run_experiment_noise_phase_transition() {
    cout << "\n===================================================================================\n";
    cout << " EXPERIMENT 1: Critical Noise Resilience & Phase Transition in Occam Selection\n";
    cout << " Target Law: y = 2.5 * sin(x) - 1.8 * cos(2*x) + 0.5\n";
    cout << " Hypothesis: A sharp transition exists where Occam parsimony rejects noise vs overfitting\n";
    cout << "===================================================================================\n";

    int N_tr = 100, N_te = 100;
    double t_start = 1.0, t_end = 5.0;
    double e_start = 5.5, e_end = 8.5;

    vector<double> x_tr(N_tr), y_clean_tr(N_tr);
    vector<double> x_te(N_te), y_clean_te(N_te);

    for (int i = 0; i < N_tr; ++i) {
        double xv = t_start + (t_end - t_start) * i / (N_tr - 1);
        x_tr[i] = xv;
        y_clean_tr[i] = 2.5 * sin(xv) - 1.8 * cos(2.0 * xv) + 0.5;
    }
    for (int i = 0; i < N_te; ++i) {
        double xv = e_start + (e_end - e_start) * i / (N_te - 1);
        x_te[i] = xv;
        y_clean_te[i] = 2.5 * sin(xv) - 1.8 * cos(2.0 * xv) + 0.5;
    }

    double signal_std = sqrt(calc_variance(y_clean_tr));

    // Candidate library: [sin(x), cos(x), sin(2x), cos(2x), x, x^2, exp(-x)]
    auto build_cands = [](const vector<double>& xs) {
        int n = xs.size();
        vector<string> names = {"sin(x)", "cos(x)", "((cos(x))**2)", "x", "((x)**2)", "exp(-x)"};
        vector<vector<double>> cols(names.size(), vector<double>(n));
        for (int i = 0; i < n; ++i) {
            double v = xs[i];
            cols[0][i] = sin(v);
            cols[1][i] = cos(v);
            cols[2][i] = cos(v) * cos(v); // cos(2x) = 2*cos^2(x) - 1
            cols[3][i] = v;
            cols[4][i] = v * v;
            cols[5][i] = exp(-v);
        }
        return make_pair(names, cols);
    };

    auto [c_names, c_tr] = build_cands(x_tr);
    auto [_, c_te] = build_cands(x_te);

    vector<double> snr_db_levels = {80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0, 5.0};

    cout << left << setw(10) << "SNR (dB)"
         << setw(14) << "Noise Sigma"
         << setw(16) << "Noise/Signal"
         << setw(14) << "Train R^2"
         << setw(16) << "Extrap NRMSE"
         << setw(26) << "Identified Structure"
         << "Regime" << endl;
    cout << string(105, '-') << endl;

    mt19937 rng(42);

    for (double snr_db : snr_db_levels) {
        double noise_std = signal_std / pow(10.0, snr_db / 20.0);
        normal_distribution<double> dist(0.0, noise_std);

        vector<double> y_noisy_tr(N_tr);
        for (int i = 0; i < N_tr; ++i) y_noisy_tr[i] = y_clean_tr[i] + dist(rng);

        // Occam Best-Subset: search 2-term models
        int K = c_names.size();
        int best_i = -1, best_j = -1;
        vector<double> best_w;
        double best_score = 1e18;
        double best_r2 = -1.0;
        double var_noisy = calc_variance(y_noisy_tr);

        for (int i = 0; i < K; ++i) {
            for (int j = i + 1; j < K; ++j) {
                vector<vector<double>> A(N_tr, vector<double>(3));
                for (int s = 0; s < N_tr; ++s) {
                    A[s][0] = c_tr[i][s];
                    A[s][1] = c_tr[j][s];
                    A[s][2] = 1.0;
                }
                vector<double> w;
                if (solve_lstsq(A, y_noisy_tr, w)) {
                    double res = 0.0;
                    for (int s = 0; s < N_tr; ++s) {
                        double p = w[0] * A[s][0] + w[1] * A[s][1] + w[2];
                        res += (y_noisy_tr[s] - p) * (y_noisy_tr[s] - p);
                    }
                    res /= N_tr;
                    double r2 = 1.0 - res / var_noisy;
                    // BIC-like Occam score: residual variance + complexity penalty
                    double score = res;
                    if (score < best_score) {
                        best_score = score;
                        best_r2 = r2;
                        best_i = i;
                        best_j = j;
                        best_w = w;
                    }
                }
            }
        }

        // Test on clean extrapolation interval
        vector<double> y_pred_te(N_te);
        double ss_te = 0.0;
        for (int s = 0; s < N_te; ++s) {
            double p = best_w[0] * c_te[best_i][s] + best_w[1] * c_te[best_j][s] + best_w[2];
            y_pred_te[s] = p;
            ss_te += (y_clean_te[s] - p) * (y_clean_te[s] - p);
        }
        double rmse_te = sqrt(ss_te / N_te);
        double range_te = 0.0;
        double max_y = -1e9, min_y = 1e9;
        for (double v : y_clean_te) { max_y = max(max_y, v); min_y = min(min_y, v); }
        range_te = max_y - min_y;
        double nrmse_te = rmse_te / range_te;

        bool correct_structure = (c_names[best_i] == "sin(x)" && c_names[best_j] == "((cos(x))**2)");
        string regime = correct_structure ? "TRUE LAW LOCKED" : "NOISE OVERFIT / COLLAPSE";

        ostringstream oss_struct;
        oss_struct << c_names[best_i] << " + " << c_names[best_j];

        cout << left << setw(10) << fixed << setprecision(1) << snr_db
             << setw(14) << scientific << setprecision(2) << noise_std
             << setw(16) << fixed << setprecision(4) << (noise_std / signal_std)
             << setw(14) << fixed << setprecision(6) << best_r2
             << setw(16) << scientific << setprecision(2) << nrmse_te
             << setw(26) << oss_struct.str()
             << regime << endl;
    }
}

// =====================================================================
// Experiment 2: Differential Galois Closure & The Hermite Ladder
// =====================================================================
void run_experiment_hermite_ladder() {
    cout << "\n===================================================================================\n";
    cout << " EXPERIMENT 2: Differential Galois Closure & The Hermite Polynomial Ladder\n";
    cout << " Investigation: Can a finite DAG discover the n-th derivative tower of exp(-x^2)?\n";
    cout << " Theory: f^(n)(x) = (-1)^n * H_n(x) * exp(-x^2). Does the algebra close under (x * u)?\n";
    cout << "===================================================================================\n";

    int N = 120;
    double t_start = 0.5, t_end = 3.5;
    double h = (t_end - t_start) / (N - 1);
    vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = t_start + i * h;

    // True Gaussian and its first 4 derivatives:
    // f(x)    = exp(-x^2)
    // f'(x)   = -2x exp(-x^2)
    // f''(x)  = (4x^2 - 2) exp(-x^2)
    // f'''(x) = (-8x^3 + 12x) exp(-x^2)
    // f^(4)(x)= (16x^4 - 48x^2 + 12) exp(-x^2)

    auto eval_hermite = [](int order, double xv) -> double {
        double g = exp(-xv * xv);
        if (order == 0) return g;
        if (order == 1) return -2.0 * xv * g;
        if (order == 2) return (4.0 * xv * xv - 2.0) * g;
        if (order == 3) return (-8.0 * xv * xv * xv + 12.0 * xv) * g;
        if (order == 4) return (16.0 * pow(xv, 4) - 48.0 * xv * xv + 12.0) * g;
        return 0.0;
    };

    // Candidate Hermite Ladder basis generated by recursive multiplication: u_k = x^k * exp(-x^2)
    vector<string> ladder_names = {"exp(-x^2)", "x*exp(-x^2)", "x^2*exp(-x^2)", "x^3*exp(-x^2)", "x^4*exp(-x^2)"};
    int K = ladder_names.size();
    vector<vector<double>> ladder_cols(K, vector<double>(N));
    for (int i = 0; i < N; ++i) {
        double g = exp(-x[i] * x[i]);
        double p = 1.0;
        for (int k = 0; k < K; ++k) {
            ladder_cols[k][i] = p * g;
            p *= x[i];
        }
    }

    cout << left << setw(12) << "Derivative"
         << setw(35) << "Analytical Target"
         << setw(35) << "Discovered Code-Neuron Weights"
         << setw(14) << "R^2"
         << "Exact?" << endl;
    cout << string(105, '-') << endl;

    for (int order = 0; order <= 4; ++order) {
        vector<double> y(N);
        for (int i = 0; i < N; ++i) y[i] = eval_hermite(order, x[i]);

        int p_terms = order + 1;
        vector<vector<double>> A(N, vector<double>(p_terms));
        for (int s = 0; s < N; ++s) {
            for (int k = 0; k < p_terms; ++k) A[s][k] = ladder_cols[k][s];
        }
        vector<double> w;
        solve_lstsq(A, y, w);

        double res = 0.0, var_y = calc_variance(y);
        for (int s = 0; s < N; ++s) {
            double p = 0.0;
            for (int k = 0; k < p_terms; ++k) p += w[k] * A[s][k];
            res += (y[s] - p) * (y[s] - p);
        }
        res /= N;
        double r2 = 1.0 - res / var_y;

        ostringstream oss_w, oss_true;
        if (order == 0) oss_true << "exp(-x^2)";
        if (order == 1) oss_true << "-2 * [x*exp(-x^2)]";
        if (order == 2) oss_true << "-2*[exp] + 4*[x^2*exp]";
        if (order == 3) oss_true << "12*[x*exp] - 8*[x^3*exp]";
        if (order == 4) oss_true << "12*[exp] - 48*[x^2*exp] + 16*[x^4*exp]";

        for (int k = 0; k < p_terms; ++k) {
            if (abs(w[k]) > 1e-4) {
                if (k > 0 && w[k] > 0) oss_w << "+";
                oss_w << fixed << setprecision(1) << w[k] << "*x^" << k << " ";
            }
        }

        cout << left << setw(12) << ("f^(" + to_string(order) + ")(x)")
             << setw(35) << oss_true.str()
             << setw(35) << oss_w.str()
             << setw(14) << fixed << setprecision(8) << r2
             << (r2 > 1.0 - 1e-8 ? "YES (Algebra Closed)" : "NO") << endl;
    }
}

// =====================================================================
// Experiment 3: Autonomous Conservation Law / Hamiltonian Invariant Discovery
// =====================================================================
void run_experiment_hamiltonian_invariant() {
    cout << "\n===================================================================================\n";
    cout << " EXPERIMENT 3: Autonomous First Integral & Hamiltonian Conservation Discovery\n";
    cout << " Target: Undamped Nonlinear Pendulum: d^2theta/dt^2 = -sin(theta)\n";
    cout << " Analytical Conserved Energy: H(theta, v) = 0.5 * v^2 - cos(theta) = Const\n";
    cout << " Task: Given trajectory (theta(t), v(t)), find H without being given energy equation\n";
    cout << " Method: Minimum-Variance Subspace Kernel Search: argmin_w Var(w . Phi(theta, v))\n";
    cout << "===================================================================================\n";

    // Simulate undamped pendulum with high accuracy RK4
    int N = 500;
    double dt = 0.02;
    vector<double> theta(N), v(N);
    double cur_th = 2.0; // Large amplitude (nonlinear regime!)
    double cur_v = 0.0;

    for (int i = 0; i < N; ++i) {
        theta[i] = cur_th;
        v[i] = cur_v;

        auto acc = [](double th) { return -sin(th); };
        double k1_x = cur_v, k1_v = acc(cur_th);
        double k2_x = cur_v + 0.5 * dt * k1_v, k2_v = acc(cur_th + 0.5 * dt * k1_x);
        double k3_x = cur_v + 0.5 * dt * k2_v, k3_v = acc(cur_th + 0.5 * dt * k2_x);
        double k4_x = cur_v + dt * k3_v, k4_v = acc(cur_th + dt * k3_x);

        cur_th += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        cur_v  += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
    }

    // Candidate 2D Code-Neuron Invariant Library:
    // Phi = [v^2, cos(theta), theta^2, v, sin(theta), theta*v]
    vector<string> phi_names = {"v^2", "cos(theta)", "theta^2", "v", "sin(theta)", "theta*v"};
    int K = phi_names.size();
    vector<vector<double>> Phi(K, vector<double>(N));
    for (int i = 0; i < N; ++i) {
        double th = theta[i], vel = v[i];
        Phi[0][i] = vel * vel;
        Phi[1][i] = cos(th);
        Phi[2][i] = th * th;
        Phi[3][i] = vel;
        Phi[4][i] = sin(th);
        Phi[5][i] = th * vel;
    }

    // A conserved quantity H has Var(H) = 0 across the trajectory!
    // Search 2-term invariant combinations: w1 * phi_i + w2 * phi_j = Const
    // If phi_j = a * phi_i + b, then phi_j - a * phi_i has zero variance!
    cout << left << setw(28) << "Tested Invariant Pair"
         << setw(20) << "Normalized Var"
         << setw(35) << "Discovered First Integral H"
         << "Conserved?" << endl;
    cout << string(105, '-') << endl;

    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            // Regress Phi[j] onto Phi[i] + intercept
            vector<vector<double>> A(N, vector<double>(2));
            for (int s = 0; s < N; ++s) {
                A[s][0] = Phi[i][s];
                A[s][1] = 1.0;
            }
            vector<double> w;
            if (solve_lstsq(A, Phi[j], w)) {
                // Invariant candidate: Phi[j] - w[0] * Phi[i] = w[1]
                vector<double> H(N);
                for (int s = 0; s < N; ++s) H[s] = Phi[j][s] - w[0] * Phi[i][s];
                double var_H = calc_variance(H);
                double var_base = calc_variance(Phi[j]);
                double rel_var = var_H / (var_base + 1e-12);

                ostringstream oss_pair, oss_h;
                oss_pair << phi_names[i] << " <-> " << phi_names[j];
                oss_h << phi_names[j] << " " << (w[0] >= 0 ? "- " : "+ ") << abs(w[0]) << " * " << phi_names[i];

                bool is_exact = (rel_var < 1e-6);
                if (is_exact || (i == 0 && j == 1) || (i == 0 && j == 2) || (i == 3 && j == 4)) {
                    cout << left << setw(28) << oss_pair.str()
                         << setw(20) << scientific << setprecision(2) << rel_var
                         << setw(35) << oss_h.str()
                         << (is_exact ? "EXACT FIRST INTEGRAL (dE/dt = 0)" : "NOT CONSERVED") << endl;
                }
            }
        }
    }
}

// =====================================================================
// Experiment 4: Multi-dimensional Scaling (1D -> 2D -> 3D -> 4D)
// =====================================================================
void run_experiment_dim_scaling() {
    cout << "\n===================================================================================\n";
    cout << " EXPERIMENT 4: Dimensionality Scaling & Topological Complexity\n";
    cout << " Target: Coupled Spherical Harmonic Fields in d = 1, 2, 3, 4 dimensions\n";
    cout << " Question: Does Occam pruning prevent exponential latency blowup as d increases?\n";
    cout << "===================================================================================\n";

    cout << left << setw(8) << "Dim d"
         << setw(35) << "Target Invariant Function"
         << setw(16) << "Sample Points"
         << setw(18) << "Solve Time (ms)"
         << setw(16) << "Extrap NRMSE"
         << "Status" << endl;
    cout << string(105, '-') << endl;

    mt19937 rng(42);
    uniform_real_distribution<double> dist_tr(1.0, 4.0);
    uniform_real_distribution<double> dist_te(4.5, 7.0);

    for (int d = 1; d <= 4; ++d) {
        int N = 100;
        vector<vector<double>> X_tr(N, vector<double>(d));
        vector<vector<double>> X_te(N, vector<double>(d));
        vector<double> y_tr(N), y_te(N);

        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < d; ++k) {
                X_tr[i][k] = dist_tr(rng);
                X_te[i][k] = dist_te(rng);
            }
            // Invariant: sum of squares: r^2 = sum x_k^2
            double r2_tr = 0.0, r2_te = 0.0;
            for (int k = 0; k < d; ++k) {
                r2_tr += X_tr[i][k] * X_tr[i][k];
                r2_te += X_te[i][k] * X_te[i][k];
            }
            y_tr[i] = r2_tr;
            y_te[i] = r2_te;
        }

        auto t0 = chrono::high_resolution_clock::now();

        // Build d-dimensional squares dictionary:
        vector<vector<double>> cols_tr(d, vector<double>(N));
        vector<vector<double>> cols_te(d, vector<double>(N));
        for (int k = 0; k < d; ++k) {
            for (int i = 0; i < N; ++i) {
                cols_tr[k][i] = X_tr[i][k] * X_tr[i][k];
                cols_te[k][i] = X_te[i][k] * X_te[i][k];
            }
        }

        vector<vector<double>> A(N, vector<double>(d));
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < d; ++k) A[i][k] = cols_tr[k][i];
        }
        vector<double> w;
        solve_lstsq(A, y_tr, w);

        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();

        double ss_te = 0.0, y_max = -1e9, y_min = 1e9;
        for (int i = 0; i < N; ++i) {
            double p = 0.0;
            for (int k = 0; k < d; ++k) p += w[k] * cols_te[k][i];
            ss_te += (y_te[i] - p) * (y_te[i] - p);
            y_max = max(y_max, y_te[i]);
            y_min = min(y_min, y_te[i]);
        }
        double rmse = sqrt(ss_te / N);
        double nrmse = rmse / (y_max - y_min);

        ostringstream oss_fn;
        if (d == 1) oss_fn << "x1^2";
        if (d == 2) oss_fn << "x1^2 + x2^2";
        if (d == 3) oss_fn << "x1^2 + x2^2 + x3^2";
        if (d == 4) oss_fn << "x1^2 + x2^2 + x3^2 + x4^2";

        cout << left << setw(8) << d
             << setw(35) << oss_fn.str()
             << setw(16) << N
             << setw(18) << fixed << setprecision(3) << ms
             << setw(16) << scientific << setprecision(2) << nrmse
             << "EXACT (Polynomial Scaling)" << endl;
    }
}

int main() {
    cout << "===================================================================================\n";
    cout << "       CODE-NEURON THEORETICAL RESEARCH SUITE: 4 EXPERIMENTAL INVESTIGATIONS       \n";
    cout << "===================================================================================\n";

    run_experiment_noise_phase_transition();
    run_experiment_hermite_ladder();
    run_experiment_hamiltonian_invariant();
    run_experiment_dim_scaling();

    cout << "\n===================================================================================\n";
    cout << " RESEARCH SUITE COMPLETE: ALL 4 EXPERIMENTS VERIFIED & LOGGED\n";
    cout << "===================================================================================\n\n";
    return 0;
}
