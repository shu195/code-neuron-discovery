#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <functional>
#include <algorithm>

using namespace std;

// =====================================================================
// 1. Math Functions & Stable Primitives
// =====================================================================
inline double fn_sq(double x) { return x * x; }
inline double fn_cube(double x) { return x * x * x; }
inline double fn_sqrt(double x) { return (x >= 0.0) ? std::sqrt(x) : NAN; }
inline double fn_inv(double x) { return (std::abs(x) > 1e-7) ? (1.0 / x) : NAN; }
inline double fn_sin(double x) { return std::sin(x); }
inline double fn_cos(double x) { return std::cos(x); }
inline double fn_tan(double x) { return (std::abs(std::cos(x)) > 1e-4) ? std::tan(x) : NAN; }
inline double fn_exp(double x) { return (x < 25.0) ? std::exp(x) : NAN; }
inline double fn_expneg(double x) { return (-x < 25.0) ? std::exp(-x) : NAN; }
inline double fn_log(double x) { return (x > 1e-7) ? std::log(x) : NAN; }
inline double fn_log1p(double x) { return (x > -0.999) ? std::log1p(x) : NAN; }
inline double fn_sqrt1p(double x) { return (x >= -1.0) ? std::sqrt(std::max(1.0 + x, 0.0)) : NAN; }
inline double fn_inv1p(double x) { return (std::abs(1.0 + x) > 1e-7) ? (1.0 / (1.0 + x)) : NAN; }
inline double fn_tanh(double x) { return std::tanh(x); }
inline double fn_abs(double x) { return std::abs(x); }
inline double fn_gauss(double x) { return std::exp(-std::min(x * x, 30.0)); }

struct UnaryOp {
    string name;
    function<double(double)> fn;
    string fmt;
};

const vector<UnaryOp> UNARY_OPS = {
    {"neg", [](double x){ return -x; }, "(-%s)"},
    {"sq", fn_sq, "((%s)**2)"},
    {"cube", fn_cube, "((%s)**3)"},
    {"sqrt", fn_sqrt, "sqrt(%s)"},
    {"inv", fn_inv, "(1/(%s))"},
    {"sin", fn_sin, "sin(%s)"},
    {"cos", fn_cos, "cos(%s)"},
    {"tan", fn_tan, "tan(%s)"},
    {"exp", fn_exp, "exp(%s)"},
    {"expneg", fn_expneg, "exp(-%s)"},
    {"log", fn_log, "log(%s)"},
    {"log1p", fn_log1p, "log(1+%s)"},
    {"sqrt1p", fn_sqrt1p, "sqrt(1+%s)"},
    {"inv1p", fn_inv1p, "(1/(1+%s))"},
    {"tanh", fn_tanh, "tanh(%s)"},
    {"abs", fn_abs, "abs(%s)"},
    {"gauss", fn_gauss, "exp(-((%s)**2))"}
};

// =====================================================================
// 2. Linear System Solver (Gaussian Elimination with Partial Pivoting)
// =====================================================================
bool solve_linear_system(vector<vector<double>>& A, vector<double>& b, vector<double>& x_out) {
    int n = A.size();
    x_out.assign(n, 0.0);
    // Augmented matrix
    vector<vector<double>> M(n, vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n] = b[i];
    }

    for (int i = 0; i < n; ++i) {
        int max_row = i;
        double max_val = std::abs(M[i][i]);
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(M[k][i]) > max_val) {
                max_val = std::abs(M[k][i]);
                max_row = k;
            }
        }
        if (max_val < 1e-12) return false; // Singular matrix
        std::swap(M[i], M[max_row]);

        for (int k = i + 1; k < n; ++k) {
            double factor = M[k][i] / M[i][i];
            for (int j = i; j <= n; ++j) {
                M[k][j] -= factor * M[i][j];
            }
        }
    }

    // Back-substitution
    for (int i = n - 1; i >= 0; --i) {
        double sum = M[i][n];
        for (int j = i + 1; j < n; ++j) {
            sum -= M[i][j] * x_out[j];
        }
        x_out[i] = sum / M[i][i];
    }
    return true;
}

// Normal equations: solve (A^T A + lambda*I) w = A^T y
bool solve_least_squares(const vector<vector<double>>& A, const vector<double>& y, vector<double>& w_out, double ridge = 1e-11) {
    int N = A.size();
    if (N == 0) return false;
    int p = A[0].size();
    vector<vector<double>> AtA(p, vector<double>(p, 0.0));
    vector<double> Aty(p, 0.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
            Aty[j] += A[i][j] * y[i];
            for (int k = j; k < p; ++k) {
                AtA[j][k] += A[i][j] * A[i][k];
            }
        }
    }
    for (int j = 0; j < p; ++j) {
        AtA[j][j] += ridge;
        for (int k = 0; k < j; ++k) AtA[j][k] = AtA[k][j];
    }
    return solve_linear_system(AtA, Aty, w_out);
}

// =====================================================================
// 3. Topological Code-Neuron DAG
// =====================================================================
enum OpType { OP_INPUT, OP_UNARY, OP_BINARY };

struct DAGNode {
    string name;
    OpType type;
    int op_id;
    int p1, p2;
    int depth;
};

vector<DAGNode> build_candidate_dag(int d, int max_depth = 2) {
    vector<DAGNode> nodes;
    for (int i = 0; i < d; ++i) {
        DAGNode node;
        node.name = (d > 1) ? ("x" + to_string(i + 1)) : "x";
        node.type = OP_INPUT;
        node.op_id = -1;
        node.p1 = node.p2 = -1;
        node.depth = 0;
        nodes.push_back(node);
    }

    // Depth 1
    vector<int> lvl1_indices;
    for (int idx = 0; idx < d; ++idx) {
        for (size_t op_i = 0; op_i < UNARY_OPS.size(); ++op_i) {
            DAGNode n;
            char buf[128];
            snprintf(buf, sizeof(buf), UNARY_OPS[op_i].fmt.c_str(), nodes[idx].name.c_str());
            n.name = buf;
            n.type = OP_UNARY;
            n.op_id = op_i;
            n.p1 = idx; n.p2 = -1;
            n.depth = 1;
            nodes.push_back(n);
            lvl1_indices.push_back(nodes.size() - 1);
        }
    }

    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            DAGNode mul_node;
            mul_node.name = "(" + nodes[i].name + " * " + nodes[j].name + ")";
            mul_node.type = OP_BINARY; mul_node.op_id = 0; // mul
            mul_node.p1 = i; mul_node.p2 = j; mul_node.depth = 1;
            nodes.push_back(mul_node);
            lvl1_indices.push_back(nodes.size() - 1);

            DAGNode add_node;
            add_node.name = "(" + nodes[i].name + " + " + nodes[j].name + ")";
            add_node.type = OP_BINARY; add_node.op_id = 1; // add
            add_node.p1 = i; add_node.p2 = j; add_node.depth = 1;
            nodes.push_back(add_node);
            lvl1_indices.push_back(nodes.size() - 1);

            DAGNode sub_node;
            sub_node.name = "(" + nodes[i].name + " - " + nodes[j].name + ")";
            sub_node.type = OP_BINARY; sub_node.op_id = 2; // sub
            sub_node.p1 = i; sub_node.p2 = j; sub_node.depth = 1;
            nodes.push_back(sub_node);
            lvl1_indices.push_back(nodes.size() - 1);
        }
    }

    if (max_depth < 2) return nodes;

    // Depth 2
    vector<int> lvl2_squares;
    for (int idx : lvl1_indices) {
        for (size_t op_i = 0; op_i < UNARY_OPS.size(); ++op_i) {
            DAGNode n;
            char buf[256];
            snprintf(buf, sizeof(buf), UNARY_OPS[op_i].fmt.c_str(), nodes[idx].name.c_str());
            n.name = buf;
            n.type = OP_UNARY;
            n.op_id = op_i;
            n.p1 = idx; n.p2 = -1;
            n.depth = 2;
            nodes.push_back(n);
            if (UNARY_OPS[op_i].name == "sq" || UNARY_OPS[op_i].name == "cube") {
                lvl2_squares.push_back(nodes.size() - 1);
            }
        }
    }

    vector<int> base_indices;
    for (int i = 0; i < d; ++i) base_indices.push_back(i);
    for (int idx : lvl1_indices) base_indices.push_back(idx);

    for (size_t ia = 0; ia < base_indices.size(); ++ia) {
        int i = base_indices[ia];
        for (size_t ib = ia + 1; ib < base_indices.size(); ++ib) {
            int j = base_indices[ib];
            DAGNode mul_node;
            mul_node.name = "(" + nodes[i].name + " * " + nodes[j].name + ")";
            mul_node.type = OP_BINARY; mul_node.op_id = 0;
            mul_node.p1 = i; mul_node.p2 = j; mul_node.depth = 2;
            nodes.push_back(mul_node);
        }
    }

    // Depth 3 (Compositional re-inputting of squares)
    for (int idx : lvl2_squares) {
        for (size_t op_i = 0; op_i < UNARY_OPS.size(); ++op_i) {
            const string& op_n = UNARY_OPS[op_i].name;
            if (op_n == "log1p" || op_n == "sqrt1p" || op_n == "inv1p" || op_n == "exp" || op_n == "expneg" || op_n == "sin" || op_n == "cos") {
                DAGNode n;
                char buf[512];
                snprintf(buf, sizeof(buf), UNARY_OPS[op_i].fmt.c_str(), nodes[idx].name.c_str());
                n.name = buf;
                n.type = OP_UNARY;
                n.op_id = op_i;
                n.p1 = idx; n.p2 = -1;
                n.depth = 3;
                nodes.push_back(n);
            }
        }
    }

    return nodes;
}

// Evaluate DAG topologically
vector<vector<double>> eval_dag(const vector<DAGNode>& nodes, const vector<vector<double>>& X) {
    int N = X.size();
    int d = X[0].size();
    int P = nodes.size();
    vector<vector<double>> F(P, vector<double>(N));

    for (int i = 0; i < d; ++i) {
        for (int s = 0; s < N; ++s) F[i][s] = X[s][i];
    }

    for (int p = d; p < P; ++p) {
        const DAGNode& node = nodes[p];
        if (node.type == OP_UNARY) {
            const auto& fn = UNARY_OPS[node.op_id].fn;
            const vector<double>& parent = F[node.p1];
            for (int s = 0; s < N; ++s) {
                F[p][s] = fn(parent[s]);
            }
        } else if (node.type == OP_BINARY) {
            const vector<double>& p1 = F[node.p1];
            const vector<double>& p2 = F[node.p2];
            int op = node.op_id;
            for (int s = 0; s < N; ++s) {
                if (op == 0) F[p][s] = p1[s] * p2[s];
                else if (op == 1) F[p][s] = p1[s] + p2[s];
                else if (op == 2) F[p][s] = p1[s] - p2[s];
            }
        }
    }
    return F;
}

// =====================================================================
// 4. Occam's Hierarchical 5-Tier Solver
// =====================================================================
struct Model {
    string type; // "constant", "linear", "rational"
    string t_name;
    function<double(double)> t_inv;
    double c0;
    vector<int> selected_nodes;
    vector<double> w;
    vector<int> num_nodes, den_nodes;
    vector<double> w_num, w_den;
    double r2;
    string names_str;
};

struct TargetTransform {
    string name;
    function<double(double)> fwd;
    function<double(double)> inv;
    function<bool(const vector<double>&)> chk;
};

const vector<TargetTransform> TARGET_TRANSFORMS = {
    {"identity", [](double y){ return y; }, [](double t){ return t; }, [](const vector<double>& y){ return true; }},
    {"log_y", [](double y){ return std::log(y); }, [](double t){ return std::exp(t); }, [](const vector<double>& y){
        for (double v : y) if (v <= 1e-7) return false; return true;
    }},
    {"sqrt_y", [](double y){ return std::sqrt(y); }, [](double t){ return t * t; }, [](const vector<double>& y){
        for (double v : y) if (v < 0.0) return false; return true;
    }},
    {"inv_y", [](double y){ return 1.0 / y; }, [](double t){ return 1.0 / t; }, [](const vector<double>& y){
        for (double v : y) if (std::abs(v) <= 1e-7) return false; return true;
    }},
    {"sq_y", [](double y){ return y * y; }, [](double t){ return std::sqrt(std::max(t, 0.0)); }, [](const vector<double>& y){
        for (double v : y) if (v < 0.0) return false; return true;
    }}
};

double calc_var(const vector<double>& y) {
    double m = 0.0;
    for (double v : y) m += v;
    m /= y.size();
    double s = 0.0;
    for (double v : y) s += (v - m) * (v - m);
    return s / y.size();
}

double calc_mean(const vector<double>& y) {
    double m = 0.0;
    for (double v : y) m += v;
    return m / y.size();
}

int get_node_complexity(const DAGNode& node) {
    int c = node.depth * 5 + node.name.length();
    for (char ch : node.name) if (ch == '(') c += 1;
    return c;
}

Model solve_math_program(const vector<vector<double>>& X_tr, const vector<double>& y_tr, const vector<DAGNode>& nodes) {
    int N = y_tr.size();
    double var_y_raw = calc_var(y_tr);

    // --- Tier 0: Constant Identity Check ---
    if (var_y_raw < 1e-12) {
        Model m;
        m.type = "constant";
        m.c0 = calc_mean(y_tr);
        m.r2 = 1.0;
        m.names_str = "['const(" + to_string(m.c0) + ")']";
        return m;
    }

    vector<vector<double>> F_tr = eval_dag(nodes, X_tr);
    int P = nodes.size();

    // Filter valid non-constant columns
    vector<int> valid_cols;
    for (int j = 0; j < P; ++j) {
        bool finite = true;
        for (int s = 0; s < N; ++s) {
            if (!std::isfinite(F_tr[j][s]) || std::abs(F_tr[j][s]) > 1e12) { finite = false; break; }
        }
        if (finite && calc_var(F_tr[j]) > 1e-7) {
            valid_cols.push_back(j);
        }
    }

    // Sort by Occam's complexity
    std::sort(valid_cols.begin(), valid_cols.end(), [&](int a, int b){
        return get_node_complexity(nodes[a]) < get_node_complexity(nodes[b]);
    });

    // Deduplicate
    vector<int> unique_cols;
    vector<vector<double>> normed_cache;
    for (int j : valid_cols) {
        double m = calc_mean(F_tr[j]);
        double s = std::sqrt(calc_var(F_tr[j]));
        vector<double> cn(N);
        for (int i = 0; i < N; ++i) cn[i] = (F_tr[j][i] - m) / (s + 1e-12);

        bool is_dup = false;
        for (const auto& prev : normed_cache) {
            double dot = 0.0;
            for (int i = 0; i < N; ++i) dot += prev[i] * cn[i];
            if (std::abs(dot / N) > 1.0 - 1e-11) { is_dup = true; break; }
        }
        if (!is_dup) {
            unique_cols.push_back(j);
            normed_cache.push_back(cn);
        }
    }
    valid_cols = unique_cols;

    const double EXACT_TOL = 1.0 - 1e-10;

    // --- Tier 1: Single Node Exact Check ---
    for (const auto& trans : TARGET_TRANSFORMS) {
        if (!trans.chk(y_tr)) continue;
        vector<double> y_t(N);
        bool valid_t = true;
        for (int i = 0; i < N; ++i) {
            y_t[i] = trans.fwd(y_tr[i]);
            if (!std::isfinite(y_t[i])) { valid_t = false; break; }
        }
        if (!valid_t) continue;
        double var_t = calc_var(y_t);
        if (var_t < 1e-12) continue;
        double mean_t = calc_mean(y_t);

        for (int j : valid_cols) {
            double mean_f = calc_mean(F_tr[j]);
            double var_f = calc_var(F_tr[j]);
            double cov = 0.0;
            for (int i = 0; i < N; ++i) cov += (F_tr[j][i] - mean_f) * (y_t[i] - mean_t);
            cov /= N;

            double corr2 = (cov * cov) / (var_f * var_t + 1e-16);
            if (corr2 > EXACT_TOL) {
                double a = cov / var_f;
                double b = mean_t - a * mean_f;
                // Verify raw y fit
                double res_var = 0.0;
                bool ok = true;
                for (int i = 0; i < N; ++i) {
                    double pred_raw = trans.inv(a * F_tr[j][i] + b);
                    if (!std::isfinite(pred_raw)) { ok = false; break; }
                    res_var += (y_tr[i] - pred_raw) * (y_tr[i] - pred_raw);
                }
                res_var /= N;
                double raw_r2 = 1.0 - res_var / var_y_raw;
                if (ok && raw_r2 > EXACT_TOL) {
                    Model m;
                    m.type = "linear";
                    m.t_name = trans.name;
                    m.t_inv = trans.inv;
                    m.selected_nodes = {j};
                    m.w = {a, b};
                    m.r2 = raw_r2;
                    m.names_str = "['" + nodes[j].name + "']";
                    return m;
                }
            }
        }
    }

    // --- Tier 2: Polynomial Basis Check (up to degree 5) ---
    vector<int> poly_nodes;
    for (int j : valid_cols) {
        const string& nm = nodes[j].name;
        if (nm == "x" || nm == "x1" || nm == "x2" ||
            nm == "((x)**2)" || nm == "((x)**3)" ||
            nm == "(x * ((x)**3))" || nm == "((((x)**2))**2)" || nm == "(((x)**2) * ((x)**3))" ||
            nm == "((x1)**2)" || nm == "((x2)**2)" || nm == "(x1 * x2)" ||
            nm == "((x1)**3)" || nm == "((x2)**3)" || nm == "(x1 * ((x2)**2))" || nm == "(x2 * ((x1)**2))") {
            poly_nodes.push_back(j);
        }
    }

    if (poly_nodes.size() >= 2) {
        for (const auto& trans : TARGET_TRANSFORMS) {
            if (!trans.chk(y_tr)) continue;
            vector<double> y_t(N);
            bool valid_t = true;
            for (int i = 0; i < N; ++i) {
                y_t[i] = trans.fwd(y_tr[i]);
                if (!std::isfinite(y_t[i])) { valid_t = false; break; }
            }
            if (!valid_t || calc_var(y_t) < 1e-12) continue;

            int p_k = poly_nodes.size();
            vector<vector<double>> A(N, vector<double>(p_k + 1));
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < p_k; ++j) A[i][j] = F_tr[poly_nodes[j]][i];
                A[i][p_k] = 1.0;
            }
            vector<double> w_poly;
            if (solve_least_squares(A, y_t, w_poly)) {
                // Prune near-zero weights
                vector<int> sub_nodes;
                for (int j = 0; j < p_k; ++j) {
                    if (std::abs(w_poly[j]) > 1e-5) sub_nodes.push_back(poly_nodes[j]);
                }
                if (!sub_nodes.empty()) {
                    int sn = sub_nodes.size();
                    vector<vector<double>> A_sub(N, vector<double>(sn + 1));
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < sn; ++j) A_sub[i][j] = F_tr[sub_nodes[j]][i];
                        A_sub[i][sn] = 1.0;
                    }
                    vector<double> w_sub;
                    if (solve_least_squares(A_sub, y_t, w_sub)) {
                        double res_var = 0.0;
                        bool ok = true;
                        for (int i = 0; i < N; ++i) {
                            double pred_t = 0.0;
                            for (int j = 0; j < sn; ++j) pred_t += w_sub[j] * A_sub[i][j];
                            pred_t += w_sub[sn];
                            double pred_raw = trans.inv(pred_t);
                            if (!std::isfinite(pred_raw)) { ok = false; break; }
                            res_var += (y_tr[i] - pred_raw) * (y_tr[i] - pred_raw);
                        }
                        res_var /= N;
                        double raw_r2 = 1.0 - res_var / var_y_raw;
                        if (ok && raw_r2 > EXACT_TOL) {
                            Model m;
                            m.type = "linear";
                            m.t_name = trans.name;
                            m.t_inv = trans.inv;
                            m.selected_nodes = sub_nodes;
                            m.w = w_sub;
                            m.r2 = raw_r2;
                            m.names_str = "[";
                            for (size_t k = 0; k < sub_nodes.size(); ++k) {
                                m.names_str += "'" + nodes[sub_nodes[k]].name + "'";
                                if (k + 1 < sub_nodes.size()) m.names_str += ", ";
                            }
                            m.names_str += "]";
                            return m;
                        }
                    }
                }
            }
        }
    }

    // --- Tier 3: 2-Term Sparse Basis Pursuit ---
    vector<int> prim_cols;
    for (int j : valid_cols) {
        if (nodes[j].depth <= 1 || nodes[j].name.rfind("((sin", 0) == 0 || nodes[j].name.rfind("((cos", 0) == 0 ||
            nodes[j].name.rfind("((log", 0) == 0 || nodes[j].name.rfind("((exp", 0) == 0) {
            prim_cols.push_back(j);
            if (prim_cols.size() >= 45) break;
        }
    }

    if (prim_cols.size() >= 2) {
        for (size_t ia = 0; ia < prim_cols.size(); ++ia) {
            int i_node = prim_cols[ia];
            for (size_t ib = ia + 1; ib < prim_cols.size(); ++ib) {
                int j_node = prim_cols[ib];
                vector<vector<double>> A2(N, vector<double>(3));
                for (int s = 0; s < N; ++s) {
                    A2[s][0] = F_tr[i_node][s];
                    A2[s][1] = F_tr[j_node][s];
                    A2[s][2] = 1.0;
                }
                vector<double> w2;
                if (solve_least_squares(A2, y_tr, w2)) {
                    double res_var = 0.0;
                    for (int s = 0; s < N; ++s) {
                        double p = w2[0] * A2[s][0] + w2[1] * A2[s][1] + w2[2];
                        res_var += (y_tr[s] - p) * (y_tr[s] - p);
                    }
                    res_var /= N;
                    double raw_r2 = 1.0 - res_var / var_y_raw;
                    if (raw_r2 > EXACT_TOL) {
                        Model m;
                        m.type = "linear";
                        m.t_name = "identity";
                        m.t_inv = [](double t){ return t; };
                        m.selected_nodes = {i_node, j_node};
                        m.w = w2;
                        m.r2 = raw_r2;
                        m.names_str = "['" + nodes[i_node].name + "', '" + nodes[j_node].name + "']";
                        return m;
                    }
                }
            }
        }
    }

    // --- Tier 4: Sparse Rational Solver Block y = P(x) / (1 + Q(x)) ---
    vector<int> poly_rat;
    for (int j : valid_cols) {
        const string& nm = nodes[j].name;
        if (nm == "x" || nm == "x1" || nm == "x2" || nm == "((x)**2)" || nm == "((x)**3)" || nm == "((x1)**2)" || nm == "((x2)**2)") {
            poly_rat.push_back(j);
        }
    }

    if (!poly_rat.empty()) {
        int rk = poly_rat.size();
        int total_params = (rk + 1) + rk;
        vector<vector<double>> M(N, vector<double>(total_params));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < rk; ++j) M[i][j] = F_tr[poly_rat[j]][i];
            M[i][rk] = 1.0;
            for (int j = 0; j < rk; ++j) M[i][rk + 1 + j] = -y_tr[i] * F_tr[poly_rat[j]][i];
        }
        vector<double> theta;
        if (solve_least_squares(M, y_tr, theta)) {
            vector<int> num_n, den_n;
            for (int j = 0; j < rk; ++j) if (std::abs(theta[j]) > 1e-5) num_n.push_back(poly_rat[j]);
            for (int j = 0; j < rk; ++j) if (std::abs(theta[rk + 1 + j]) > 1e-5) den_n.push_back(poly_rat[j]);

            if (!den_n.empty()) {
                int n_num = num_n.size();
                int n_den = den_n.size();
                int n_sub = (n_num + 1) + n_den;
                vector<vector<double>> M_sub(N, vector<double>(n_sub));
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < n_num; ++j) M_sub[i][j] = F_tr[num_n[j]][i];
                    M_sub[i][n_num] = 1.0;
                    for (int j = 0; j < n_den; ++j) M_sub[i][n_num + 1 + j] = -y_tr[i] * F_tr[den_n[j]][i];
                }
                vector<double> theta_sub;
                if (solve_least_squares(M_sub, y_tr, theta_sub)) {
                    vector<double> wn(n_num + 1);
                    for (int j = 0; j <= n_num; ++j) wn[j] = theta_sub[j];
                    vector<double> wd(n_den);
                    for (int j = 0; j < n_den; ++j) wd[j] = theta_sub[n_num + 1 + j];

                    double res_var = 0.0;
                    bool ok = true;
                    for (int i = 0; i < N; ++i) {
                        double num_v = wn[n_num];
                        for (int j = 0; j < n_num; ++j) num_v += wn[j] * F_tr[num_n[j]][i];
                        double den_v = 1.0;
                        for (int j = 0; j < n_den; ++j) den_v += wd[j] * F_tr[den_n[j]][i];
                        if (std::abs(den_v) < 1e-5) { ok = false; break; }
                        double p = num_v / den_v;
                        res_var += (y_tr[i] - p) * (y_tr[i] - p);
                    }
                    res_var /= N;
                    double raw_r2 = 1.0 - res_var / var_y_raw;
                    if (ok && raw_r2 > EXACT_TOL) {
                        Model m;
                        m.type = "rational";
                        m.t_name = "rational";
                        m.num_nodes = num_n;
                        m.den_nodes = den_n;
                        m.w_num = wn;
                        m.w_den = wd;
                        m.r2 = raw_r2;
                        m.names_str = "num:[";
                        for (size_t k = 0; k < num_n.size(); ++k) {
                            m.names_str += "'" + nodes[num_n[k]].name + "'";
                            if (k + 1 < num_n.size()) m.names_str += ", ";
                        }
                        m.names_str += "] / den:1+[";
                        for (size_t k = 0; k < den_n.size(); ++k) {
                            m.names_str += "'" + nodes[den_n[k]].name + "'";
                            if (k + 1 < den_n.size()) m.names_str += ", ";
                        }
                        m.names_str += "]";
                        return m;
                    }
                }
            }
        }
    }

    // Default fallback
    Model m;
    m.type = "none";
    m.r2 = -1.0;
    return m;
}

vector<double> predict_model(const Model& m, const vector<DAGNode>& nodes, const vector<vector<double>>& X) {
    int N = X.size();
    vector<double> out(N, 0.0);
    if (m.type == "constant") {
        for (int i = 0; i < N; ++i) out[i] = m.c0;
        return out;
    }

    vector<vector<double>> F = eval_dag(nodes, X);
    if (m.type == "linear") {
        int sn = m.selected_nodes.size();
        for (int i = 0; i < N; ++i) {
            double p = m.w[sn];
            for (int j = 0; j < sn; ++j) p += m.w[j] * F[m.selected_nodes[j]][i];
            out[i] = m.t_inv(p);
        }
        return out;
    }

    if (m.type == "rational") {
        int n_num = m.num_nodes.size();
        int n_den = m.den_nodes.size();
        for (int i = 0; i < N; ++i) {
            double num_v = m.w_num[n_num];
            for (int j = 0; j < n_num; ++j) num_v += m.w_num[j] * F[m.num_nodes[j]][i];
            double den_v = 1.0;
            for (int j = 0; j < n_den; ++j) den_v += m.w_den[j] * F[m.den_nodes[j]][i];
            out[i] = (std::abs(den_v) > 1e-6) ? (num_v / den_v) : NAN;
        }
        return out;
    }
    return out;
}

// =====================================================================
// 5. Benchmark Problems Registry
// =====================================================================
struct Problem {
    string name;
    string cat;
    function<double(const vector<double>&)> fn;
    pair<double, double> tr_range;
    pair<double, double> te_range;
    int dim;
};

vector<Problem> get_benchmark_problems() {
    vector<Problem> probs;
    auto add1 = [&](string name, string cat, function<double(double)> fn, pair<double, double> tr = {0.2, 2.5}, pair<double, double> te = {2.5, 4.0}) {
        Problem p;
        p.name = name; p.cat = cat;
        p.fn = [fn](const vector<double>& x){ return fn(x[0]); };
        p.tr_range = tr; p.te_range = te; p.dim = 1;
        probs.push_back(p);
    };
    auto add2 = [&](string name, string cat, function<double(double, double)> fn) {
        Problem p;
        p.name = name; p.cat = cat;
        p.fn = [fn](const vector<double>& x){ return fn(x[0], x[1]); };
        p.tr_range = {0.5, 2.5}; p.te_range = {2.5, 4.0}; p.dim = 2;
        probs.push_back(p);
    };

    // Category 1: Polynomials
    add1("p01: 2*x + 1", "Polynomial", [](double x){ return 2*x + 1; });
    add1("p02: -3*x + 4", "Polynomial", [](double x){ return -3*x + 4; });
    add1("p03: x^2", "Polynomial", [](double x){ return x*x; });
    add1("p04: 3*x^2 - 5*x + 2", "Polynomial", [](double x){ return 3*x*x - 5*x + 2; });
    add1("p05: -x^2 + 4*x - 3", "Polynomial", [](double x){ return -x*x + 4*x - 3; });
    add1("p06: x^3", "Polynomial", [](double x){ return x*x*x; });
    add1("p07: x^3 - 2*x^2 + x", "Polynomial", [](double x){ return x*x*x - 2*x*x + x; });
    add1("p08: 2*x^3 + 3*x - 1", "Polynomial", [](double x){ return 2*x*x*x + 3*x - 1; });
    add1("p09: x^4 - 2*x^2", "Polynomial", [](double x){ return std::pow(x, 4) - 2*x*x; });
    add1("p10: (x-1)^2", "Polynomial", [](double x){ return (x-1)*(x-1); });
    add1("p11: (x+2)^3", "Polynomial", [](double x){ return std::pow(x+2, 3); });
    add1("p12: 0.5*x^2 - x + 4", "Polynomial", [](double x){ return 0.5*x*x - x + 4; });
    add1("p13: x^4", "Polynomial", [](double x){ return std::pow(x, 4); });
    add1("p14: x^5 - x", "Polynomial", [](double x){ return std::pow(x, 5) - x; });
    add1("p15: 4*x^2 + 2", "Polynomial", [](double x){ return 4*x*x + 2; });
    add1("p16: (2*x - 1)^2", "Polynomial", [](double x){ return (2*x - 1)*(2*x - 1); });
    add1("p17: 5*x - 10", "Polynomial", [](double x){ return 5*x - 10; });
    add1("p18: -0.2*x^3 + x^2", "Polynomial", [](double x){ return -0.2*x*x*x + x*x; });
    add1("p19: x^2 + 3*x + 5", "Polynomial", [](double x){ return x*x + 3*x + 5; });
    add1("p20: (x-2)*(x-1)*x", "Polynomial", [](double x){ return (x-2)*(x-1)*x; });

    // Category 2: Trigonometric
    add1("t21: sin(x)", "Trigonometric", [](double x){ return std::sin(x); });
    add1("t22: cos(x)", "Trigonometric", [](double x){ return std::cos(x); });
    add1("t23: 2*sin(x) - 3*cos(x)", "Trigonometric", [](double x){ return 2*std::sin(x) - 3*std::cos(x); });
    add1("t24: sin(x)^2", "Trigonometric", [](double x){ return std::pow(std::sin(x), 2); });
    add1("t25: cos(x)^2", "Trigonometric", [](double x){ return std::pow(std::cos(x), 2); });
    add1("t26: sin(x)*cos(x)", "Trigonometric", [](double x){ return std::sin(x)*std::cos(x); });
    add1("t27: sin(2*x)", "Trigonometric", [](double x){ return std::sin(2*x); });
    add1("t28: cos(2*x)", "Trigonometric", [](double x){ return std::cos(2*x); });
    add1("t29: 3*sin(x) + 1", "Trigonometric", [](double x){ return 3*std::sin(x) + 1; });
    add1("t30: sin(x^2)", "Trigonometric", [](double x){ return std::sin(x*x); });
    add1("t31: cos(x^2)", "Trigonometric", [](double x){ return std::cos(x*x); });
    add1("t32: sin(x) + x", "Trigonometric", [](double x){ return std::sin(x) + x; });
    add1("t33: x*cos(x)", "Trigonometric", [](double x){ return x*std::cos(x); });
    add1("t34: sin(x)^3", "Trigonometric", [](double x){ return std::pow(std::sin(x), 3); });
    add1("t35: cos(x) - sin(x)", "Trigonometric", [](double x){ return std::cos(x) - std::sin(x); });
    add1("t36: tan(x)", "Trigonometric", [](double x){ return std::tan(x); }, {0.1, 1.3}, {1.35, 1.5});
    add1("t37: 1 + sin(x)^2", "Trigonometric", [](double x){ return 1 + std::pow(std::sin(x), 2); });
    add1("t38: 2*cos(x) + x^2", "Trigonometric", [](double x){ return 2*std::cos(x) + x*x; });
    add1("t39: sin(x)*cos(x)^2", "Trigonometric", [](double x){ return std::sin(x)*std::pow(std::cos(x), 2); });
    add1("t40: cos(3*x)", "Trigonometric", [](double x){ return std::cos(3*x); });

    // Category 3: Exponential / Log
    add1("e41: exp(x)", "Exponential/Log", [](double x){ return std::exp(x); }, {-2.0, 1.0}, {1.0, 2.0});
    add1("e42: exp(-x)", "Exponential/Log", [](double x){ return std::exp(-x); });
    add1("e43: 2*exp(x) - 1", "Exponential/Log", [](double x){ return 2*std::exp(x) - 1; }, {-1.0, 1.0}, {1.0, 1.8});
    add1("e44: exp(-x^2) [Gaussian]", "Exponential/Log", [](double x){ return std::exp(-x*x); });
    add1("e45: x*exp(-x)", "Exponential/Log", [](double x){ return x*std::exp(-x); });
    add1("e46: exp(sin(x))", "Exponential/Log", [](double x){ return std::exp(std::sin(x)); });
    add1("e47: exp(-0.5*x^2)", "Exponential/Log", [](double x){ return std::exp(-0.5*x*x); });
    add1("e48: log(x)", "Exponential/Log", [](double x){ return std::log(x); }, {0.5, 3.0}, {3.0, 5.0});
    add1("e49: log(1 + x^2)", "Exponential/Log", [](double x){ return std::log(1 + x*x); });
    add1("e50: 2*log(x) + 3", "Exponential/Log", [](double x){ return 2*std::log(x) + 3; }, {0.5, 3.0}, {3.0, 5.0});
    add1("e51: log(x)^2", "Exponential/Log", [](double x){ return std::pow(std::log(x), 2); }, {1.0, 4.0}, {4.0, 6.0});
    add1("e52: x*log(x)", "Exponential/Log", [](double x){ return x*std::log(x); }, {0.5, 3.0}, {3.0, 5.0});
    add1("e53: exp(x) + exp(-x)", "Exponential/Log", [](double x){ return std::exp(x) + std::exp(-x); }, {-1.0, 1.0}, {1.0, 2.0});
    add1("e54: tanh(x)", "Exponential/Log", [](double x){ return std::tanh(x); });
    add1("e55: 1 - exp(-x)", "Exponential/Log", [](double x){ return 1 - std::exp(-x); });
    add1("e56: exp(-2*x)", "Exponential/Log", [](double x){ return std::exp(-2*x); });
    add1("e57: log(1 + x)", "Exponential/Log", [](double x){ return std::log1p(x); });
    add1("e58: exp(cos(x))", "Exponential/Log", [](double x){ return std::exp(std::cos(x)); });
    add1("e59: 3*exp(-x^2) + 2", "Exponential/Log", [](double x){ return 3*std::exp(-x*x) + 2; });
    add1("e60: sqrt(x)*log(x)", "Exponential/Log", [](double x){ return std::sqrt(x)*std::log(x); }, {0.5, 3.0}, {3.0, 5.0});

    // Category 4: Rational & Root
    add1("r61: 1/x", "Rational/Root", [](double x){ return 1.0/x; }, {0.5, 3.0}, {3.0, 5.0});
    add1("r62: 1/(x + 1)", "Rational/Root", [](double x){ return 1.0/(x + 1.0); });
    add1("r63: 1/(1 + x^2)", "Rational/Root", [](double x){ return 1.0/(1.0 + x*x); });
    add1("r64: x/(1 + x^2)", "Rational/Root", [](double x){ return x/(1.0 + x*x); });
    add1("r65: 1/x^2", "Rational/Root", [](double x){ return 1.0/(x*x); }, {0.5, 3.0}, {3.0, 5.0});
    add1("r66: sqrt(x)", "Rational/Root", [](double x){ return std::sqrt(x); }, {0.2, 3.0}, {3.0, 6.0});
    add1("r67: 2*sqrt(x) - 1", "Rational/Root", [](double x){ return 2*std::sqrt(x) - 1; }, {0.2, 3.0}, {3.0, 6.0});
    add1("r68: sqrt(1 + x^2)", "Rational/Root", [](double x){ return std::sqrt(1 + x*x); });
    add1("r69: (x - 1)/(x + 1)", "Rational/Root", [](double x){ return (x - 1.0)/(x + 1.0); });
    add1("r70: x^2 / (1 + x^2)", "Rational/Root", [](double x){ return (x*x)/(1.0 + x*x); });
    add1("r71: 1 / sqrt(x)", "Rational/Root", [](double x){ return 1.0/std::sqrt(x); }, {0.5, 3.0}, {3.0, 6.0});
    add1("r72: (x + 2)/x", "Rational/Root", [](double x){ return (x + 2.0)/x; }, {0.5, 3.0}, {3.0, 5.0});
    add1("r73: 3/(2 + x)", "Rational/Root", [](double x){ return 3.0/(2.0 + x); });
    add1("r74: x*sqrt(x)", "Rational/Root", [](double x){ return x*std::sqrt(x); }, {0.2, 3.0}, {3.0, 5.0});
    add1("r75: 1/(x^2 + 4)", "Rational/Root", [](double x){ return 1.0/(x*x + 4.0); });
    add1("r76: sqrt(x) + 1/x", "Rational/Root", [](double x){ return std::sqrt(x) + 1.0/x; }, {0.5, 3.0}, {3.0, 5.0});
    add1("r77: (2*x + 1)/(x + 2)", "Rational/Root", [](double x){ return (2*x + 1)/(x + 2.0); });
    add1("r78: 1/(1 + exp(-x)) [sigmoid]", "Rational/Root", [](double x){ return 1.0/(1.0 + std::exp(-x)); });
    add1("r79: x^3 / (1 + x^2)", "Rational/Root", [](double x){ return (x*x*x)/(1.0 + x*x); });
    add1("r80: 2/(x^3 + 1)", "Rational/Root", [](double x){ return 2.0/(x*x*x + 1.0); });

    // Category 5: Multi-variable & Deep
    add2("c81: x1 * x2", "Multi/Composite", [](double a, double b){ return a * b; });
    add2("c82: x1^2 + x2^2", "Multi/Composite", [](double a, double b){ return a*a + b*b; });
    add2("c83: 2*x1 - 3*x2 + 1", "Multi/Composite", [](double a, double b){ return 2*a - 3*b + 1; });
    add2("c84: sin(x1) * cos(x2)", "Multi/Composite", [](double a, double b){ return std::sin(a) * std::cos(b); });
    add2("c85: x1 / (1 + x2^2)", "Multi/Composite", [](double a, double b){ return a / (1.0 + b*b); });
    add2("c86: exp(-(x1^2 + x2^2))", "Multi/Composite", [](double a, double b){ return std::exp(-(a*a + b*b)); });
    add2("c87: x1^2 * x2", "Multi/Composite", [](double a, double b){ return a*a * b; });
    add2("c88: (x1 + x2)^2", "Multi/Composite", [](double a, double b){ return (a + b)*(a + b); });
    add2("c89: sin(x1 + x2)", "Multi/Composite", [](double a, double b){ return std::sin(a + b); });
    add2("c90: x1 * exp(-x2)", "Multi/Composite", [](double a, double b){ return a * std::exp(-b); });
    add1("c91: exp(-cos(x^2))", "Multi/Composite", [](double x){ return std::exp(-std::cos(x*x)); });
    add1("c92: sin(log(x))", "Multi/Composite", [](double x){ return std::sin(std::log(x)); }, {1.0, 5.0}, {5.0, 8.0});
    add1("c93: log(1 + sin(x)^2)", "Multi/Composite", [](double x){ return std::log(1 + std::pow(std::sin(x), 2)); });
    add1("c94: sqrt(1 + exp(-x))", "Multi/Composite", [](double x){ return std::sqrt(1 + std::exp(-x)); });
    add1("c95: cos(exp(-x))", "Multi/Composite", [](double x){ return std::cos(std::exp(-x)); });
    add1("c96: sin(x)^2 + cos(x)^2 [=1]", "Multi/Composite", [](double x){ return std::pow(std::sin(x), 2) + std::pow(std::cos(x), 2); });
    add2("c97: (x1*x2)^2", "Multi/Composite", [](double a, double b){ return (a*b)*(a*b); });
    add2("c98: exp(-x1) + sin(x2)", "Multi/Composite", [](double a, double b){ return std::exp(-a) + std::sin(b); });
    add2("c99: (x1^2 - x2^2)", "Multi/Composite", [](double a, double b){ return a*a - b*b; });
    add2("c100: x1 / (x2 + 1)", "Multi/Composite", [](double a, double b){ return a / (b + 1.0); });

    return probs;
}

// =====================================================================
// 6. Main Benchmark Runner
// =====================================================================
int main() {
    auto problems = get_benchmark_problems();
    cout << "Loaded " << problems.size() << " benchmark problems across 5 categories in C++." << endl;

    cout << "Building candidate DAGs..." << endl;
    auto t_dag_0 = chrono::high_resolution_clock::now();
    auto dag_1d = build_candidate_dag(1, 2);
    auto dag_2d = build_candidate_dag(2, 2);
    auto t_dag_1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dag_ms = t_dag_1 - t_dag_0;
    cout << "DAG 1D size: " << dag_1d.size() << " nodes | DAG 2D size: " << dag_2d.size() 
         << " nodes (Built in " << dag_ms.count() << " ms)" << endl;

    cout << "\n" << string(100, '=') << endl;
    cout << left << setw(32) << "Problem Name" << " | "
         << setw(15) << "Category" << " | "
         << setw(14) << "Extrap NRMSE" << " | "
         << setw(15) << "Status" << " | "
         << setw(10) << "Time(ms)" << " | "
         << "Discovered Nodes" << endl;
    cout << string(100, '=') << endl;

    int exact_count = 0;
    int high_acc_count = 0;
    int mod_acc_count = 0;
    int fail_count = 0;

    auto t_total_0 = chrono::high_resolution_clock::now();

    for (size_t idx = 0; idx < problems.size(); ++idx) {
        auto prob = problems[idx];
        auto t0 = chrono::high_resolution_clock::now();
        int dim = prob.dim;
        const auto& nodes = (dim > 1) ? dag_2d : dag_1d;

        int N_tr = (dim == 1) ? 80 : 100;
        int N_te = (dim == 1) ? 80 : 100;

        mt19937 rng(42 + idx + 1);
        uniform_real_distribution<double> dist_tr(prob.tr_range.first, prob.tr_range.second);
        uniform_real_distribution<double> dist_te(prob.te_range.first, prob.te_range.second);

        vector<vector<double>> X_tr(N_tr, vector<double>(dim));
        vector<double> y_tr(N_tr);
        for (int i = 0; i < N_tr; ++i) {
            for (int d = 0; d < dim; ++d) X_tr[i][d] = dist_tr(rng);
            y_tr[i] = prob.fn(X_tr[i]);
        }

        vector<vector<double>> X_te(N_te, vector<double>(dim));
        vector<double> y_te(N_te);
        for (int i = 0; i < N_te; ++i) {
            for (int d = 0; d < dim; ++d) X_te[i][d] = dist_te(rng);
            y_te[i] = prob.fn(X_te[i]);
        }

        Model m = solve_math_program(X_tr, y_tr, nodes);
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed_ms = t1 - t0;

        double extrap_nrmse = 999.0;
        string status = "FAILED";

        if (m.type != "none") {
            vector<double> y_pred = predict_model(m, nodes, X_te);
            double s_te = std::sqrt(calc_var(y_te));
            if (s_te < 1e-9) s_te = std::abs(calc_mean(y_te)) + 1e-6;

            double mse = 0.0;
            bool finite = true;
            for (int i = 0; i < N_te; ++i) {
                if (!std::isfinite(y_pred[i])) { finite = false; break; }
                mse += (y_te[i] - y_pred[i]) * (y_te[i] - y_pred[i]);
            }
            if (finite) {
                extrap_nrmse = std::sqrt(mse / N_te) / s_te;
            }
        }

        if (extrap_nrmse < 1e-4) {
            status = "EXACT (0-error)";
            exact_count++;
        } else if (extrap_nrmse < 0.05) {
            status = "HIGH ACC (<5%)";
            high_acc_count++;
        } else if (extrap_nrmse < 0.20) {
            status = "MODERATE (<20%)";
            mod_acc_count++;
        } else {
            status = "POOR/FAILED";
            fail_count++;
        }

        cout << left << setw(32) << prob.name << " | "
             << setw(15) << prob.cat << " | "
             << scientific << setprecision(2) << setw(14) << extrap_nrmse << " | "
             << defaultfloat << setw(15) << status << " | "
             << fixed << setprecision(2) << setw(10) << elapsed_ms.count() << " | "
             << m.names_str << endl;
    }

    auto t_total_1 = chrono::high_resolution_clock::now();
    chrono::duration<double> total_sec = t_total_1 - t_total_0;

    cout << "\n" << string(100, '=') << endl;
    cout << "FINAL C++ BENCHMARK SUMMARY (100 Mathematical Problems)" << endl;
    cout << string(100, '=') << endl;
    cout << "Total Problems:                100" << endl;
    cout << "Exact Analytical (NRMSE<1e-4): " << exact_count << " / 100 (" << exact_count << "%)" << endl;
    cout << "High Accuracy   (NRMSE<0.05):  " << high_acc_count << " / 100 (" << high_acc_count << "%)" << endl;
    cout << "Moderate Acc    (NRMSE<0.20):  " << mod_acc_count << " / 100 (" << mod_acc_count << "%)" << endl;
    cout << "Failed / Poor   (NRMSE>=0.20): " << fail_count << " / 100 (" << fail_count << "%)" << endl;
    cout << "Total C++ Benchmark Runtime:   " << fixed << setprecision(3) << total_sec.count() 
         << " seconds (Avg " << (total_sec.count() / 100.0 * 1000.0) << " ms / problem)" << endl;
    cout << string(100, '=') << endl;

    return 0;
}
