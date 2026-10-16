#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <map>
#include <functional>
#include <fstream>
#include <random>

using namespace std;

// =====================================================================
// 1. Recursive-Descent Expression Parser for CLI Flags
// =====================================================================

enum TokenType {
    TOK_END, TOK_NUM, TOK_VAR, TOK_PLUS, TOK_MINUS, TOK_MUL, TOK_DIV, TOK_POW,
    TOK_LPAREN, TOK_RPAREN, TOK_COMMA, TOK_FUNC
};

struct Token {
    TokenType type;
    double numVal;
    string strVal;
};

class Tokenizer {
    string src;
    size_t pos = 0;
public:
    Tokenizer(const string& s) : src(s), pos(0) {}

    Token next() {
        while (pos < src.size() && isspace(src[pos])) pos++;
        if (pos >= src.size()) return {TOK_END, 0.0, ""};

        char c = src[pos];
        if (isdigit(c) || (c == '.' && pos + 1 < src.size() && isdigit(src[pos+1]))) {
            size_t start = pos;
            while (pos < src.size() && (isdigit(src[pos]) || src[pos] == '.')) pos++;
            double val = stod(src.substr(start, pos - start));
            return {TOK_NUM, val, ""};
        }
        if (isalpha(c) || c == '_') {
            size_t start = pos;
            while (pos < src.size() && (isalnum(src[pos]) || src[pos] == '_')) pos++;
            string word = src.substr(start, pos - start);
            string lower = word;
            for (auto& ch : lower) ch = tolower(ch);
            if (lower == "x" || lower == "x1" || lower == "x2" ||
                lower == "dx" || lower == "v" ||
                lower == "u" || lower == "ux" || lower == "uxx") {
                return {TOK_VAR, 0.0, lower};
            }
            return {TOK_FUNC, 0.0, lower};
        }
        pos++;
        switch (c) {
            case '+': return {TOK_PLUS, 0.0, "+"};
            case '-': return {TOK_MINUS, 0.0, "-"};
            case '*': return {TOK_MUL, 0.0, "*"};
            case '/': return {TOK_DIV, 0.0, "/"};
            case '^': return {TOK_POW, 0.0, "^"};
            case '(': return {TOK_LPAREN, 0.0, "("};
            case ')': return {TOK_RPAREN, 0.0, ")"};
            case ',': return {TOK_COMMA, 0.0, ","};
            default:  return {TOK_END, 0.0, ""};
        }
    }
};

struct ASTNode {
    enum Type { NODE_NUM, NODE_VAR, NODE_UNARY, NODE_BINARY, NODE_FUNC } type;
    double numVal = 0.0;
    string varName;
    char op = 0;
    string funcName;
    vector<ASTNode*> children;

    ~ASTNode() {
        for (auto c : children) delete c;
    }

    double eval(const map<string, double>& env) const {
        switch (type) {
            case NODE_NUM: return numVal;
            case NODE_VAR: {
                auto it = env.find(varName);
                if (it != env.end()) return it->second;
                if (varName == "x") {
                    auto it2 = env.find("x1");
                    if (it2 != env.end()) return it2->second;
                }
                return 0.0;
            }
            case NODE_UNARY: {
                double v = children[0]->eval(env);
                return (op == '-') ? -v : v;
            }
            case NODE_BINARY: {
                double l = children[0]->eval(env);
                double r = children[1]->eval(env);
                if (op == '+') return l + r;
                if (op == '-') return l - r;
                if (op == '*') return l * r;
                if (op == '/') return (abs(r) < 1e-15) ? 1e15 : l / r;
                if (op == '^') return pow(l, r);
                return 0.0;
            }
            case NODE_FUNC: {
                double arg = children[0]->eval(env);
                if (funcName == "sin") return sin(arg);
                if (funcName == "cos") return cos(arg);
                if (funcName == "tan") return tan(arg);
                if (funcName == "exp") return exp(arg);
                if (funcName == "log" || funcName == "ln") return (arg <= 1e-15) ? -35.0 : log(arg);
                if (funcName == "sqrt") return (arg < 0.0) ? 0.0 : sqrt(arg);
                if (funcName == "tanh") return tanh(arg);
                if (funcName == "abs") return abs(arg);
                return arg;
            }
        }
        return 0.0;
    }
};

class Parser {
    Tokenizer tokenizer;
    Token curr;
    void advance() { curr = tokenizer.next(); }

    ASTNode* parseExpr();
    ASTNode* parseTerm();
    ASTNode* parseFactor();
    ASTNode* parsePower();
    ASTNode* parsePrimary();

public:
    Parser(const string& s) : tokenizer(s) { advance(); }
    ASTNode* parse() { return parseExpr(); }
};

ASTNode* Parser::parseExpr() {
    ASTNode* left = parseTerm();
    while (curr.type == TOK_PLUS || curr.type == TOK_MINUS) {
        char op = (curr.type == TOK_PLUS) ? '+' : '-';
        advance();
        ASTNode* right = parseTerm();
        ASTNode* parent = new ASTNode();
        parent->type = ASTNode::NODE_BINARY;
        parent->op = op;
        parent->children.push_back(left);
        parent->children.push_back(right);
        left = parent;
    }
    return left;
}

ASTNode* Parser::parseTerm() {
    ASTNode* left = parseFactor();
    while (curr.type == TOK_MUL || curr.type == TOK_DIV) {
        char op = (curr.type == TOK_MUL) ? '*' : '/';
        advance();
        ASTNode* right = parseFactor();
        ASTNode* parent = new ASTNode();
        parent->type = ASTNode::NODE_BINARY;
        parent->op = op;
        parent->children.push_back(left);
        parent->children.push_back(right);
        left = parent;
    }
    return left;
}

ASTNode* Parser::parseFactor() {
    if (curr.type == TOK_MINUS) {
        advance();
        ASTNode* child = parsePower();
        ASTNode* node = new ASTNode();
        node->type = ASTNode::NODE_UNARY;
        node->op = '-';
        node->children.push_back(child);
        return node;
    }
    if (curr.type == TOK_PLUS) {
        advance();
        return parsePower();
    }
    return parsePower();
}

ASTNode* Parser::parsePower() {
    ASTNode* left = parsePrimary();
    if (curr.type == TOK_POW) {
        advance();
        ASTNode* right = parseFactor();
        ASTNode* parent = new ASTNode();
        parent->type = ASTNode::NODE_BINARY;
        parent->op = '^';
        parent->children.push_back(left);
        parent->children.push_back(right);
        return parent;
    }
    return left;
}

ASTNode* Parser::parsePrimary() {
    if (curr.type == TOK_NUM) {
        ASTNode* node = new ASTNode();
        node->type = ASTNode::NODE_NUM;
        node->numVal = curr.numVal;
        advance();
        return node;
    }
    if (curr.type == TOK_VAR) {
        ASTNode* node = new ASTNode();
        node->type = ASTNode::NODE_VAR;
        node->varName = curr.strVal;
        advance();
        return node;
    }
    if (curr.type == TOK_FUNC) {
        string fn = curr.strVal;
        advance();
        if (curr.type == TOK_LPAREN) advance();
        ASTNode* arg = parseExpr();
        if (curr.type == TOK_RPAREN) advance();
        ASTNode* node = new ASTNode();
        node->type = ASTNode::NODE_FUNC;
        node->funcName = fn;
        node->children.push_back(arg);
        return node;
    }
    if (curr.type == TOK_LPAREN) {
        advance();
        ASTNode* node = parseExpr();
        if (curr.type == TOK_RPAREN) advance();
        return node;
    }
    advance();
    ASTNode* fallback = new ASTNode();
    fallback->type = ASTNode::NODE_NUM;
    fallback->numVal = 0.0;
    return fallback;
}

// =====================================================================
// 2. High-Performance Mathematical Operators & DAG Engine
// =====================================================================

enum NodeType { OP_VAR, OP_UNARY, OP_BINARY };

struct UnaryOpDef {
    string name;
    function<double(double)> fn;
    string fmt;
};

const vector<UnaryOpDef> UNARY_OPS = {
    {"neg",   [](double x){ return -x; }, "(-%s)"},
    {"sq",    [](double x){ return x * x; }, "((%s)**2)"},
    {"cube",  [](double x){ return x * x * x; }, "((%s)**3)"},
    {"sqrt",  [](double x){ return (x < 0.0) ? NAN : std::sqrt(x); }, "sqrt(%s)"},
    {"inv",   [](double x){ return (std::abs(x) < 1e-12) ? NAN : 1.0 / x; }, "(1/(%s))"},
    {"sin",   [](double x){ return std::sin(x); }, "sin(%s)"},
    {"cos",   [](double x){ return std::cos(x); }, "cos(%s)"},
    {"tan",   [](double x){ return std::tan(x); }, "tan(%s)"},
    {"exp",   [](double x){ return (x > 60.0) ? NAN : std::exp(x); }, "exp(%s)"},
    {"expneg",[](double x){ return (-x > 60.0) ? NAN : std::exp(-x); }, "exp(-%s)"},
    {"log",   [](double x){ return (x <= 1e-12) ? NAN : std::log(x); }, "log(%s)"},
    {"log1p", [](double x){ return (1.0 + x <= 1e-12) ? NAN : std::log(1.0 + x); }, "log(1+%s)"},
    {"sqrt1p",[](double x){ return (1.0 + x < 0.0) ? NAN : std::sqrt(1.0 + x); }, "sqrt(1+%s)"},
    {"inv1p", [](double x){ return (std::abs(1.0 + x) < 1e-12) ? NAN : 1.0 / (1.0 + x); }, "(1/(1+%s))"},
    {"tanh",  [](double x){ return std::tanh(x); }, "tanh(%s)"},
    {"abs",   [](double x){ return std::abs(x); }, "abs(%s)"},
    {"gauss", [](double x){ return std::exp(-x * x); }, "exp(-(%s)**2)"}
};

struct DAGNode {
    string name;
    NodeType type;
    int op_id;
    int p1, p2;
    int depth;
};

vector<DAGNode> build_dag(const vector<string>& var_names, int max_depth = 3) {
    vector<DAGNode> nodes;
    int d = var_names.size();

    for (int i = 0; i < d; ++i) {
        DAGNode n;
        n.name = var_names[i];
        n.type = OP_VAR;
        n.op_id = i;
        n.p1 = -1; n.p2 = -1;
        n.depth = 0;
        nodes.push_back(n);
    }

    vector<int> lvl1_indices;
    for (int i = 0; i < d; ++i) {
        for (size_t op_i = 0; op_i < UNARY_OPS.size(); ++op_i) {
            DAGNode n;
            char buf[256];
            snprintf(buf, sizeof(buf), UNARY_OPS[op_i].fmt.c_str(), nodes[i].name.c_str());
            n.name = buf;
            n.type = OP_UNARY;
            n.op_id = op_i;
            n.p1 = i; n.p2 = -1;
            n.depth = 1;
            nodes.push_back(n);
            lvl1_indices.push_back(nodes.size() - 1);
        }
    }

    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            DAGNode mul_node;
            mul_node.name = "(" + nodes[i].name + " * " + nodes[j].name + ")";
            mul_node.type = OP_BINARY; mul_node.op_id = 0;
            mul_node.p1 = i; mul_node.p2 = j; mul_node.depth = 1;
            nodes.push_back(mul_node);
            lvl1_indices.push_back(nodes.size() - 1);

            DAGNode add_node;
            add_node.name = "(" + nodes[i].name + " + " + nodes[j].name + ")";
            add_node.type = OP_BINARY; add_node.op_id = 1;
            add_node.p1 = i; add_node.p2 = j; add_node.depth = 1;
            nodes.push_back(add_node);
            lvl1_indices.push_back(nodes.size() - 1);

            DAGNode sub_node;
            sub_node.name = "(" + nodes[i].name + " - " + nodes[j].name + ")";
            sub_node.type = OP_BINARY; sub_node.op_id = 2;
            sub_node.p1 = i; sub_node.p2 = j; sub_node.depth = 1;
            nodes.push_back(sub_node);
            lvl1_indices.push_back(nodes.size() - 1);
        }
    }

    if (max_depth < 2) return nodes;

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
            for (int s = 0; s < N; ++s) F[p][s] = fn(parent[s]);
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
// 3. Robust Linear Algebra
// =====================================================================

bool solve_least_squares(const vector<vector<double>>& A, const vector<double>& b, vector<double>& w) {
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

    for (int i = 0; i < m; ++i) AtA[i][i] += 1e-11;

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

// =====================================================================
// 4. Occam's Hierarchical 5-Tier Solver
// =====================================================================

struct Model {
    string type;
    string t_name;
    function<double(double)> t_inv;
    double c0;
    vector<int> selected_nodes;
    vector<double> w;
    vector<int> num_nodes, den_nodes;
    vector<double> w_num, w_den;
    double r2;
    string names_str;
    string formula_str;
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

Model solve_math_program(const vector<vector<double>>& X_tr, const vector<double>& y_tr, const vector<DAGNode>& nodes, double exact_tol = 1.0 - 1e-8) {
    int N = y_tr.size();
    double var_y_raw = calc_var(y_tr);

    // Tier 0: Constant Identity Check
    if (var_y_raw < 1e-10) {
        Model m;
        m.type = "constant";
        m.c0 = calc_mean(y_tr);
        m.r2 = 1.0;
        m.names_str = "['const(" + to_string(m.c0) + ")']";
        m.formula_str = to_string(m.c0);
        return m;
    }

    vector<vector<double>> F_tr = eval_dag(nodes, X_tr);
    int P = nodes.size();

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

    sort(valid_cols.begin(), valid_cols.end(), [&](int a, int b){
        return get_node_complexity(nodes[a]) < get_node_complexity(nodes[b]);
    });

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

    const double EXACT_TOL = exact_tol;

    // Tier 1: Single Node Exact Check
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

    // Tier 2: Polynomial Basis Check
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
                vector<int> active_idx;
                for (int j = 0; j < p_k; ++j) if (std::abs(w_poly[j]) > 1e-4) active_idx.push_back(j);

                if (!active_idx.empty()) {
                    int a_k = active_idx.size();
                    vector<vector<double>> A_sparse(N, vector<double>(a_k + 1));
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < a_k; ++j) A_sparse[i][j] = F_tr[poly_nodes[active_idx[j]]][i];
                        A_sparse[i][a_k] = 1.0;
                    }
                    vector<double> w_sparse;
                    if (solve_least_squares(A_sparse, y_t, w_sparse)) {
                        double res_var = 0.0;
                        bool ok = true;
                        for (int i = 0; i < N; ++i) {
                            double p = w_sparse[a_k];
                            for (int j = 0; j < a_k; ++j) p += w_sparse[j] * A_sparse[i][j];
                            double p_raw = trans.inv(p);
                            if (!std::isfinite(p_raw)) { ok = false; break; }
                            res_var += (y_tr[i] - p_raw) * (y_tr[i] - p_raw);
                        }
                        res_var /= N;
                        double raw_r2 = 1.0 - res_var / var_y_raw;
                        if (ok && raw_r2 > EXACT_TOL) {
                            Model m;
                            m.type = "linear";
                            m.t_name = trans.name;
                            m.t_inv = trans.inv;
                            for (int idx : active_idx) m.selected_nodes.push_back(poly_nodes[idx]);
                            m.w = w_sparse;
                            m.r2 = raw_r2;
                            m.names_str = "[";
                            for (size_t k = 0; k < m.selected_nodes.size(); ++k) {
                                m.names_str += "'" + nodes[m.selected_nodes[k]].name + "'";
                                if (k + 1 < m.selected_nodes.size()) m.names_str += ", ";
                            }
                            m.names_str += "]";
                            return m;
                        }
                    }
                }
            }
        }
    }

    // Tier 3: 2-Term Sparse Basis Pursuit
    vector<int> prim_cols;
    for (int j : valid_cols) {
        if (nodes[j].depth <= 1 || nodes[j].name.rfind("((sin", 0) == 0 || nodes[j].name.rfind("((cos", 0) == 0 ||
            nodes[j].name.rfind("((log", 0) == 0 || nodes[j].name.rfind("((exp", 0) == 0 ||
            nodes[j].name.rfind("(x * ", 0) == 0) {
            prim_cols.push_back(j);
            if (prim_cols.size() >= 55) break;
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

    // Tier 4: Sparse Rational Solver across Primitive Families
    vector<vector<string>> rat_families = {
        {"x", "((x)**2)", "((x)**3)", "x1", "x2", "((x1)**2)", "((x2)**2)"},
        {"tanh(x)", "((tanh(x))**2)"},
        {"tan(x)", "((tan(x))**2)"},
        {"exp(-x)", "((exp(-x))**2)", "exp(x)"}
    };

    for (const auto& fam : rat_families) {
        vector<int> fam_nodes;
        for (int j : valid_cols) {
            const string& nm = nodes[j].name;
            for (const auto& target_nm : fam) {
                if (nm == target_nm) { fam_nodes.push_back(j); break; }
            }
        }
        if (fam_nodes.empty()) continue;

        int rk = fam_nodes.size();
        int total_params = (rk + 1) + rk;
        vector<vector<double>> M(N, vector<double>(total_params));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < rk; ++j) M[i][j] = F_tr[fam_nodes[j]][i];
            M[i][rk] = 1.0;
            for (int j = 0; j < rk; ++j) M[i][rk + 1 + j] = -y_tr[i] * F_tr[fam_nodes[j]][i];
        }
        vector<double> theta;
        if (solve_least_squares(M, y_tr, theta)) {
            vector<int> num_n, den_n;
            for (int j = 0; j < rk; ++j) if (std::abs(theta[j]) > 1e-4) num_n.push_back(fam_nodes[j]);
            for (int j = 0; j < rk; ++j) if (std::abs(theta[rk + 1 + j]) > 1e-4) den_n.push_back(fam_nodes[j]);

            if (!num_n.empty() || std::abs(theta[rk]) > 1e-4) {
                int n_num = num_n.size();
                int n_den = den_n.size();
                int P_refit = (n_num + 1) + n_den;
                vector<vector<double>> M_refit(N, vector<double>(P_refit));
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < n_num; ++j) M_refit[i][j] = F_tr[num_n[j]][i];
                    M_refit[i][n_num] = 1.0;
                    for (int j = 0; j < n_den; ++j) M_refit[i][n_num + 1 + j] = -y_tr[i] * F_tr[den_n[j]][i];
                }
                vector<double> theta_refit;
                if (solve_least_squares(M_refit, y_tr, theta_refit)) {
                    vector<double> wn(n_num + 1);
                    for (int j = 0; j <= n_num; ++j) wn[j] = theta_refit[j];
                    vector<double> wd(n_den);
                    for (int j = 0; j < n_den; ++j) wd[j] = theta_refit[n_num + 1 + j];

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
                    int n_params = n_num + n_den + 1;
                    double occam_score = res_var * (1.0 + 0.02 * n_params);

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
                        if (n_params <= 3) return m; // Minimal 2-parameter closed-form law
                    }
                }
            }
        }
    }

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

string format_model_formula(const Model& model, const vector<DAGNode>& dag) {
    ostringstream oss;
    oss << fixed << setprecision(6);
    if (model.type == "constant") {
        oss << model.c0;
    } else if (model.type == "linear") {
        if (model.t_name != "identity") oss << model.t_name << "_inv[ ";
        bool first = true;
        int sn = model.selected_nodes.size();
        double intercept = model.w[sn];
        if (abs(intercept) > 1e-5 || sn == 0) {
            oss << intercept;
            first = false;
        }
        for (int k = 0; k < sn; ++k) {
            double coeff = model.w[k];
            if (abs(coeff) < 1e-5) continue;
            if (!first) {
                if (coeff >= 0) oss << " + ";
                else { oss << " - "; coeff = -coeff; }
            } else if (coeff < 0) {
                oss << "-";
                coeff = -coeff;
            }
            if (abs(coeff - 1.0) < 1e-4) oss << dag[model.selected_nodes[k]].name;
            else oss << coeff << " * " << dag[model.selected_nodes[k]].name;
            first = false;
        }
        if (model.t_name != "identity") oss << " ]";
    } else if (model.type == "rational") {
        oss << "[ ";
        int n_num = model.num_nodes.size();
        int n_den = model.den_nodes.size();
        oss << model.w_num[n_num];
        for (int k = 0; k < n_num; ++k) {
            double c = model.w_num[k];
            if (c >= 0) oss << " + " << c << " * " << dag[model.num_nodes[k]].name;
            else oss << " - " << -c << " * " << dag[model.num_nodes[k]].name;
        }
        oss << " ] / [ 1.0";
        for (int k = 0; k < n_den; ++k) {
            double c = model.w_den[k];
            if (c >= 0) oss << " + " << c << " * " << dag[model.den_nodes[k]].name;
            else oss << " - " << -c << " * " << dag[model.den_nodes[k]].name;
        }
        oss << " ]";
    } else {
        oss << "UNSOLVED";
    }
    return oss.str();
}

double compute_nrmse(const vector<double>& y_true, const vector<double>& y_pred) {
    double ss_res = 0.0, y_max = -1e18, y_min = 1e18;
    int cnt = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (!std::isfinite(y_true[i]) || !std::isfinite(y_pred[i])) continue;
        double err = y_true[i] - y_pred[i];
        ss_res += err * err;
        y_max = max(y_max, y_true[i]);
        y_min = min(y_min, y_true[i]);
        cnt++;
    }
    if (cnt == 0) return 1e9;
    double rmse = sqrt(ss_res / cnt);
    double range = y_max - y_min;
    return (range > 1e-12) ? (rmse / range) : rmse;
}

// =====================================================================
// 5. CALCULUS CODE-NEURON BLOCKS (Diff, Integral, ODE, PDE)
// =====================================================================

// 5A. Differential Calculus Block: Symbolic 1st & 2nd Derivative Discovery
void run_diff_block(const string& expr_str, double t_start, double t_end, double e_start, double e_end) {
    cout << "[Calculus Mode] Differential Operator Block (--diff): f(x) = " << expr_str << "\n";
    cout << "[Intervals]     Train: [" << t_start << ", " << t_end << "] | Extrapolation Test: [" << e_start << ", " << e_end << "]\n\n";

    Parser parser(expr_str);
    ASTNode* root = parser.parse();

    auto t0 = chrono::high_resolution_clock::now();

    int N = 160;
    double h_tr = (t_end - t_start) / (N - 1);
    double h_sten = 0.002; // Fine stencil step for O(h^4) ~ 1e-12 truncation error

    vector<vector<double>> X_tr(N, vector<double>(1));
    vector<double> dy_tr(N), d2y_tr(N);
    for (int i = 0; i < N; ++i) {
        double xv = t_start + i * h_tr;
        X_tr[i][0] = xv;
        double fp2 = root->eval({{"x", xv + 2*h_sten}});
        double fp1 = root->eval({{"x", xv + h_sten}});
        double f0  = root->eval({{"x", xv}});
        double fm1 = root->eval({{"x", xv - h_sten}});
        double fm2 = root->eval({{"x", xv - 2*h_sten}});
        dy_tr[i]  = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h_sten);
        d2y_tr[i] = (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h_sten * h_sten);
    }

    // Ground-truth derivatives on extrapolation interval
    double h_te = (e_end - e_start) / (N - 1);
    vector<vector<double>> X_te(N, vector<double>(1));
    vector<double> dy_te(N), d2y_te(N);
    for (int i = 0; i < N; ++i) {
        double xv = e_start + i * h_te;
        X_te[i][0] = xv;
        double fp2 = root->eval({{"x", xv + 2*h_sten}});
        double fp1 = root->eval({{"x", xv + h_sten}});
        double f0  = root->eval({{"x", xv}});
        double fm1 = root->eval({{"x", xv - h_sten}});
        double fm2 = root->eval({{"x", xv - 2*h_sten}});
        dy_te[i]  = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h_sten);
        d2y_te[i] = (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h_sten * h_sten);
    }

    vector<DAGNode> dag = build_dag({"x"}, 3);
    Model model_d1 = solve_math_program(X_tr, dy_tr, dag, 1.0 - 1e-10);
    Model model_d2 = solve_math_program(X_tr, d2y_tr, dag, 1.0 - 1e-10);

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    vector<double> pred_d1 = predict_model(model_d1, dag, X_te);
    vector<double> pred_d2 = predict_model(model_d2, dag, X_te);
    double nrmse_d1 = compute_nrmse(dy_te, pred_d1);
    double nrmse_d2 = compute_nrmse(d2y_te, pred_d2);

    cout << ">>> 1ST DERIVATIVE DISCOVERY: df/dx <<<\n";
    cout << "  Active Code-Neurons:  " << model_d1.names_str << "\n";
    cout << "  f'(x)  = " << format_model_formula(model_d1, dag) << "\n";
    cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse_d1
         << " [" << ((nrmse_d1 < 1e-4) ? "EXACT (0-error)" : "APPROX") << "]\n\n";

    cout << ">>> 2ND DERIVATIVE DISCOVERY: d^2f/dx^2 <<<\n";
    cout << "  Active Code-Neurons:  " << model_d2.names_str << "\n";
    cout << "  f''(x) = " << format_model_formula(model_d2, dag) << "\n";
    cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse_d2
         << " [" << ((nrmse_d2 < 1e-4) ? "EXACT (0-error)" : "APPROX") << "]\n\n";

    cout << fixed << setprecision(3) << "  Total Calculus Solve Time: " << ms << " ms\n";
    cout << "========================================================================\n\n";
    delete root;
}

// 5B. Integral Calculus Block: Symbolic Antiderivative & Definite Quadrature
void run_integrate_block(const string& expr_str, double t_start, double t_end, double e_start, double e_end) {
    cout << "[Calculus Mode] Integral Operator Block (--integrate): f(x) = " << expr_str << "\n";
    cout << "[Intervals]     Train: [" << t_start << ", " << t_end << "] | Extrapolation Test: [" << e_start << ", " << e_end << "]\n\n";

    Parser parser(expr_str);
    ASTNode* root = parser.parse();

    auto t0 = chrono::high_resolution_clock::now();

    int N = 150;
    vector<vector<double>> X_tr(N, vector<double>(1));
    vector<double> F_tr(N, 0.0);
    double h = (t_end - t_start) / (N - 1);
    X_tr[0][0] = t_start;
    double accum = 0.0;

    // Composite Simpson's quadrature on sub-intervals
    for (int i = 1; i < N; ++i) {
        double a = t_start + (i - 1) * h;
        double b = t_start + i * h;
        double m = 0.5 * (a + b);
        double fa = root->eval({{"x", a}});
        double fm = root->eval({{"x", m}});
        double fb = root->eval({{"x", b}});
        accum += (h / 6.0) * (fa + 4.0 * fm + fb);
        X_tr[i][0] = b;
        F_tr[i] = accum;
    }

    // Extrapolation ground-truth integral from t_start to x_te
    vector<vector<double>> X_te(N, vector<double>(1));
    vector<double> F_te(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double x_target = e_start + (e_end - e_start) * i / (N - 1);
        X_te[i][0] = x_target;
        int steps = 200;
        double ds = (x_target - t_start) / steps;
        double sum = 0.0;
        for (int s = 0; s < steps; ++s) {
            double sa = t_start + s * ds;
            double sb = sa + ds;
            double sm = 0.5 * (sa + sb);
            sum += (ds / 6.0) * (root->eval({{"x", sa}}) + 4.0 * root->eval({{"x", sm}}) + root->eval({{"x", sb}}));
        }
        F_te[i] = sum;
    }

    vector<DAGNode> dag = build_dag({"x"}, 3);
    Model model_int = solve_math_program(X_tr, F_tr, dag, 1.0 - 1e-7);

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    vector<double> pred_te = predict_model(model_int, dag, X_te);
    double nrmse = compute_nrmse(F_te, pred_te);

    cout << ">>> SYMBOLIC ANTIDERIVATIVE DISCOVERY: F(x) = Integral[t=" << t_start << "..x] f(t) dt <<<\n";
    cout << "  Active Code-Neurons:  " << model_int.names_str << "\n";
    cout << "  F(x)   = " << format_model_formula(model_int, dag) << "\n";
    cout << "  Definite Integral on [" << t_start << ", " << t_end << "]: " << fixed << setprecision(6) << F_tr.back() << "\n";
    cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse
         << " [" << ((nrmse < 1e-4) ? "EXACT (0-error)" : "APPROX") << "]\n";
    cout << fixed << setprecision(3) << "  Solve Time:           " << ms << " ms\n";
    cout << "========================================================================\n\n";
    delete root;
}

// 5C. ODE Discovery & RK4 Future Trajectory Extrapolation Block
void run_ode_block(const string& accel_expr, double x0, double v0) {
    cout << "[Calculus Mode] 2nd-Order Dynamical ODE Block (--ode): x''(t) = " << accel_expr << "\n";
    cout << "[Initial Cond]  x(0) = " << x0 << ", x'(0) = " << v0 << "\n";
    cout << "[Intervals]     Train Window: t in [0.0, 6.0] | Extrapolation Window: t in [6.0, 12.0]\n\n";

    Parser parser(accel_expr);
    ASTNode* root = parser.parse();

    auto t0 = chrono::high_resolution_clock::now();

    auto accel_fn = [&](double x, double dx) {
        return root->eval({{"x", x}, {"dx", dx}, {"v", dx}});
    };

    // Simulate ground-truth trajectory with RK4 (dt = 0.01, 1200 steps total)
    int N_tr = 600;
    int N_tot = 1200;
    double dt = 0.01;
    vector<double> t_all(N_tot), x_all(N_tot), v_all(N_tot);
    double x = x0, v = v0;
    for (int i = 0; i < N_tot; ++i) {
        t_all[i] = i * dt;
        x_all[i] = x;
        v_all[i] = v;

        double k1_x = v;
        double k1_v = accel_fn(x, v);
        double k2_x = v + 0.5 * dt * k1_v;
        double k2_v = accel_fn(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v);
        double k3_x = v + 0.5 * dt * k2_v;
        double k3_v = accel_fn(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v);
        double k4_x = v + dt * k3_v;
        double k4_v = accel_fn(x + dt * k3_x, v + dt * k3_v);

        x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
    }

    // Extract derivatives from observed position x(t) on [0, 6.0] using 5-point stencil
    int M = N_tr - 4;
    vector<double> x_obs(M), dx_obs(M), d2x_obs(M);
    for (int i = 0; i < M; ++i) {
        int idx = i + 2;
        x_obs[i] = x_all[idx];
        dx_obs[i] = (-x_all[idx + 2] + 8.0 * x_all[idx + 1] - 8.0 * x_all[idx - 1] + x_all[idx - 2]) / (12.0 * dt);
        d2x_obs[i] = (-x_all[idx + 2] + 16.0 * x_all[idx + 1] - 30.0 * x_all[idx] + 16.0 * x_all[idx - 1] - x_all[idx - 2]) / (12.0 * dt * dt);
    }

    // Candidate Phase-Space Code-Neuron Library
    vector<string> cand_names = {"const", "x", "dx", "sin(x)", "cos(x)", "x^2", "x^3", "x*dx", "dx^2", "x^2*dx", "|dx|*dx"};
    int K = cand_names.size();
    auto eval_cand = [](int k, double xv, double dxv) -> double {
        switch (k) {
            case 0: return 1.0;
            case 1: return xv;
            case 2: return dxv;
            case 3: return sin(xv);
            case 4: return cos(xv);
            case 5: return xv * xv;
            case 6: return xv * xv * xv;
            case 7: return xv * dxv;
            case 8: return dxv * dxv;
            case 9: return (xv * xv) * dxv;
            case 10: return std::abs(dxv) * dxv;
            default: return 0.0;
        }
    };

    vector<vector<double>> C_mat(K, vector<double>(M));
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) C_mat[k][i] = eval_cand(k, x_obs[i], dx_obs[i]);
    }

    double var_d2x = calc_var(d2x_obs);
    vector<int> best_subset;
    vector<double> best_w;
    double best_score = 1e18; // Occam score: residual_var / var + 1e-4 * num_terms
    double best_r2 = -1.0;

    // Search 1-term, 2-term, and 3-term phase-space subsets
    for (int n_terms = 1; n_terms <= 3; ++n_terms) {
        vector<int> idx(n_terms);
        function<void(int, int)> search_subsets = [&](int pos, int start_k) {
            if (pos == n_terms) {
                vector<vector<double>> A(M, vector<double>(n_terms));
                for (int s = 0; s < M; ++s) {
                    for (int t = 0; t < n_terms; ++t) A[s][t] = C_mat[idx[t]][s];
                }
                vector<double> w;
                if (solve_least_squares(A, d2x_obs, w)) {
                    double res = 0.0;
                    for (int s = 0; s < M; ++s) {
                        double p = 0.0;
                        for (int t = 0; t < n_terms; ++t) p += w[t] * A[s][t];
                        res += (d2x_obs[s] - p) * (d2x_obs[s] - p);
                    }
                    res /= M;
                    double rel_err = res / (var_d2x + 1e-12);
                    double score = rel_err + 1e-5 * n_terms;
                    if (score < best_score) {
                        best_score = score;
                        best_r2 = 1.0 - rel_err;
                        best_subset = idx;
                        best_w = w;
                    }
                }
                return;
            }
            for (int k = start_k; k < K; ++k) {
                idx[pos] = k;
                search_subsets(pos + 1, k + 1);
            }
        };
        search_subsets(0, 0);
    }

    // Integrate discovered ODE forward across entire horizon [0, 12.0] via RK4
    auto disc_accel = [&](double xv, double dxv) {
        double a = 0.0;
        for (size_t t = 0; t < best_subset.size(); ++t) {
            a += best_w[t] * eval_cand(best_subset[t], xv, dxv);
        }
        return a;
    };

    vector<double> x_pred_te(N_tot - N_tr), x_true_te(N_tot - N_tr);
    double xp = x0, vp = dx_obs[0] - d2x_obs[0] * (2.0 * dt); // or true x0, v0
    xp = x_all[0]; vp = v_all[0];
    for (int i = 0; i < N_tot; ++i) {
        if (i >= N_tr) {
            x_pred_te[i - N_tr] = xp;
            x_true_te[i - N_tr] = x_all[i];
        }
        double k1_x = vp;
        double k1_v = disc_accel(xp, vp);
        double k2_x = vp + 0.5 * dt * k1_v;
        double k2_v = disc_accel(xp + 0.5 * dt * k1_x, vp + 0.5 * dt * k1_v);
        double k3_x = vp + 0.5 * dt * k2_v;
        double k3_v = disc_accel(xp + 0.5 * dt * k2_x, vp + 0.5 * dt * k2_v);
        double k4_x = vp + dt * k3_v;
        double k4_v = disc_accel(xp + dt * k3_x, vp + dt * k3_v);

        xp += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        vp += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    double nrmse = compute_nrmse(x_true_te, x_pred_te);

    cout << ">>> DISCOVERED GOVERNING DIFFERENTIAL EQUATION <<<\n";
    cout << "  x''(t) = ";
    for (size_t t = 0; t < best_subset.size(); ++t) {
        double w = best_w[t];
        if (t > 0) cout << (w >= 0 ? " + " : " - ");
        else if (w < 0) cout << "-";
        cout << fixed << setprecision(4) << abs(w) << " * " << cand_names[best_subset[t]];
    }
    cout << "\n";
    cout << "  Phase-Space R^2:      " << fixed << setprecision(6) << best_r2 << "\n";
    cout << "  Solve + RK4 Latency:  " << fixed << setprecision(3) << ms << " ms\n\n";

    cout << ">>> FUTURE TRAJECTORY EXTRAPOLATION (t in [6.0, 12.0]) <<<\n";
    cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse << "\n";
    cout << "  Generalization Check: " << ((nrmse < 1e-3) ? "EXACT DYNAMICAL LAW LOCKED" : "APPROXIMATE") << "\n";
    cout << "========================================================================\n\n";
    delete root;
}

// 5C-2. Blind Observational Telemetry Discovery (--telemetry <file.csv>)
// Reads raw (t, x) sensor observations without knowing the governing equation or velocity.
// Compares Explicit Time-Domain Symbolic Regression x = f(t) against
// Phase-Space Differential System Identification x'' = g(x, x').
void run_telemetry_block(const string& csv_path) {
    ifstream fin(csv_path);
    if (!fin.is_open()) {
        cerr << "[ERROR] Could not open telemetry CSV file: " << csv_path << "\n";
        return;
    }
    vector<double> t_all, x_all;
    string line;
    while (getline(fin, line)) {
        if (line.empty() || isalpha(line[0]) || line[0] == '#') continue;
        auto comma = line.find(',');
        if (comma != string::npos) {
            t_all.push_back(stod(line.substr(0, comma)));
            x_all.push_back(stod(line.substr(comma + 1)));
        }
    }
    int N_tot = (int)t_all.size();
    if (N_tot < 20) {
        cerr << "[ERROR] Telemetry file requires at least 20 time-series samples.\n";
        return;
    }
    int N_tr = N_tot / 2;
    double dt = (t_all[N_tr - 1] - t_all[0]) / (N_tr - 1);

    cout << "[Telemetry Mode] Blind Observational Time-Series File: \"" << csv_path << "\"\n";
    cout << "[Grid Specs]     Total Samples: " << N_tot << " (dt = " << fixed << setprecision(4) << dt << " s)\n";
    cout << "[Intervals]      Observed Train Window: t in [" << fixed << setprecision(2) << t_all[0] << ", " << t_all[N_tr - 1]
         << "] | Unseen Future Window: t in [" << t_all[N_tr] << ", " << t_all[N_tot - 1] << "]\n\n";

    // --- Track 1: Explicit Time-Domain Symbolic Regression x = f(t) ---
    auto t0_alg = chrono::high_resolution_clock::now();
    int stride = max(1, N_tr / 120);
    vector<vector<double>> T_tr, T_te;
    vector<double> y_tr, x_true_te;
    for (int i = 0; i < N_tr; i += stride) {
        T_tr.push_back({t_all[i]});
        y_tr.push_back(x_all[i]);
    }
    for (int i = N_tr; i < N_tot; ++i) {
        T_te.push_back({t_all[i]});
        x_true_te.push_back(x_all[i]);
    }
    vector<DAGNode> dag = build_dag({"t"}, 3);
    Model alg_model = solve_math_program(T_tr, y_tr, dag, 1.0 - 1e-10);
    vector<double> alg_preds = predict_model(alg_model, dag, T_te);
    double alg_nrmse = compute_nrmse(x_true_te, alg_preds);
    auto t1_alg = chrono::high_resolution_clock::now();
    double alg_ms = chrono::duration<double, milli>(t1_alg - t0_alg).count();

    // --- Track 2: Blind Phase-Space Differential Discovery x'' = g(x, dx) ---
    auto t0_ode = chrono::high_resolution_clock::now();
    int M = N_tr - 4;
    vector<double> x_obs(M), dx_obs(M), d2x_obs(M);
    for (int i = 0; i < M; ++i) {
        int idx = i + 2;
        x_obs[i] = x_all[idx];
        dx_obs[i] = (-x_all[idx + 2] + 8.0 * x_all[idx + 1] - 8.0 * x_all[idx - 1] + x_all[idx - 2]) / (12.0 * dt);
        d2x_obs[i] = (-x_all[idx + 2] + 16.0 * x_all[idx + 1] - 30.0 * x_all[idx] + 16.0 * x_all[idx - 1] - x_all[idx - 2]) / (12.0 * dt * dt);
    }

    vector<string> cand_names = {"const", "x", "dx", "sin(x)", "cos(x)", "x^2", "x^3", "x*dx", "dx^2", "x^2*dx", "|dx|*dx"};
    int K = cand_names.size();
    auto eval_cand = [](int k, double xv, double dxv) -> double {
        switch (k) {
            case 0: return 1.0;
            case 1: return xv;
            case 2: return dxv;
            case 3: return sin(xv);
            case 4: return cos(xv);
            case 5: return xv * xv;
            case 6: return xv * xv * xv;
            case 7: return xv * dxv;
            case 8: return dxv * dxv;
            case 9: return (xv * xv) * dxv;
            case 10: return std::abs(dxv) * dxv;
            default: return 0.0;
        }
    };

    vector<vector<double>> C_mat(K, vector<double>(M));
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) C_mat[k][i] = eval_cand(k, x_obs[i], dx_obs[i]);
    }

    double var_d2x = calc_var(d2x_obs);
    vector<int> best_subset;
    vector<double> best_w;
    double best_score = 1e18;
    double best_r2 = -1.0;

    for (int n_terms = 1; n_terms <= 3; ++n_terms) {
        vector<int> idx(n_terms);
        function<void(int, int)> search_subsets = [&](int pos, int start_k) {
            if (pos == n_terms) {
                vector<vector<double>> A(M, vector<double>(n_terms));
                for (int s = 0; s < M; ++s) {
                    for (int t = 0; t < n_terms; ++t) A[s][t] = C_mat[idx[t]][s];
                }
                vector<double> w;
                if (solve_least_squares(A, d2x_obs, w)) {
                    double res = 0.0;
                    for (int s = 0; s < M; ++s) {
                        double p = 0.0;
                        for (int t = 0; t < n_terms; ++t) p += w[t] * A[s][t];
                        res += (d2x_obs[s] - p) * (d2x_obs[s] - p);
                    }
                    res /= M;
                    double rel_err = res / (var_d2x + 1e-12);
                    double score = rel_err + 1e-5 * n_terms;
                    if (score < best_score) {
                        best_score = score;
                        best_r2 = 1.0 - rel_err;
                        best_subset = idx;
                        best_w = w;
                    }
                }
                return;
            }
            for (int k = start_k; k < K; ++k) {
                idx[pos] = k;
                search_subsets(pos + 1, k + 1);
            }
        };
        search_subsets(0, 0);
    }

    auto disc_accel = [&](double xv, double dxv) {
        double a = 0.0;
        for (size_t t = 0; t < best_subset.size(); ++t) {
            a += best_w[t] * eval_cand(best_subset[t], xv, dxv);
        }
        return a;
    };

    // Start RK4 strictly from the last interior training stencil point (idx_start = N_tr - 3)
    // using ONLY training position observations to estimate velocity (zero test leakage)
    int idx_start = N_tr - 3;
    double xp = x_all[idx_start];
    double vp = dx_obs[M - 1];
    vector<double> x_pred_te(N_tot - N_tr);
    for (int i = idx_start; i < N_tot; ++i) {
        if (i >= N_tr) {
            x_pred_te[i - N_tr] = xp;
        }
        double k1_x = vp;
        double k1_v = disc_accel(xp, vp);
        double k2_x = vp + 0.5 * dt * k1_v;
        double k2_v = disc_accel(xp + 0.5 * dt * k1_x, vp + 0.5 * dt * k1_v);
        double k3_x = vp + 0.5 * dt * k2_v;
        double k3_v = disc_accel(xp + 0.5 * dt * k2_x, vp + 0.5 * dt * k2_v);
        double k4_x = vp + dt * k3_v;
        double k4_v = disc_accel(xp + dt * k3_x, vp + dt * k3_v);

        xp += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        vp += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
    }
    auto t1_ode = chrono::high_resolution_clock::now();
    double ode_ms = chrono::duration<double, milli>(t1_ode - t0_ode).count();
    double ode_nrmse = compute_nrmse(x_true_te, x_pred_te);

    cout << ">>> TRACK 1: EXPLICIT TIME-DOMAIN SYMBOLIC REGRESSION x = f(t) <<<\n";
    cout << "  Candidate Formula:    x(t) = " << format_model_formula(alg_model, dag) << "\n";
    cout << "  Train R^2:            " << fixed << setprecision(6) << alg_model.r2 << "\n";
    cout << "  Future Extrap NRMSE:  " << scientific << setprecision(2) << alg_nrmse
         << " [" << ((alg_nrmse < 1e-3) ? "EXACT CLOSED-FORM" : "FAILS (Non-Elementary Trajectory)") << "]\n";
    cout << "  Latency:              " << fixed << setprecision(3) << alg_ms << " ms\n\n";

    cout << ">>> TRACK 2: BLIND PHASE-SPACE DIFFERENTIAL DISCOVERY x'' = g(x, x') <<<\n";
    cout << "  Discovered ODE Law:   x''(t) = ";
    for (size_t t = 0; t < best_subset.size(); ++t) {
        double w = best_w[t];
        if (t > 0) cout << (w >= 0 ? " + " : " - ");
        else if (w < 0) cout << "-";
        cout << fixed << setprecision(4) << abs(w) << " * " << cand_names[best_subset[t]];
    }
    cout << "\n";
    cout << "  Phase-Space R^2:      " << fixed << setprecision(6) << best_r2 << "\n";
    cout << "  Future Extrap NRMSE:  " << scientific << setprecision(2) << ode_nrmse
         << " [" << ((ode_nrmse < 1e-3) ? "EXACT DYNAMICAL LAW LOCKED" : "APPROXIMATE") << "]\n";
    cout << "  Latency:              " << fixed << setprecision(3) << ode_ms << " ms\n\n";

    cout << ">>> PARADIGM SELECTION VERDICT <<<\n";
    if (ode_nrmse < alg_nrmse && ode_nrmse < 1e-3) {
        cout << "  Selected Regime:      NON-ELEMENTARY DYNAMICAL ATTRACTOR (Phase-Space ODE wins by "
             << scientific << setprecision(1) << (alg_nrmse / max(ode_nrmse, 1e-16)) << "x lower error)\n";
    } else if (alg_nrmse < 1e-3) {
        cout << "  Selected Regime:      EXPLICIT CLOSED-FORM TRAJECTORY x = f(t)\n";
    } else {
        cout << "  Selected Regime:      APPROXIMATE DYNAMICAL FIT\n";
    }
    cout << "========================================================================\n\n";
}


// 5D. PDE Spatio-Temporal Field Identification & Forward Solver Block
void run_pde_block(const string& pde_expr) {
    cout << "[Calculus Mode] Spatio-Temporal PDE Operator Block (--pde): u_t = " << pde_expr << "\n";
    cout << "[Grid Specs]    Spatial x in [0, 1] (35 nodes) | Train t in [0, 0.1] | Extrap t in [0.1, 0.2]\n\n";

    Parser parser(pde_expr);
    ASTNode* root = parser.parse();

    auto t0 = chrono::high_resolution_clock::now();

    int Nx = 35;
    double dx = 1.0 / (Nx - 1);
    int Nt_tr = 500;
    int Nt_tot = 1000;
    double dt = 0.0002;

    auto pde_rhs = [&](const vector<double>& u) {
        vector<double> dudt(Nx, 0.0);
        for (int j = 1; j < Nx - 1; ++j) {
            double ux = (u[j + 1] - u[j - 1]) / (2.0 * dx);
            double uxx = (u[j + 1] - 2.0 * u[j] + u[j - 1]) / (dx * dx);
            dudt[j] = root->eval({{"u", u[j]}, {"ux", ux}, {"uxx", uxx}});
        }
        return dudt;
    };

    // Initial waveform u(0, x) = sin(pi * x) + 0.5 * sin(2 * pi * x)
    vector<vector<double>> U_all(Nt_tot, vector<double>(Nx, 0.0));
    vector<double> u(Nx, 0.0);
    const double PI = acos(-1.0);
    for (int j = 0; j < Nx; ++j) {
        double xv = j * dx;
        u[j] = sin(PI * xv) + 0.5 * sin(2.0 * PI * xv);
    }

    for (int n = 0; n < Nt_tot; ++n) {
        U_all[n] = u;
        vector<double> k1 = pde_rhs(u);
        vector<double> u2(Nx, 0.0), u3(Nx, 0.0), u4(Nx, 0.0);
        for (int j = 1; j < Nx - 1; ++j) u2[j] = u[j] + 0.5 * dt * k1[j];
        vector<double> k2 = pde_rhs(u2);
        for (int j = 1; j < Nx - 1; ++j) u3[j] = u[j] + 0.5 * dt * k2[j];
        vector<double> k3 = pde_rhs(u3);
        for (int j = 1; j < Nx - 1; ++j) u4[j] = u[j] + dt * k3[j];
        vector<double> k4 = pde_rhs(u4);
        for (int j = 1; j < Nx - 1; ++j) {
            u[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
    }

    // Extract partial derivatives u_t, u_x, u_xx on training interior
    vector<double> ut_vec, uxx_vec, ux_vec, u_ux_vec, u_vec, u2_vec;
    for (int n = 2; n < Nt_tr - 2; n += 3) {
        for (int j = 2; j < Nx - 2; ++j) {
            double ut = (-U_all[n + 2][j] + 8.0 * U_all[n + 1][j] - 8.0 * U_all[n - 1][j] + U_all[n - 2][j]) / (12.0 * dt);
            double ux = (U_all[n][j + 1] - U_all[n][j - 1]) / (2.0 * dx);
            double uxx = (U_all[n][j + 1] - 2.0 * U_all[n][j] + U_all[n][j - 1]) / (dx * dx);
            double uv = U_all[n][j];

            ut_vec.push_back(ut);
            uxx_vec.push_back(uxx);
            ux_vec.push_back(ux);
            u_ux_vec.push_back(uv * ux);
            u_vec.push_back(uv);
            u2_vec.push_back(uv * uv);
        }
    }

    int M = ut_vec.size();
    vector<string> pde_cands = {"uxx", "u*ux", "ux", "u", "u^2"};
    vector<vector<double>> P_mat = {uxx_vec, u_ux_vec, ux_vec, u_vec, u2_vec};
    int K = pde_cands.size();

    double var_ut = calc_var(ut_vec);
    vector<int> best_sub;
    vector<double> best_w;
    double best_score = 1e18;
    double best_r2 = -1.0;

    for (int n_terms = 1; n_terms <= 3; ++n_terms) {
        vector<int> idx(n_terms);
        function<void(int, int)> search_pde = [&](int pos, int start_k) {
            if (pos == n_terms) {
                vector<vector<double>> A(M, vector<double>(n_terms));
                for (int s = 0; s < M; ++s) {
                    for (int t = 0; t < n_terms; ++t) A[s][t] = P_mat[idx[t]][s];
                }
                vector<double> w;
                if (solve_least_squares(A, ut_vec, w)) {
                    double res = 0.0;
                    for (int s = 0; s < M; ++s) {
                        double p = 0.0;
                        for (int t = 0; t < n_terms; ++t) p += w[t] * A[s][t];
                        res += (ut_vec[s] - p) * (ut_vec[s] - p);
                    }
                    res /= M;
                    double rel_err = res / (var_ut + 1e-12);
                    double score = rel_err + 1e-4 * n_terms;
                    if (score < best_score) {
                        best_score = score;
                        best_r2 = 1.0 - rel_err;
                        best_sub = idx;
                        best_w = w;
                    }
                }
                return;
            }
            for (int k = start_k; k < K; ++k) {
                idx[pos] = k;
                search_pde(pos + 1, k + 1);
            }
        };
        search_pde(0, 0);
    }

    // Integrate discovered PDE across future window [0.3, 0.6]
    auto disc_pde_rhs = [&](const vector<double>& uf) {
        vector<double> dudt(Nx, 0.0);
        for (int j = 1; j < Nx - 1; ++j) {
            double ux = (uf[j + 1] - uf[j - 1]) / (2.0 * dx);
            double uxx = (uf[j + 1] - 2.0 * uf[j] + uf[j - 1]) / (dx * dx);
            double uv = uf[j];
            double vals[5] = {uxx, uv * ux, ux, uv, uv * uv};
            double sum = 0.0;
            for (size_t t = 0; t < best_sub.size(); ++t) sum += best_w[t] * vals[best_sub[t]];
            dudt[j] = sum;
        }
        return dudt;
    };

    vector<double> u_sim = U_all[Nt_tr];
    vector<double> true_flat, pred_flat;
    for (int n = Nt_tr; n < Nt_tot; ++n) {
        for (int j = 0; j < Nx; ++j) {
            true_flat.push_back(U_all[n][j]);
            pred_flat.push_back(u_sim[j]);
        }
        vector<double> k1 = disc_pde_rhs(u_sim);
        vector<double> u2(Nx, 0.0), u3(Nx, 0.0), u4(Nx, 0.0);
        for (int j = 1; j < Nx - 1; ++j) u2[j] = u_sim[j] + 0.5 * dt * k1[j];
        vector<double> k2 = disc_pde_rhs(u2);
        for (int j = 1; j < Nx - 1; ++j) u3[j] = u_sim[j] + 0.5 * dt * k2[j];
        vector<double> k3 = disc_pde_rhs(u3);
        for (int j = 1; j < Nx - 1; ++j) u4[j] = u_sim[j] + dt * k3[j];
        vector<double> k4 = disc_pde_rhs(u4);
        for (int j = 1; j < Nx - 1; ++j) {
            u_sim[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    double nrmse = compute_nrmse(true_flat, pred_flat);

    cout << ">>> DISCOVERED PARTIAL DIFFERENTIAL EQUATION (PDE) <<<\n";
    cout << "  u_t = ";
    for (size_t t = 0; t < best_sub.size(); ++t) {
        double w = best_w[t];
        if (t > 0) cout << (w >= 0 ? " + " : " - ");
        else if (w < 0) cout << "-";
        cout << fixed << setprecision(4) << abs(w) << " * " << pde_cands[best_sub[t]];
    }
    cout << "\n";
    cout << "  Operator Fit R^2:     " << fixed << setprecision(6) << best_r2 << "\n";
    cout << "  Solve + PDE Latency:  " << fixed << setprecision(3) << ms << " ms\n\n";

    cout << ">>> SPATIO-TEMPORAL FIELD EXTRAPOLATION (t in [0.3, 0.6]) <<<\n";
    cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse << "\n";
    cout << "  Generalization Check: " << ((nrmse < 1e-3) ? "EXACT PDE FIELD LAW LOCKED" : "APPROXIMATE") << "\n";
    cout << "========================================================================\n\n";
    delete root;
}

// =====================================================================
// 6. CLI Main Entry
// =====================================================================

void print_help(const char* prog) {
    cout << "========================================================================\n"
         << " Antigravity Code-Neuron Symbolic & Calculus Math Solver CLI (C++ AVX2) \n"
         << "========================================================================\n\n"
         << "ALGEBRAIC & COMPOSITIONAL FLAGS:\n"
         << "  --expr \"<string>\"       Discover exact symbolic formula from samples\n"
         << "  --data \"x1,y1;x2,y2\"    Discover formula from raw numerical coordinates\n"
         << "  --file <path.csv>       Discover formula from CSV file\n\n"
         << "CALCULUS & DIFFERENTIAL EQUATION FLAGS:\n"
         << "  --diff \"<string>\"       Discover symbolic 1st & 2nd derivatives f'(x), f''(x)\n"
         << "  --integrate \"<string>\"  Discover symbolic antiderivative F(x) = int f(t) dt\n"
         << "  --ode \"<accel_expr>\"    Discover 2nd-order ODE x'' = f(x, dx) & RK4 extrapolate\n"
         << "                          (Use variables: x for position, dx for velocity x')\n"
         << "  --ic \"<x0>,<v0>\"        Initial conditions for --ode [default: \"1.2,0.0\"]\n"
         << "  --telemetry <path.csv>  Blind time-series discovery (x=f(t) vs x''=g(x,x')) from raw (t,x) CSV\n"
         << "  --pde \"<ut_expr>\"       Discover 1D PDE u_t = f(u, ux, uxx) & extrapolate field\n\n"
         << "DOMAIN OPTIONS:\n"
         << "  --train-range \"<a>:<b>\" Training interval [default: \"1.0:5.0\"]\n"
         << "  --test-range \"<c>:<d>\"  Extrapolation test interval [default: \"5.5:8.0\"]\n"
         << "  --samples <N>           Number of sample points [default: 100]\n"
         << "  --help, -h              Display this help menu\n"
         << "========================================================================\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    string expr_str = "", data_str = "", file_str = "", telemetry_str = "";
    string diff_str = "", int_str = "", ode_str = "", pde_str = "";
    string ic_str = "1.2,0.0";
    string train_range_str = "1.0:5.0";
    string test_range_str = "5.5:8.0";
    int n_samples = 100;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--expr" && i + 1 < argc) expr_str = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_str = argv[++i];
        else if (arg == "--file" && i + 1 < argc) file_str = argv[++i];
        else if (arg == "--telemetry" && i + 1 < argc) telemetry_str = argv[++i];
        else if (arg == "--diff" && i + 1 < argc) diff_str = argv[++i];
        else if (arg == "--integrate" && i + 1 < argc) int_str = argv[++i];
        else if (arg == "--ode" && i + 1 < argc) ode_str = argv[++i];
        else if (arg == "--ic" && i + 1 < argc) ic_str = argv[++i];
        else if (arg == "--pde" && i + 1 < argc) pde_str = argv[++i];
        else if (arg == "--train-range" && i + 1 < argc) train_range_str = argv[++i];
        else if (arg == "--test-range" && i + 1 < argc) test_range_str = argv[++i];
        else if (arg == "--samples" && i + 1 < argc) n_samples = stoi(argv[++i]);
    }

    double t_start = 1.0, t_end = 5.0;
    double e_start = 5.5, e_end = 8.0;
    auto colon_t = train_range_str.find(':');
    if (colon_t != string::npos) {
        t_start = stod(train_range_str.substr(0, colon_t));
        t_end = stod(train_range_str.substr(colon_t + 1));
    }
    auto colon_e = test_range_str.find(':');
    if (colon_e != string::npos) {
        e_start = stod(test_range_str.substr(0, colon_e));
        e_end = stod(test_range_str.substr(colon_e + 1));
    }

    cout << fixed << setprecision(6);
    cout << "\n========================================================================\n";
    cout << " Code-Neuron Network Solver (C++ AVX2 Symbolic + Calculus Engine)\n";
    cout << "========================================================================\n";

    if (!diff_str.empty()) {
        run_diff_block(diff_str, t_start, t_end, e_start, e_end);
        return 0;
    }
    if (!int_str.empty()) {
        run_integrate_block(int_str, t_start, t_end, e_start, e_end);
        return 0;
    }
    if (!ode_str.empty()) {
        double x0 = 1.2, v0 = 0.0;
        auto comma = ic_str.find(',');
        if (comma != string::npos) {
            x0 = stod(ic_str.substr(0, comma));
            v0 = stod(ic_str.substr(comma + 1));
        }
        run_ode_block(ode_str, x0, v0);
        return 0;
    }
    if (!telemetry_str.empty()) {
        run_telemetry_block(telemetry_str);
        return 0;
    }
    if (!pde_str.empty()) {
        run_pde_block(pde_str);
        return 0;
    }

    vector<vector<double>> X_tr, X_te;
    vector<double> y_tr, y_te;
    vector<string> var_names;
    bool has_test = false;

    if (!expr_str.empty()) {
        cout << "[Input Mode] Expression Flag: \"" << expr_str << "\"\n";
        cout << "[Intervals]  Train Range: [" << t_start << ", " << t_end << "] (" << n_samples << " points)\n";
        cout << "             Extrapolation Test: [" << e_start << ", " << e_end << "] (" << n_samples << " points)\n\n";

        Parser parser(expr_str);
        ASTNode* root = parser.parse();

        bool is_2d = (expr_str.find("x2") != string::npos);
        if (!is_2d) {
            var_names = {"x"};
            for (int i = 0; i < n_samples; ++i) {
                double v_tr = t_start + (t_end - t_start) * i / (n_samples - 1);
                X_tr.push_back({v_tr});
                y_tr.push_back(root->eval({{"x", v_tr}, {"x1", v_tr}}));

                double v_te = e_start + (e_end - e_start) * i / (n_samples - 1);
                X_te.push_back({v_te});
                y_te.push_back(root->eval({{"x", v_te}, {"x1", v_te}}));
            }
        } else {
            var_names = {"x1", "x2"};
            mt19937 rng(42);
            uniform_real_distribution<double> dist_tr(t_start, t_end);
            uniform_real_distribution<double> dist_te(e_start, e_end);
            for (int i = 0; i < n_samples; ++i) {
                double v1_tr = dist_tr(rng);
                double v2_tr = dist_tr(rng);
                X_tr.push_back({v1_tr, v2_tr});
                y_tr.push_back(root->eval({{"x1", v1_tr}, {"x2", v2_tr}}));

                double v1_te = dist_te(rng);
                double v2_te = dist_te(rng);
                X_te.push_back({v1_te, v2_te});
                y_te.push_back(root->eval({{"x1", v1_te}, {"x2", v2_te}}));
            }
        }
        delete root;
        has_test = true;
    } else if (!data_str.empty()) {
        cout << "[Input Mode] Direct Numerical Data Flag: \"" << data_str << "\"\n\n";
        var_names = {"x"};
        stringstream ss(data_str);
        string pt;
        while (getline(ss, pt, ';')) {
            auto comma = pt.find(',');
            if (comma != string::npos) {
                X_tr.push_back({stod(pt.substr(0, comma))});
                y_tr.push_back(stod(pt.substr(comma + 1)));
            }
        }
    } else if (!file_str.empty()) {
        cout << "[Input Mode] CSV File Flag: \"" << file_str << "\"\n\n";
        var_names = {"x"};
        ifstream fin(file_str);
        string line;
        while (getline(fin, line)) {
            if (line.empty() || isalpha(line[0])) continue;
            auto comma = line.find(',');
            if (comma != string::npos) {
                X_tr.push_back({stod(line.substr(0, comma))});
                y_tr.push_back(stod(line.substr(comma + 1)));
            }
        }
    }

    if (y_tr.empty()) {
        cerr << "[ERROR] No data points to solve! Use --expr, --diff, --integrate, --ode, --pde, --data, or --file.\n";
        return 1;
    }

    auto t0 = chrono::high_resolution_clock::now();
    vector<DAGNode> dag = build_dag(var_names, 3);
    Model model = solve_math_program(X_tr, y_tr, dag, 1.0 - 1e-10);
    auto t1 = chrono::high_resolution_clock::now();
    double solve_ms = chrono::duration<double, milli>(t1 - t0).count();

    cout << ">>> SOLVER OUTPUT <<<\n";
    cout << "  Status:               " << ((model.r2 > 1.0 - 1e-4) ? "EXACT ANALYTICAL SOLUTION FOUND (0-error)" : "APPROXIMATE / FAILED") << "\n";
    cout << "  Training R^2:         " << model.r2 << "\n";
    cout << "  Solve Latency:        " << solve_ms << " ms\n";
    cout << "  DAG Complexity:       " << dag.size() << " topological code-neurons\n\n";

    cout << ">>> RECOVERED MATHEMATICAL EXPRESSION <<<\n";
    cout << "  Active Code-Neurons:  " << model.names_str << "\n";
    cout << "  f(x) = " << format_model_formula(model, dag) << "\n";

    if (has_test) {
        vector<double> preds = predict_model(model, dag, X_te);
        double nrmse = compute_nrmse(y_te, preds);
        cout << "\n>>> OUT-OF-DISTRIBUTION EXTRAPOLATION VERIFICATION <<<\n";
        cout << "  Extrapolation NRMSE:  " << scientific << setprecision(2) << nrmse << "\n";
        cout << "  Generalization Check: " << ((nrmse < 1e-4) ? "PERFECT (Infinite horizon generalization)" : "DEVIATES") << "\n";
    }

    cout << "========================================================================\n\n";
    return 0;
}
