// =========================================================================
// Blind Observational Telemetry Benchmark:
// Non-Elementary Dynamical Systems vs. Explicit Time-Domain Regression
// =========================================================================
// Generates raw 2-column (t, x) observational telemetry CSV files for five
// physical systems whose explicit time solutions x(t) cannot be expressed
// in closed-form elementary functions (requiring Jacobi elliptic functions,
// incomplete elliptic integrals, or non-integrable limit cycles), plus one
// closed-form elementary control trajectory.
// =========================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <functional>
#include <filesystem>

using namespace std;

struct TelemetryCase {
    string filename;
    string name;
    string analytic_obstacle;
    string true_law_desc;
    double x0;
    double v0;
    function<double(double, double)> accel;
    bool is_explicit_control;
    function<double(double)> explicit_fn;
};

int main() {
    filesystem::create_directories("benchmarks/telemetry");

    vector<TelemetryCase> cases = {
        {
            "benchmarks/telemetry/01_nonlinear_pendulum.csv",
            "Large-Amplitude Damped Pendulum",
            "Requires Incomplete Elliptic Integrals F(phi, k)",
            "x''(t) = -0.2000 * dx - 9.8100 * sin(x)",
            1.4, 0.0,
            [](double x, double dx) { return -0.2 * dx - 9.81 * sin(x); },
            false, nullptr
        },
        {
            "benchmarks/telemetry/02_duffing_bistable.csv",
            "Duffing Bistable Buckling Beam Oscillator",
            "Requires Jacobi Elliptic Functions cn(t; k), dn(t; k)",
            "x''(t) = 1.0000 * x - 0.1500 * dx - 1.0000 * x^3",
            0.5, 0.8,
            [](double x, double dx) { return 1.0 * x - 0.15 * dx - 1.0 * x * x * x; },
            false, nullptr
        },
        {
            "benchmarks/telemetry/03_vanderpol_pacemaker.csv",
            "Van der Pol Self-Sustained Limit-Cycle Oscillator",
            "Non-Integrable Limit Cycle (No Closed-Form in Special Functions)",
            "x''(t) = -1.0000 * x + 1.0000 * dx - 1.0000 * x^2*dx",
            0.5, 0.0,
            [](double x, double dx) { return 1.0 * dx - 1.0 * x * x * dx - 1.0 * x; },
            false, nullptr
        },
        {
            "benchmarks/telemetry/04_ballistic_quadratic_drag.csv",
            "Subsonic Terminal Ballistics with Quadratic Drag",
            "Piecewise Non-Smooth Riccati Differential Equation",
            "x''(t) = -9.8100 * const + 0.2500 * dx^2",
            100.0, 0.0,
            [](double x, double dx) { (void)x; return -9.81 - 0.25 * dx * std::abs(dx); },
            false, nullptr
        },
        {
            "benchmarks/telemetry/05_anharmonic_lattice.csv",
            "Anharmonic Asymmetric Crystal Lattice Potential",
            "Requires Weierstrass Elliptic Function P(t; g2, g3)",
            "x''(t) = -4.0000 * x - 0.1000 * dx + 1.5000 * x^2",
            0.8, 0.0,
            [](double x, double dx) { return -4.0 * x - 0.10 * dx + 1.5 * x * x; },
            false, nullptr
        },
        {
            "benchmarks/telemetry/06_elementary_control.csv",
            "Elementary Closed-Form Control Signal",
            "Exact Elementary Composition: x(t) = 2.5*exp(-t) + sin(t)",
            "x(t) = 2.5000 * exp(-t) + 1.0000 * sin(t)",
            0.0, 0.0,
            nullptr,
            true, [](double t) { return 2.5 * exp(-t) + sin(t); }
        }
    };

    int N_tot = 1200;
    double dt = 0.01;

    cout << "Generating 6 blind telemetry CSV files (t, x only; 1,200 points each)...\n";
    for (const auto& c : cases) {
        ofstream fout(c.filename);
        fout << "# Blind Observational Telemetry: " << c.name << "\n";
        fout << "# Obstacle: " << c.analytic_obstacle << "\n";
        fout << "t,x\n";
        fout << setprecision(12) << fixed;

        if (c.is_explicit_control) {
            for (int i = 0; i < N_tot; ++i) {
                double t = 0.5 + i * dt;
                fout << t << "," << c.explicit_fn(t) << "\n";
            }
        } else {
            double x = c.x0, v = c.v0;
            for (int i = 0; i < N_tot; ++i) {
                double t = i * dt;
                fout << t << "," << x << "\n";
                double k1_x = v;
                double k1_v = c.accel(x, v);
                double k2_x = v + 0.5 * dt * k1_v;
                double k2_v = c.accel(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v);
                double k3_x = v + 0.5 * dt * k2_v;
                double k3_v = c.accel(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v);
                double k4_x = v + dt * k3_v;
                double k4_v = c.accel(x + dt * k3_x, v + dt * k3_v);

                x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
                v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
            }
        }
        fout.close();
        cout << "  -> Wrote " << c.filename << " (" << c.name << ")\n";
    }
    return 0;
}
