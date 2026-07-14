#include <cmath>
#include <string>
#include <fstream>
#include <print>
#include <format>
#include <chrono>

#include <complex>
#include <Array.hpp>
#include "../numerical_methods.hpp"

//******************************************************************************
// Examples of initial conditions one can try
//******************************************************************************

double KP_ic_1(double x, double y, double t)
{

    /*

    double xi = -20.0;
    double xf = 20.0;
    double yi = -40.0;
    double yf = 40.0;
    double ti = -1;
    double tf = 1;

    int64_t nx = 512;
    int64_t ny = 2;
    int64_t nt = 20001;

    */

    return 12 * ((3 + 4 * std::cosh(2.0 * x - 8.0 * t) + std::cosh(4.0 * x - 64.0 * t)))
              / std::pow((3 * std::cosh(x - 28.0 * t) + std::cosh(3.0 * x - 36.0 * t)), 2.0);
}

double KP_ic_2(double x, double y, double t)
{

    /*

    double xi = -10;
    double xf = 50.0;
    double yi = -15.0;
    double yf = 15.0;
    double ti = 0.0;
    double tf = 4.0;

    int64_t nx = 512;
    int64_t ny = 256;
    int64_t nt = 40001;

    */

    double x0 = 15.0;
    double y0 = 0.0;
    double kI = std::sqrt(6.0) / 4.0;

    double n = -  4.0 * (x - x0) * (x - x0)
               + 16.0 * kI * kI * (y - y0) * (y - y0) + 1.0 / (kI * kI);

    double d =    4.0 * (x - x0) * (x - x0)
               + 16.0 * kI * kI * (y - y0) * (y - y0) + 1.0 / (kI * kI);

    d *= d;

    return 16.0 * n / d;
}

double KP_ic_3(double x, double y, double t)
{

    /*

    double xi = 0.0;
    double xf = 90.0;
    double yi = -30.0;
    double yf = 30.0;
    double ti = 0.0;
    double tf = 10.0;

    int64_t nx = 1024;
    int64_t ny = 512;
    int64_t nt = 100001;

    */

    double x01 = 15.0;
    double y01 = 0.0;
    double k1I = std::sqrt(6.0) / 4.0;

    double n1 = -  4.0 * (x - x01) * (x - x01)
               + 16.0 * k1I * k1I * (y - y01) * (y - y01) + 1.0 / (k1I * k1I);

    double d1 =    4.0 * (x - x01) * (x - x01)
               + 16.0 * k1I * k1I * (y - y01) * (y - y01) + 1.0 / (k1I * k1I);

    d1 *= d1;

    double x02 = 31.0;
    double y02 = 0.0;
    double k2I = std::sqrt(6.0) / 8.0;

    double n2 = - 4.0 * (x - x02) * (x - x02)
                + 16.0 * k2I * k2I * (y - y02) * (y - y02) + 1.0 / (k2I * k2I);

    double d2 =   4.0 * (x - x02) * (x - x02)
                + 16.0 * k2I * k2I * (y - y02) * (y - y02) + 1.0 / (k2I * k2I);

    d2 *= d2;

    return 16.0 * n1 / d1 + 16.0 * n2 / d2 ;
}

//******************************************************************************
// Solver functions
//******************************************************************************

constexpr double sigma_sq = -1.0;

auto RK4_f (
    const double &t,
    const Array<std::complex<double>, 2> &u,
    const Array<double, 2> &hy_hx
) -> Array<std::complex<double>, 2>
{
    int64_t ny = u.extents(0);
    int64_t nx = u.extents(1);

    int64_t log2_ny = std::countr_zero(uint64_t(ny));
    int64_t log2_nx = std::countr_zero(uint64_t(nx));

    double hy = hy_hx[0, 0];
    double hx = hy_hx[0, 1];

    auto u_t = u;

    for(int64_t i = ny / 2 - (ny / 7); i < ny / 2 + (ny / 7); i++)
    for(int64_t j = nx / 2 - (nx / 7); j < nx / 2 + (nx / 7); j++)
        u_t[i, j] = std::complex<double>{0.0, 0.0};

    dft_2d(u_t, true);

    u_t *= u_t;

    dft_2d(u_t, false);

    double delta_kx = 2.0 * pi / (nx * hx);
    double delta_ky = 2.0 * pi / (ny * hy);
    double kx = 0.0;
    double ky = 0.0;

    for ( int64_t i = 0; i < ny; i++ )
    {
        for ( int64_t j = 0; j < nx; j++ )
        {
            u_t[i, j] *= -3.0 * kx * std::complex<double>{0.0, 1.0};
            kx += delta_kx;
            if(j == nx / 2)
                kx -= 2 * pi / (hx);
        }
        kx = 0.0;
        ky += delta_ky;
        if(i == ny / 2)
            ky -= 2 * pi / (hy);
    }

    return u_t;
}

Array<double, 2> solve_KP (
    const Array<double, 2> &ic,
    double ti, double tf, int64_t nt,
    double xi, double xf, int64_t log2_nx,
    double yi, double yf, int64_t log2_ny
)
{
    auto sol = ic;

    int64_t ny = ic.extents(0);
    int64_t nx = ic.extents(1);

    double hx = (xf - xi) / (nx - 1);
    double hy = (yf - yi) / (ny - 1);
    double ht = (tf - ti) / (nt - 1);

    Array<double, 2> hy_hx(1, 2);
    hy_hx[0, 0] = hy;
    hy_hx[0, 1] = hx;

    Array<std::complex<double>, 2> ic_c(ny, nx);

    for(int64_t i = 0; i < ny; i++)
    for(int64_t j = 0; j < nx; j++)
        ic_c[i, j] = {sol[i, j], 0.0};

    ode_ivp_RK4_state<double, Array<std::complex<double>, 2>, Array<double, 2>> RK4_state(ti, ic_c, hy_hx);

    dft_2d(RK4_state.y, false);

    for ( int64_t l = 0; l < nt - 1; l++ )
    {
        double delta_kx = 2.0 * pi / (nx * hx);
        double delta_ky = 2.0 * pi / (ny * hy);
        double kx = 0.0;
        double ky = 0.0;

        for ( int64_t i = 0; i < ny; i++ )
        {
            kx = 0.0;
            for ( int64_t j = 0; j < nx; j++ )
            {
                std::complex<double> z;
                if ( kx != 0.0 )
                    z = std::complex<double>{0.0, 0.5 * (kx * kx * kx - 3 * sigma_sq * (ky * ky / kx)) * ht};
                else
                    z = std::complex<double>{0.0, 0.5 * kx * kx * kx * ht};
                RK4_state.y[i, j] *= exp(z);

                kx += delta_kx;
                if ( j == nx / 2 )
                    kx -= 2 * pi / hx;
            }
            ky += delta_ky;
            if(i == ny / 2)
                ky -= 2 * pi / hy;
        }

        ode_ivp_RK4(RK4_f, RK4_state, RK4_state.t + ht, 1);

        for ( int64_t i = 0; i < ny; i++ )
        {
            kx = 0.0;
            for ( int64_t j = 0; j < nx; j++ )
            {
                std::complex<double> z;
                if(kx != 0.0)
                    z = std::complex<double>{0.0, 0.5 * (kx * kx * kx - 3 * sigma_sq * (ky * ky / kx)) * ht};
                else
                    z = std::complex<double>{0.0, 0.5 * kx * kx * kx * ht};
                RK4_state.y[i, j] *= exp(z);

                kx += delta_kx;
                if(j == nx / 2)
                    kx -= 2 * pi / (hx);
            }
            ky += delta_ky;
            if(i == ny / 2)
                ky -= 2 * pi / (hy);
        }

        RK4_state.t += ht;
    }

    dft_2d(RK4_state.y, true);

    for ( int64_t i = 0; i < sol.size(); i++ )
        sol.p_elements()[i] = RK4_state.y.p_elements()[i].real();

    return sol;
}

Array<Array<double, 2>, 2>
solve_KP_richardson (
    const Array<double, 2> &ic,
    double ti, double tf, int64_t nt,
    double xi, double xf, int64_t log2_nx,
    double yi, double yf, int64_t log2_ny,
    int64_t n_sols
)
{
    Array<int64_t, 1> nts(n_sols);
    for(int64_t i = 0; i < nts.size(); i++)
        nts[i] = (nt - 1) * (1 << (nts.size() - i - 1)) + 1;

    Array<Array<double, 2>, 2> sols(n_sols, n_sols);

    for ( int64_t j = 0; j < n_sols; j++ )
    {
        sols[0, j] = ic;
        sols[0, j] = solve_KP (
            sols[0, j],
            ti, tf, nts[j],
            xi, xf, log2_nx,
            yi, yf, log2_ny
        );
    }

    for ( int64_t i = 1; i < n_sols; i++ )
        for ( int64_t j = 0; j < n_sols - i; j++ )
            sols[i, j] = (sols[i - 1, j] * std::pow(2, 2 * i) - sols[i - 1, j + 1])
                       / (std::pow(2, 2 * i) - 1.0);

    return sols;
}

template<typename T>
void store_snapshot(const Array<T, 2> &u, const std::string &filename)
{
    std::ofstream file;
    file.open(filename);

    for ( int64_t i = 0; i < u.extents()[0]; i++ )
    {
        for ( int64_t j = 0; j < u.extents()[1]; j++ )
        {
            file << u[i, j];
            if ( j < u.extents()[1] - 1 )
                file << ',';
        }
        file << '\n';
    }

    file.close();
}

//******************************************************************************
// Executable
//******************************************************************************

int main()
{
    // Input starts here

    double xi = 0.0;
    double xf = 90.0;
    double yi = -30.0;
    double yf = 30.0;
    double ti = 0.0;
    double tf = 10.0;

    int64_t nx = 1024;
    int64_t ny = 512;
    int64_t nt = 100001;

    double (*initial_conditions)(double, double, double) = KP_ic_3;

    int64_t n_sols = 1;

    int64_t snapshot_step_interval = 200;

    std::string snapshot_csv_directory = "/home/dimitris/Documents/KP_Snapshots/";

    // Input ends here

    double hx = (xf - xi) / (nx - 1);
    double hy = (yf - yi) / (ny - 1);
    double ht = (tf - ti) / (nt - 1);

    int64_t n_steps = nt - 1;

    Array<double, 2> ic(ny, nx);
    Array<double, 1> x(nx);
    Array<double, 1> y(ny);

    int64_t log2_nx = std::countr_zero(uint64_t(nx));
    int64_t log2_ny = std::countr_zero(uint64_t(ny));

    for ( int64_t i = 0; i < ny; i++ )
    for ( int64_t j = 0; j < nx; j++ )
    {
        x[j] = xi + j * hx;
        y[i] = yi + i * hy;
        ic[i, j] = initial_conditions(x[j], y[i], ti);
    }

    Array<Array<double, 2>, 2> sols(n_sols, n_sols);
    sols[n_sols - 1, 0] = ic;

    double  time_point = ti;
    int64_t time_point_idx = 0;

    auto solver_time_start = std::chrono::steady_clock::now();

    while ( true )
    {
        auto time_now = std::chrono::steady_clock::now();
        double solver_runtime_s = std::chrono::duration<double>(time_now - solver_time_start).count();

        std::print (
            "Time point {:>8} of {:>10} | t = {:>14.5e} | sol norm: {:.16e} | solver_time {:10.3f} s\n",
            time_point_idx, nt - 1, time_point, norm_L2(sols[n_sols - 1, 0]), solver_runtime_s
        );

        store_snapshot (
            sols[n_sols - 1, 0], snapshot_csv_directory + "KP_sol_snap_" + std::to_string(time_point_idx) + ".txt"
        );

        if ( time_point_idx == nt - 1 ) break;

        int64_t n_steps_left = nt - time_point_idx - 1;

        sols = solve_KP_richardson (
            sols[n_sols - 1, 0],
            time_point,
            time_point + std::min(snapshot_step_interval, n_steps_left) * ht,
            std::min(snapshot_step_interval + 1, n_steps_left + 1),
            xi, xf, log2_nx,
            yi, yf, log2_ny,
            n_sols
        );

        time_point_idx += snapshot_step_interval;

        if (time_point_idx > nt - 1) time_point_idx = nt - 1;

        time_point = ti + time_point_idx * ht;
    }

    // Display convergence results

    std::print("\n");

    std::print (
        "{:>12}{:>7}{:>12}{:>7}{:>12}{:>7}\n",
        "ti:", ti, "tf:", tf, "nt:", nt
    );

    std::print (
        "{:>12}{:>7}{:>12}{:>7}{:>12}{:>7}\n",
        "xi:", xi, "xf:", xf, "nx:", nx
    );

    std::print (
        "{:>12}{:>7}{:>12}{:>7}{:>12}{:>7}\n",
        "yi:", yi, "yf:", yf, "ny:", ny
    );

    std::print (
        "{:>12}{:>7}\n\n\n",
        "n_sols:", n_sols
    );

    for ( int64_t j = 0; j < n_sols; j++ )
        std::print("{:>16}", (nt - 1) * (1 << (n_sols - j - 1)));
    std::print("\n\n");

    for ( int64_t i = 0; i < n_sols - 1; i++ )
    {
        for ( int64_t j = 0; j < n_sols - i - 1; j++ )
            std::print("{:>16.7e}",
                norm_L2(sols[i, j] - sols[i, j + 1]) /
                norm_L2(sols[i, j])
            );
        std::print("{:>16}\n", "n/a");
    }
    std::print("{:>16}\n\n\n", "n/a");

    for ( int64_t i = 0; i < n_sols - 1; i++ )
    {
        for ( int64_t j = 0; j < n_sols - i - 2; j++ )
            std::print("{:>16.7f}",
                norm_L2(sols[i, j + 1] - sols[i, j + 2]) /
                norm_L2(sols[i, j]     - sols[i, j + 1])
            );
        std::print("{:>16}{:>16}\n", "n/a", "n/a");
    }
    std::print("{:>16}\n\n\n", "n/a");

    return 0;
}

