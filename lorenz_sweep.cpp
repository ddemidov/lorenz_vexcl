#include <iostream>
#include <vector>
#include <string>

#include <vexcl/vexcl.hpp>

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

#include "config.hpp"
#include "lorenz_system.hpp"

namespace odeint = boost::numeric::odeint;

//---------------------------------------------------------------------------
// Generate a monolythic kernel that does a single Runge-Kutta step.
//---------------------------------------------------------------------------
vex::generator::Kernel<5> make_kernel(const vex::Context &ctx) {
    typedef vex::symbolic<double>       sym_vector;
    typedef boost::array<sym_vector, 3> sym_state;

    // Kernel body will be recorded here:
    std::ostringstream body;
    vex::generator::set_recorder(body);

    // Symbolic variables. These will be fed to odeint algorithm.
    sym_state sym_S = {{
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter)
    }};

    sym_vector sym_alpha (sym_vector::VectorParameter, sym_vector::Const);
    sym_vector sym_lambda(sym_vector::VectorParameter, sym_vector::Const);

    // Stepper type
    odeint::runge_kutta4_classic<
        sym_state, double, sym_state, double,
        odeint::range_algebra, odeint::default_operations
        > stepper;

    // Record single RK4 step
    lorenz_system<sym_state, sym_vector> sys(sym_alpha, sym_lambda, config::B);
    stepper.do_step(std::ref(sys), sym_S, 0, config::dt);

    // Generate the kernel from the recorded sequence
    return vex::generator::build_kernel(ctx, "lorenz_sweep",
            body.str(), sym_S[0], sym_S[1], sym_S[2], sym_alpha, sym_lambda);

}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    config::read(argc, argv);
    const int n = config::alpha_steps * config::lambda_steps;

    // Initialize VexCL context
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;

    // Fill parameters.
    double ha = (config::alpha_max  - config::alpha_min ) / (config::alpha_steps  - 1);
    double hl = (config::lambda_max - config::lambda_min) / (config::lambda_steps - 1);

    vex::vector<double> alpha (ctx, n);
    vex::vector<double> lambda(ctx, n);

    alpha  = config::alpha_min  + ha * (vex::element_index() % config::alpha_steps);
    lambda = config::lambda_min + hl * (vex::element_index() / config::alpha_steps);

    // Set initial position.
    vex::vector<double> x(ctx, n);
    vex::vector<double> y(ctx, n);
    vex::vector<double> z(ctx, n);

    x = config::x0;
    y = config::y0;
    z = config::z0;

    // Create kernel.
    auto lorenz = make_kernel(ctx);

    // Integrate over time.
    for(double time = 0; time < config::tmax; time += config::dt)
        lorenz(x, y, z, alpha, lambda);
}
