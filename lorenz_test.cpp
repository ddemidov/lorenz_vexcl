#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

#include "config.hpp"
#include "lorenz_system.hpp"

namespace odeint = boost::numeric::odeint;
typedef boost::array<double, 3> state_type;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    config::read(argc, argv);

    const double alpha  = config::alpha_min;
    const double lambda = config::lambda_min;

    // Stepper type
    odeint::runge_kutta4_classic<
        state_type, double, state_type, double,
        odeint::range_algebra, odeint::default_operations
        > stepper;

    lorenz_system<state_type, double> sys(alpha, lambda, config::B);

    state_type x = {{config::x0, config::y0, config::z0}};

    std::ofstream f("trajectory.dat");
    for(double time = 0; time < config::tmax; time += config::dt) {
        stepper.do_step(std::ref(sys), x, time, config::dt);
        f << x[0] << " " << x[1] << " " << x[2] << "\n";
    }
}
