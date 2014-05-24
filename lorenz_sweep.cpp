#include <iostream>
#include <vector>
#include <string>

#include <H5Cpp.h>

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
vex::generator::Kernel<10> make_kernel(const vex::Context &ctx) {
    typedef vex::symbolic<double>       sym_vector;
    typedef boost::array<sym_vector, 3> sym_state;

    // Kernel body will be recorded here:
    std::ostringstream body;
    vex::generator::set_recorder(body);

    // Symbolic variables. These will be fed to odeint algorithm.
    sym_state x = {{
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter)
    }};

    sym_state dx = {{
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter),
        sym_vector(sym_vector::VectorParameter)
    }};

    sym_state x_new;
    sym_state dx_new;

    sym_vector alpha (sym_vector::VectorParameter, sym_vector::Const);
    sym_vector lambda(sym_vector::VectorParameter, sym_vector::Const);

    vex::symbolic<cl_short> num(vex::symbolic<cl_short>::VectorParameter);
    vex::symbolic<cl_ulong> seq(vex::symbolic<cl_ulong>::VectorParameter);

    body << "if (" << num << " >= " << config::kmax << ") continue;\n";

    // Stepper type
    odeint::runge_kutta_dopri5<
        sym_state, double, sym_state, double,
        odeint::range_algebra, odeint::default_operations
        > stepper;

    // Record single RK4 step
    lorenz_system<sym_state, sym_vector> sys(alpha, lambda, config::B);
    stepper.do_step(std::ref(sys), x, dx, 0, x_new, dx_new, config::dt);

    // Compute kneading invariant
    /*
     * if (dx/dt == 0) and (d2x/dt2 != 0), then {
     *     update q
     *     update k
     * }
     * Note that d2x/dt2 = dy/dt.
     */
    body <<
        "if (" << dx[0] << " * " << dx_new[0] << " < 0) {\n"
        "  if ((" << dx_new[1] << " < 0) && (" << x_new[0] << " > 0)) {\n"
        "    " << seq << " |= (1 << " << num << ");\n"
        "    " << num << " += 1;\n"
        "  } else\n"
        "  if ((" << dx_new[1] << " > 0) && (" << x_new[0] << " < 0)) {\n"
        "    " << num << " += 1;\n"
        "  }\n"
        "}\n";

    // Save new values of coordinates and derivatives.
    x[0] = x_new[0];
    x[1] = x_new[1];
    x[2] = x_new[2];

    dx[0] = dx_new[0];
    dx[1] = dx_new[1];
    dx[2] = dx_new[2];

    // Generate the kernel from the recorded sequence
    return vex::generator::build_kernel(ctx, "lorenz_sweep", body.str(),
            x[0], x[1], x[2], dx[0], dx[1], dx[2], num, seq, alpha, lambda);
}

//---------------------------------------------------------------------------
void save_kneading(vex::vector<cl_ulong> &k);

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    try {
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

        vex::vector<double> dx(ctx, n);
        vex::vector<double> dy(ctx, n);
        vex::vector<double> dz(ctx, n);

        x = config::x0;
        y = config::y0;
        z = config::z0;

        dx = 0;
        dy = 0;
        dz = 0;

        // Kneading sequence
        vex::vector<cl_short> num(ctx, n);
        vex::vector<cl_ulong> seq(ctx, n);
        num = 0;
        seq = 0;

        // Create kernel.
        auto lorenz = make_kernel(ctx);

        vex::Reductor<int, vex::MIN> min(ctx);
        vex::Reductor<int, vex::MAX> max(ctx);

        // Integrate over time.
        size_t iter = 0;
        for(double time = 0; time < config::tmax; time += config::dt, ++iter) {
            lorenz(x, y, z, dx, dy, dz, num, seq, alpha, lambda);
            if (iter % 10 == 0 && min(num) >= config::kmax) break;
        }

        std::cout << "Number of kneading points: " << min(num) << " - " << max(num) << std::endl;

        save_kneading(seq);
    } catch (const vex::backend::error &e) {
        std::cerr << "VexCL error: " << e << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

//---------------------------------------------------------------------------
void save_kneading(vex::vector<cl_ulong> &k) {
    using namespace H5;

    H5File hdf(config::out, H5F_ACC_TRUNC);

    hsize_t dim[] = {
        static_cast<hsize_t>(config::lambda_steps),
        static_cast<hsize_t>(config::alpha_steps)
    };

    hsize_t one[] = { 1 };
    DataSpace adsp(1, one);

    std::vector<cl_ulong> k_host(k.size());
    vex::copy(k, k_host);

    DataSet ds = hdf.createDataSet("/K", PredType::NATIVE_UINT64, DataSpace(2, dim));
    ds.write(k_host.data(), PredType::NATIVE_UINT64);

#define CREATE_ATTRIBUTE(name, type) \
    ds.createAttribute(#name, type, adsp).write(type, &config::name)

    CREATE_ATTRIBUTE(x0,           PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(y0,           PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(z0,           PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(dt,           PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(kmax,         PredType::NATIVE_INT32);
    CREATE_ATTRIBUTE(alpha_min,    PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(alpha_max,    PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(alpha_steps,  PredType::NATIVE_INT32);
    CREATE_ATTRIBUTE(lambda_min,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(lambda_max,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(lambda_steps, PredType::NATIVE_INT32);
    CREATE_ATTRIBUTE(B,            PredType::NATIVE_DOUBLE);

#undef CREATE_ATTRIBUTE
}

