#include <iostream>
#include <vector>
#include <string>

#include <H5Cpp.h>

#include <vexcl/vexcl.hpp>

#include <boost/program_options.hpp>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

namespace odeint = boost::numeric::odeint;

//---------------------------------------------------------------------------
// Lorenz model.
//---------------------------------------------------------------------------
template <class State, class Parameter>
struct lorenz_system {
    const Parameter &sigma;
    const Parameter &R;

    lorenz_system(const Parameter &sigma, const Parameter &R)
        : sigma(sigma), R(R) {}

    void operator()(const State &x, State &dxdt, double /*t*/) const {
        dxdt[0] = -sigma * (x[0] - x[1]);
        dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
        dxdt[2] = -8.0 / 3.0 * x[2] + x[0] * x[1];
    }
};

//---------------------------------------------------------------------------
namespace config {
    std::string conf = "lorenz.cfg";
    std::string out  = "lorenz.h5";

    double x0 = 10;
    double y0 = 10;
    double z0 = 10;

    double dt   = 0.01;
    double tmax = 1000.0;
    int    kmax = 16;

    double sigma_min   = 1;
    double sigma_max   = 40;
    int    sigma_steps = 1000;

    double R_min   = 50;
    double R_max   = 110;
    int    R_steps = 1000;

    void read(int argc, char *argv[]);
} // namespace config

//---------------------------------------------------------------------------
vex::generator::Kernel<10> make_kernel(const vex::Context &ctx);
void save_kneading(vex::vector<cl_ulong> &k);

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    try {
        config::read(argc, argv);
        const int n = config::sigma_steps * config::R_steps;

        // Initialize VexCL context
        vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
        std::cout << ctx << std::endl;

        // Fill parameters.
        double hs = (config::sigma_max - config::sigma_min ) / (config::sigma_steps - 1);
        double hr = (config::R_max - config::R_min) / (config::R_steps - 1);

        vex::vector<double> R(ctx, n);
        vex::vector<double> sigma(ctx, n);

        R = config::R_min + hr * (vex::element_index() % config::R_steps);
        sigma = config::sigma_min + hs * (vex::element_index() / config::R_steps);

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
        auto sweep = make_kernel(ctx);

        vex::Reductor<int, vex::MIN> min(ctx);
        vex::Reductor<int, vex::MAX> max(ctx);

        // Integrate over time.
        int    iter = 0;
        int    kmin = 0;
        double time = 0;
        for(; time <= config::tmax; time += config::dt, ++iter) {
            sweep(x, y, z, dx, dy, dz, num, seq, sigma, R);

            int last_kmin = kmin;
            kmin = min(num);

            if (last_kmin != kmin)
                std::cout
                    << "time = " << std::scientific << time << ", "
                    << "kmin = " << kmin << std::endl;

            if (iter % 10 == 0 && kmin >= config::kmax) break;
        }

        std::cout << "Kneading points: " << min(num) << std::endl;
        std::cout << "Time to reach:   " << time << std::endl;

        save_kneading(seq);
    } catch (const vex::backend::error &e) {
        std::cerr << "VexCL error: " << e << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

namespace config {

//---------------------------------------------------------------------------
template <typename T>
std::string to_string(const T &val) {
    std::ostringstream s;
    s << std::setprecision(3) << val;
    return s.str();
}

//---------------------------------------------------------------------------
void read(int argc, char *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Options");

#define OPTION(name, descr)                                                    \
    (#name,                                                                    \
     po::value<decltype(name)>(&name)->default_value(name, to_string(name)),   \
     descr                                                                     \
    )

    desc.add_options()
        ("help,h", "Show help")
        OPTION(conf,        "Configuration file")
        OPTION(out,         "Output file")
        OPTION(x0,          "Initial X xoordinate")
        OPTION(y0,          "Initial Y xoordinate")
        OPTION(z0,          "Initial Z xoordinate")
        OPTION(dt,          "Time step")
        OPTION(tmax,        "Time limit")
        OPTION(kmax,        "Length of kneading sequence")
        OPTION(sigma_min,   "Minimum sigma value")
        OPTION(sigma_max,   "Maximum sigma value")
        OPTION(sigma_steps, "Number of sigma values")
        OPTION(R_min,       "Minimum R value")
        OPTION(R_max,       "Maximum R value")
        OPTION(R_steps,     "Number of R values")
        ;

#undef OPTION

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }

    std::ifstream cfg(conf);
    if (cfg) {
        po::store(po::parse_config_file(cfg, desc), vm);
        po::notify(vm);
    }

    if (config::kmax > 64)
        throw std::logic_error("kmax is too large (maximum suported is 64)");
}

} // namespace config

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

    sym_vector sigma(sym_vector::VectorParameter, sym_vector::Const);
    sym_vector R(sym_vector::VectorParameter, sym_vector::Const);

    vex::symbolic<cl_short> num(vex::symbolic<cl_short>::VectorParameter);
    vex::symbolic<cl_ulong> seq(vex::symbolic<cl_ulong>::VectorParameter);

    body << "if (" << num << " >= " << config::kmax << ") continue;\n";

    // Stepper type
    odeint::runge_kutta_dopri5<
        sym_state, double, sym_state, double,
        odeint::range_algebra, odeint::default_operations
        > stepper;

    // Record single RK4 step
    lorenz_system<sym_state, sym_vector> sys{sigma, R};
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
    return vex::generator::build_kernel(ctx, "sm_sweep", body.str(),
            x[0], x[1], x[2], dx[0], dx[1], dx[2], num, seq, sigma, R);
}

//---------------------------------------------------------------------------
void save_kneading(vex::vector<cl_ulong> &k) {
    using namespace H5;

    H5File hdf(config::out, H5F_ACC_TRUNC);

    hsize_t dim[] = {
        static_cast<hsize_t>(config::sigma_steps),
        static_cast<hsize_t>(config::R_steps)
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

#undef CREATE_ATTRIBUTE

#define CREATE_ATTRIBUTE(name, val, type) \
    ds.createAttribute(#name, type, adsp).write(type, &config::val)

    CREATE_ATTRIBUTE(xmin, R_min,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(xmax, R_max,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(xnum, R_steps, PredType::NATIVE_INT32);

    CREATE_ATTRIBUTE(ymin, sigma_min,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(ymax, sigma_max,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(ynum, sigma_steps, PredType::NATIVE_INT32);

#undef CREATE_ATTRIBUTE
}

