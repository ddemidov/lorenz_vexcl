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
// The Homoclinic Garden model
//---------------------------------------------------------------------------
template <class State, class Parameter>
struct hg_system {
    const double mu1 = 0;
    const Parameter &mu2;
    const Parameter &mu3;

    hg_system(double mu1, const Parameter &mu2, const Parameter &mu3)
        : mu1(mu1), mu2(mu2), mu3(mu3) {}

    void operator()(const State &x, State &dxdt, double /*t*/) const {
        dxdt[0] = -x[0] + x[1];
        dxdt[1] = (3 + mu1) * x[0] + x[1] * (1 + mu2) - x[0] * x[2];
        dxdt[2] = -(2 + mu3) * x[2] + x[0] * x[1];
    }
};

//---------------------------------------------------------------------------
namespace config {
    std::string conf = "hg.cfg";
    std::string out  = "hg.h5";

    double x0 = 1e-3;
    double y0 = 0;
    double z0 = 0;

    double dt   = 0.01;
    double tmax = 1000.0;
    int    kmax = 16;

    double mu1 = 1;

    double mu2_min   = -2.0;
    double mu2_max   =  0.0;
    int    mu2_steps =  1000;

    double mu3_min   = -2.0;
    double mu3_max   = -1.0;
    int    mu3_steps =  1000;

    void read(int argc, char *argv[]);
} // namespace config

//---------------------------------------------------------------------------
vex::generator::Kernel<10> make_kernel(const vex::Context &ctx);
void save_kneading(vex::vector<cl_ulong> &k);

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    try {
        config::read(argc, argv);
        const int n = config::mu2_steps * config::mu3_steps;

        // Initialize VexCL context
        vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
        std::cout << ctx << std::endl;

        // Fill parameters.
        double h2 = (config::mu2_max - config::mu2_min) / (config::mu2_steps - 1);
        double h3 = (config::mu3_max - config::mu3_min) / (config::mu3_steps - 1);

        vex::vector<double> mu2(ctx, n);
        vex::vector<double> mu3(ctx, n);

        mu2 = config::mu2_min + h2 * (vex::element_index() % config::mu2_steps);
        mu3 = config::mu3_min + h3 * (vex::element_index() / config::mu2_steps);

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
            sweep(x, y, z, dx, dy, dz, num, seq, mu2, mu3);

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
        OPTION(conf,      "Configuration file")
        OPTION(out,       "Output file")
        OPTION(x0,        "Initial X xoordinate")
        OPTION(y0,        "Initial Y xoordinate")
        OPTION(z0,        "Initial Z xoordinate")
        OPTION(dt,        "Time step")
        OPTION(tmax,      "Time limit")
        OPTION(kmax,      "Length of kneading sequence")
        OPTION(mu1,       "Model parameter mu1")
        OPTION(mu2_min,   "Minimum mu2 value")
        OPTION(mu2_max,   "Maximum mu2 value")
        OPTION(mu2_steps, "Number of mu2 values")
        OPTION(mu3_min,   "Minimum mu3 value")
        OPTION(mu3_max,   "Maximum mu3 value")
        OPTION(mu3_steps, "Number of mu3 values")
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

    sym_vector mu2(sym_vector::VectorParameter, sym_vector::Const);
    sym_vector mu3(sym_vector::VectorParameter, sym_vector::Const);

    vex::symbolic<cl_short> num(vex::symbolic<cl_short>::VectorParameter);
    vex::symbolic<cl_ulong> seq(vex::symbolic<cl_ulong>::VectorParameter);

    body << "if (" << num << " >= " << config::kmax << ") continue;\n";

    // Stepper type
    odeint::runge_kutta_dopri5<
        sym_state, double, sym_state, double,
        odeint::range_algebra, odeint::default_operations
        > stepper;

    // Record single RK4 step
    hg_system<sym_state, sym_vector> sys{config::mu1, mu2, mu3};
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
    return vex::generator::build_kernel(ctx, "hg_sweep", body.str(),
            x[0], x[1], x[2], dx[0], dx[1], dx[2], num, seq, mu2, mu3);
}

//---------------------------------------------------------------------------
void save_kneading(vex::vector<cl_ulong> &k) {
    using namespace H5;

    H5File hdf(config::out, H5F_ACC_TRUNC);

    hsize_t dim[] = {
        static_cast<hsize_t>(config::mu3_steps),
        static_cast<hsize_t>(config::mu2_steps)
    };

    hsize_t one[] = { 1 };
    DataSpace adsp(1, one);

    std::vector<cl_ulong> k_host(k.size());
    vex::copy(k, k_host);

    DataSet ds = hdf.createDataSet("/K", PredType::NATIVE_UINT64, DataSpace(2, dim));
    ds.write(k_host.data(), PredType::NATIVE_UINT64);

#define CREATE_ATTRIBUTE(name, type) \
    ds.createAttribute(#name, type, adsp).write(type, &config::name)

    CREATE_ATTRIBUTE(x0,        PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(y0,        PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(z0,        PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(dt,        PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(kmax,      PredType::NATIVE_INT32);
    CREATE_ATTRIBUTE(mu1,       PredType::NATIVE_DOUBLE);

#undef CREATE_ATTRIBUTE

#define CREATE_ATTRIBUTE(name, val, type) \
    ds.createAttribute(#name, type, adsp).write(type, &config::val)

    CREATE_ATTRIBUTE(xmin, mu2_min,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(xmax, mu2_max,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(xnum, mu2_steps, PredType::NATIVE_INT32);

    CREATE_ATTRIBUTE(ymin, mu3_min,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(ymax, mu3_max,   PredType::NATIVE_DOUBLE);
    CREATE_ATTRIBUTE(ynum, mu3_steps, PredType::NATIVE_INT32);

#undef CREATE_ATTRIBUTE
}

