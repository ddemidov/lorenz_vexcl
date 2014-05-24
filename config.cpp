#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <boost/program_options.hpp>
#include "config.hpp"

namespace config {

std::string conf = "lorenz.cfg";
std::string out  = "kneading.h5";

double x0 = 1e-3;
double y0 = 0;
double z0 = 0;

double dt   = 0.01;
double tmax = 1000.0;
int    kmax = 16;

double alpha_min   = 0.01;
double alpha_max   = 0.7;
int    alpha_steps = 1000;

double lambda_min   = 0.3;
double lambda_max   = 1.5;
int    lambda_steps = 1000;

double B = 0;

template <typename T>
std::string to_string(const T &val) {
    std::ostringstream s;
    s << std::setprecision(3) << val;
    return s.str();
}

void read(int argc, char *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Options");

    std::string conf_file = "dem.cfg";

#define OPTION(name, descr)                                                    \
    (#name,                                                                    \
     po::value<decltype(name)>(&name)->default_value(name, to_string(name)),   \
     descr                                                                     \
    )

    desc.add_options()
        ("help,h", "Show help")
        OPTION(conf,         "Configuration file")
        OPTION(out,          "Output file")
        OPTION(x0,           "Initial X xoordinate")
        OPTION(y0,           "Initial Y xoordinate")
        OPTION(z0,           "Initial Z xoordinate")
        OPTION(dt,           "Time step")
        OPTION(tmax,         "Time limit")
        OPTION(kmax,         "Length of kneading sequence")
        OPTION(alpha_min,    "Minimum alpha value")
        OPTION(alpha_max,    "Maximum alpha value")
        OPTION(alpha_steps,  "Number of alpha values")
        OPTION(lambda_min,   "Minimum lambda value")
        OPTION(lambda_max,   "Maximum lambda value")
        OPTION(lambda_steps, "Number of lambda values")
        OPTION(B,            "Model parameter B")
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

}
