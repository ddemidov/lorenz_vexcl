#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace config {

extern std::string conf;

extern double x0;
extern double y0;
extern double z0;

extern double dt;
extern double tmax;
extern int    kmax;

extern double alpha_min;
extern double alpha_max;
extern int    alpha_steps;

extern double lambda_min;
extern double lambda_max;
extern int    lambda_steps;

extern double B;
extern double q;

void read(int argc, char *argv[]);

}

#endif
