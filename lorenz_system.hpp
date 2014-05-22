#ifndef LORENZ_SYSTEM_HPP
#define LORENZ_SYSTEM_HPP

// The three-parameter extension of the Shimizu-Morioka model.
template <class State, class Parameter>
struct lorenz_system {
    const Parameter &alpha;
    const Parameter &lambda;
    const double B;

    lorenz_system(const Parameter &alpha, const Parameter &lambda, double B = 0)
        : alpha(alpha), lambda(lambda), B(B)
    {}

    void operator()(const State &x, State &dxdt, double /*t*/) const {
        dxdt[0] = x[1];
        dxdt[1] = x[0] - lambda * x[1] - x[0] * x[2] - B * x[0] * x[0] * x[0];
        dxdt[2] = -alpha * x[2] + x[0] * x[0];
    }
};

#endif
