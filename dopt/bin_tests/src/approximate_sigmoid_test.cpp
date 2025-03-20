#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/timers/include/HighPrecisionTimer.h"
#include "dopt/numerics/include/interpolation_methods.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>

#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

#if 0
double usual_sigmoid(double x)
{
    double y = 1.0 / (1.0 + exp(-x));
    return y;
}

template<double exp_minus_x0, double x0, bool x_always_negative = false>
double optimized_sigmoid_series_3(double x)
{
    constexpr double minus_exp_minus_x0 = -exp_minus_x0;

    if (x_always_negative || x < 0)
    {
        double dh_pow_1 = (x - x0);
        double dh_pow_2 = (x - x0) * (x - x0);
        double dh_pow_3 = (x - x0) * (x - x0) * (x - x0);

        double compute = 1 /( 1 + (exp_minus_x0 + 
                                   minus_exp_minus_x0 * dh_pow_1 +
                                   exp_minus_x0 * dh_pow_2 / 2 +
                                   minus_exp_minus_x0 * dh_pow_3 / 6) );
        return compute;
    }
    else 
    {
        x = -x;

        double dh_pow_1 = (x - x0);
        double dh_pow_2 = (x - x0) * (x - x0);
        double dh_pow_3 = (x - x0) * (x - x0) * (x - x0);

        double compute = 1 /( 1+ (exp_minus_x0 +
                                  minus_exp_minus_x0 * dh_pow_1 +
                                  exp_minus_x0 * dh_pow_2 / 2 +
                                  minus_exp_minus_x0 * dh_pow_3 / 6) );
        return 1.0 - compute;
    }
}

template<double exp_minus_x0, double x0, bool x_always_negative = false>
double optimized_sigmoid_series_2(double x)
{
    constexpr double minus_exp_minus_x0 = -exp_minus_x0;

    if (x_always_negative || x < 0)
    {
        double dh_pow_1 = (x - x0);
        double dh_pow_2 = (x - x0) * (x - x0);

        double compute = 1 / (1 + (exp_minus_x0 +
                                   minus_exp_minus_x0 * dh_pow_1 +
                                   exp_minus_x0 * dh_pow_2 / 2) );
        return compute;
    }
    else
    {
        x = -x;

        double dh_pow_1 = (x - x0);
        double dh_pow_2 = (x - x0) * (x - x0);

        double compute = 1 /( 1 + (exp_minus_x0 +
                                   minus_exp_minus_x0 * dh_pow_1 +
                                   exp_minus_x0 * dh_pow_2 / 2) );

        return 1.0 - compute;
    }
}

double optimized_sigmoid_a(double x)
{
    return optimized_sigmoid_series_3</*exp_minus_x0*/ 1.0, /*minus_x0*/ 0.0>(x);
}

double optimized_sigmoid_b(double x)
{
    return optimized_sigmoid_series_2</*exp_minus_x0*/ 1.0, /*minus_x0*/ 0.0>(x);
}

double optimized_sigmoid_c(double x)
{
    if (x > 0.0)
    {
        return 1.0 - optimized_sigmoid_c(-x);
    }
    
    if (x < -3.5)
    {
        constexpr double exp_at_3_5 = 33.11545195869231;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_3_5, /*x0*/ -3.5, /*x_always_negative*/ true>(x);
    }
    else if (x < -2.5)
    {
        constexpr double exp_at_2_5 = 12.182493960703473;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_2_5, /*x0*/ -2.5, /*x_always_negative*/ true>(x);
    }
    else if (x < -2.0)
    {
        constexpr double exp_at_2_0 = 7.38905609893065;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_2_0, /*x0*/ -2.0, /*x_always_negative*/ true>(x);
    }
    else if (x < -1.5)
    {
        constexpr double exp_at_1_5 = 4.4816890703380645;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_1_5, /*x0*/ -1.5, /*x_always_negative*/ true>(x);
    }
    else if (x < -1.0)
    {
        constexpr double exp_at_1_0 = 2.718281828459045;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_1_0, /*x0*/ -1.0, /*x_always_negative*/ true>(x);
    }
    else if (x < -0.5)
    {
        constexpr double exp_at_0_5 = 1.6487212707001282;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_0_5, /*x0*/ -0.5, /*x_always_negative*/ true>(x);
    }
    else if (x < -0.25)
    {
        constexpr double exp_at_0_25 = 1.2840254166877414;
        return optimized_sigmoid_series_3</*exp_minus_x0*/ exp_at_0_25, /*x0*/ -0.5, /*x_always_negative*/ true>(x);
    }

#if 0
    else if (x < -400.0)
    {
        constexpr double exp_400 = 5.221469689764144e+173;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_400, /*x0*/ -400, /*x_always_negative*/ true>(x);
    }
    else if (x < -300.0)
    {
        constexpr double exp_300 = 1.9424263952412558e+130;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_300, /*x0*/ -300, /*x_always_negative*/ true>(x);
    }
    else if (x < -200.0)
    {
        constexpr double exp_200 = 7.225973768125749e+86;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_200, /*x0*/ -200, /*x_always_negative*/ true>(x);
    }
    else if (x < -100.0)
    {
        constexpr double exp_100 = 2.6881171418161356e+43;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_100, /*x0*/ -100, /*x_always_negative*/ true>(x);
    }
    else if (x < -75.0)
    {
        constexpr double exp_75 = 3.7332419967990015e+32;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_75, /*x0*/ -75, /*x_always_negative*/ true>(x);
    }
    else if (x < -50.0)
    {
        constexpr double exp_50 = 5.184705528587072e+21;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_50, /*x0*/-50, /*x_always_negative*/ true>(x);
    }
    else if (x < -25.0)
    {
        constexpr double exp_25 = 72004899337.38588;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_25, /*x0*/ -25, /*x_always_negative*/ true>(x);
    }
    else if (x < -15.0)
    {
        constexpr double exp_15 = 3269017.3724721107;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_15, /*x0*/ -15, /*x_always_negative*/ true>(x);
    }
    else if (x < -5.0)
    {
        constexpr double exp_5 = 148.4131591025766;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_5, /*x0*/ -5, /*x_always_negative*/ true>(x);
    }
    else if (x < -2.5)
    {
        constexpr double exp_2_5 = 12.182493960703473;
        return optimized_sigmoid_series_2</*exp_minus_x0*/ exp_2_5, /*x0*/ -2.5, /*x_always_negative*/ true>(x);
    }
#endif
    else
    {
        return optimized_sigmoid_series_2</*exp_minus_x0*/ 1.0, /*x0*/ 0.0, /*x_always_negative*/ true>(x);
    }
}

template<class F>
void test(F optimized_sigmoid, const char* optimized_sigmoid_name)
{
    dopt::HighPrecisionTimer tm1;

    const int64_t experiments = 100* 1000;
    
    std::cout << "Optimization Implementation Name: " << optimized_sigmoid_name << '\n';
    std::cout << "Number of experiments: " << experiments << '\n';
    std::cout << "Sizeof of userspace pointer: " << sizeof(void*) << '\n';

    volatile double fake = 0.0;
    volatile double t_start = -10000.0;
    volatile double t_end = 10000.0;
    volatile double dt = (t_end - t_start) / experiments;
    {
        volatile double ti = t_start;
        tm1.reset();
        for (int64_t i = 0; i < experiments; ++i, ti += dt)
        {
            fake += optimized_sigmoid(ti);
        }
        printf("Time for optimized_sigmoid(): %lf seconds\n", tm1.getTimeSec());
        std::cout << "  last ti: " << ti << " (Expected around: " << t_end << ")" << '\n';
    }

    {
        volatile double ti = t_start;
        tm1.reset();
        for (int64_t i = 0; i < experiments; ++i, ti += dt)
        {
            fake += usual_sigmoid(ti);
        }
        printf("Time for usual_sigmoid(): %lf seconds\n", tm1.getTimeSec());
        std::cout << "  last ti: " << ti << " (Expected around: " << t_end << ")" << '\n';
    }

    {
        volatile double ti = t_start;
        double max_err = 0.0;
        double max_err_ti = 0.0;

        tm1.reset();
        for (int64_t i = 0; i < experiments; ++i, ti += dt)
        {
            double err = fabs(optimized_sigmoid(ti) - usual_sigmoid(ti));
            fake += err;
            if (err > max_err || i == 0)
            {
                max_err = err;
                max_err_ti = ti;
            }
        }
        printf("Time for optimized_sigmoid() - usual_sigmoid(): %lf seconds\n", tm1.getTimeSec());
        printf("Maximum error for varying argument in [%lf,%lf] is %lf\n", t_start, t_end, max_err);
        printf("Maximum error for varying argument in [%lf,%lf] is in point %lf\n", t_start, t_end, max_err_ti);
        std::cout << "  last ti: " << ti << " (Expected around: " << t_end << ")" << '\n';
    }
}

#if 0

class SigmoidInterpolator
{
public:
    SigmoidInterpolator(double min, double max, size_t theKnots)
    {
        size_t knots = theKnots;
        
        if (knots < 2)
        {
            knots = 2;
        }
        
        x_knots.resize(knots);
        y_knots.resize(knots);

        double dx = (max - min) / (knots - 1);
        double x = min;

        for (size_t i = 0; i < knots; ++i, x += dx)
        {
            x_knots[i] = x;
            y_knots[i] = usual_sigmoid(x) - optimized_sigmoid_series_2</*exp_minus_x0*/ 1.0, /*minus_x0*/ 0.0, /*x_always_negative*/ true>(x);
        }
    }
    
    double operator()(double x)
    {
        if (x > 0)
        {
            return 1.0 - SigmoidInterpolator::operator() (-x);
        }
        
        double main_part = optimized_sigmoid_series_2</*exp_minus_x0*/ 1.0, /*minus_x0*/ 0.0, /*x_always_negative*/ true>(x);
        
        dopt::interpolators::LinearInterpolation<double, double> interpolator(x_knots.data(), 
                                                                              y_knots.data(), 
                                                                              x_knots.size());
        
        double rest = interpolator.interpolate(x);
        
        double final_value = main_part + rest;
        
        return final_value;
    }
    
private:
    std::vector<double> x_knots;
    std::vector<double> y_knots;
};
#endif

TEST(dopt, ApproximateSigmoidGTest)
{
    //test(optimized_sigmoid_a, "optimized_sigmoid_a");
    //test(optimized_sigmoid_b, "optimized_sigmoid_b");
    //test(optimized_sigmoid_c, "optimized_sigmoid_c");
    //test(optimized_sigmoid_d, "optimized_sigmoid_d");
    //SigmoidInterpolator approx(-10000.0 * 2, 0.0, 10000);
    //test(approx, "approx");
}

#endif
