#pragma once

#include <stddef.h>

namespace dopt
{
    namespace diff_approximators
    {
        /** Numericall evalute deirviative of function F(t) = f(x + diretion * t) at point t = 0.
        * @param f vector argument and scalar valued function
        * @param x point in domain
        * @param direction unit vector direction that piercies domain from point "x"
        * @remark Error of this approximation is f'''/3! * h^2
        */
        template<class Func, class TVec, class TArgType>
        TArgType evalDerivative(Func& f, const TVec& x, 
                                const TVec& direction, const TArgType& dt)
        {
            // https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

            auto f_plus_h = f(x + direction * dt);
            auto f_minus_h = f(x - direction * dt);
            auto result = (f_plus_h - f_minus_h) / (2 * dt);

            return result;
        }

        /** Numericall evalute second deirviative of function F(t) = f(x + diretion * t) at point t = 0.
        * @param f vector argument and scalar valued function
        * @param x point in domain
        * @param direction unit vector direction that piercies domain from point "x"
        * @remark Error of this approximation is f''''/12 * h^2
        */
        template<class Func, class TVec, class TArgType>
        TArgType evalSecondDerivative(Func& f, const TVec& x, 
                                      const TVec& direction, const TArgType& dt)
        {
            // https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
            auto f_plus_h = f(x + direction * dt);
            auto f_ = f(x);
            auto f_minus_h = f(x - direction * dt);

            auto result = (f_plus_h - 2 * f_ + f_minus_h) / (dt * dt); 
            return result;
        }

        template<class Func, class TVec, class TArgType>
        TVec evaluateGradientNumerically(Func& f, const TVec& x, const TArgType& dt)
        {
            TVec res(x.size());
            TVec direction(x.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                direction.set(i, TArgType(1));
                res.set(i, evalDerivative(f, x, direction, dt));
                direction.set(i, TArgType(0));
            }

            return res;
        }

        /**
         * Evaluates the Hessian matrix of a given function numerically at a specified point.
         *
         * @param f A function object that takes a vector argument and returns a scalar value.
         * @param x The point in the domain at which the Hessian matrix is to be evaluated.
         * @param dt The step size for numerical differentiation.
         * @return The Hessian matrix of the function evaluated at the point x.
         */
        template<class Mat, class Func, class TVec, class TArgType>
        Mat evaluateHessianNumerically(Func& f, const TVec& x, const TArgType& dt)
        {
            Mat hessian(x.size(), x.size());
            TArgType invDtPartial = TArgType(1.0) / TArgType(4 * dt * dt);

            for (size_t i = 0; i < x.size(); ++i)
            {
                for (size_t j = 0; j < i; ++j)
                {
                    // evaluate second order mixed partial derivatives w.r.t. to different variables (Error ~ dt * dt)
                    {
                        // http://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h07/undervisningsmateriale/kap7.pdf

                        TVec pt[4] = { x, x, x, x };
                        pt[0][i] += dt;
                        pt[0][j] += dt;

                        pt[1][i] += dt;
                        pt[1][j] -= dt;

                        pt[2][i] -= dt;
                        pt[2][j] += dt;

                        pt[3][i] -= dt;
                        pt[3][j] -= dt;

                        auto f0 = f(pt[0]);
                        auto f1 = f(pt[1]);
                        auto f2 = f(pt[2]);
                        auto f3 = f(pt[3]);

                        auto res = (f0 - f1 - f2 + f3) * invDtPartial;

                        hessian.set(i, j, res);
                        hessian.set(j, i, res);
                    }
                }
                
                // evaluate usual second order partial derivatives w.r.t to variable i
                {
                    TVec direction(x.size());
                    direction.set(i, TArgType(1));
                    auto res = evalSecondDerivative(f, x, direction, dt);
                    hessian.set(i, i, res);
                }                
            }
            
            return hessian;
        }
    }
}
