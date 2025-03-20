#pragma once

#include "dopt/optimization_problems/include/problem_specification.h"
#include <stddef.h>

namespace dopt
{
    /** Symmetric Quadratics
    * f(x)=1/2 <x,Ax> + <b,x> + c 
    */
    template<class Mat, class TVec, class TElemenType>
    class SymmetricQuadratics final: public ProblemSpecification
    {
    public:
        SymmetricQuadratics(const Mat& a, const TVec& b, const TElemenType& c)
        : A(a)
        , B(b)
        , C(c)
        {
        }

        bool isOk() {
            return A.isSymmetric();
        }

        TElemenType operator()(const TVec& x)
        {
            return evaluateFunction(x);
        }
        
        TElemenType evaluateFunction(const TVec& x)
        {
            auto fist_part   = x & (A * x) * 0.5;
            auto second_part = x & B;
            auto third_part  = C;

            auto res = fist_part + second_part + third_part;
            return res;
        }

        TVec evaluateGradient(const TVec& x)
        {
            auto fist_part = A * x;
            auto second_part = B;

            auto res = fist_part + second_part;
            return res;
        }

        Mat evaluateHessian(const TVec& x)
        {
            return A;
        }

        //=========================================================//
        size_t getInputVariableDimension() override
        {
            return B.size();
        }

        bool isGradientOracleAvailable() override
        {
            return true;
        }

        bool isHessianOracleAvailable() override
        {
            return true;
        }
        //=========================================================//
    private:
        Mat A;
        TVec B;
        TElemenType C;
    };
}
