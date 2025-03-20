#pragma once

#include <stddef.h>

namespace dopt
{
    struct ProblemSpecification
    {
        virtual ~ProblemSpecification() = default;

        /** Optimization variable dimension
        */
        virtual size_t getInputVariableDimension() = 0;

        virtual bool isGradientOracleAvailable() = 0;

        virtual bool isHessianOracleAvailable() = 0;

        /** Dimension of output variable of minimization function
        */
        virtual size_t getOutputVariableDimension()
        {
            return 1;
        }

        /** Get optimization problem dimension
        */
        size_t getOptDimension()
        {
            return getInputVariableDimension();
        }
    };
}
