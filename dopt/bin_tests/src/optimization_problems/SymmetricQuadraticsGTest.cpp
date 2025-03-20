#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/random/include/RandomGenRealLinear.h"

#include "dopt/optimization_problems/include/numerics/symmetric_quadratics.h"
#include "dopt/numerics/include/differentation_helpers.h"

#include "gtest/gtest.h"

#include <math.h>

TEST(dopt, SymmetricQuadraticsGTest)
{
    using V = dopt::VectorNDStd_d;
    using M = dopt::MatrixNMD<V>;

    dopt::RandomGenRealLinear gen(1.0, 100.0);

    for (size_t dim = 20; dim < 30; dim += 2)
    {
        M a = M(dim, dim);
        V b = V(dim);

        for (size_t i = 0; i < dim * dim; ++i)
            a.matrixByCols.set(i, gen.generateReal());
        for (size_t i = 0; i < dim; ++i)
            b.set(i, gen.generateReal());
        V::TElementType c = gen.generateReal();
        a = a.getTranspose() * a;

        dopt::SymmetricQuadratics opt_problem(a, b, c);

        EXPECT_TRUE(opt_problem.getInputVariableDimension() == dim);
        EXPECT_TRUE(opt_problem.getOptDimension() == dim);

        V x(dim);
        for (size_t i = 0; i < dim; ++i)
            b.set(i, gen.generateReal());

        V grad_oracle = opt_problem.evaluateGradient(x);
        V grad_oracle_compute = dopt::diff_approximators::evaluateGradientNumerically(opt_problem, x, 0.0001);
        EXPECT_TRUE((grad_oracle - grad_oracle_compute).vectorL2Norm() < 0.01);

        M hessian_oracle = opt_problem.evaluateHessian(x);
        M hessian_compute = dopt::diff_approximators::evaluateHessianNumerically<M>(opt_problem, x, 0.0001);
        EXPECT_TRUE((hessian_compute - hessian_oracle).frobeniusNorm() < 0.01);
    }
}
