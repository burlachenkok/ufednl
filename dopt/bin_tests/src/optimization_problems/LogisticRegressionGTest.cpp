#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/random/include/RandomGenRealLinear.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"

#include "dopt/optimization_problems/include/ml/logistic_regression.h"
#include "dopt/numerics/include/differentation_helpers.h"

#include "gtest/gtest.h"

template<size_t kAlgorithm>
void executeL2RegulirizeLogisticRegressionTest()
{
    using V = dopt::VectorNDStd_d;
    using M = dopt::MatrixNMD<V>;

    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenRealLinear gen(1.0, 100.0);
    dopt::RandomGenIntegerLinear labels(0, 1, 9090);

    for (size_t dim = 20; dim < 40; dim += 2)
    {
        for (size_t m = 10; m < 60; m += 10)
        {

            M a = M(m, dim); // features
            V b = V(m);      // labels

            for (size_t i = 0; i < m * dim; ++i)
                a.matrixByCols.set(i, gen.generateReal());

            for (size_t i = 0; i < m; ++i)
            {
                if (labels.generateInteger() == 0)
                    b.set(i, 1.0);
                else
                    b.set(i, -1.0);
            }

            dopt::L2RegulirizeLogisticRegression opt_problem(a.getTranspose(), b, 0.01);

            EXPECT_TRUE(opt_problem.getInputVariableDimension() == dim);
            EXPECT_TRUE(opt_problem.getOptDimension() == dim);
            EXPECT_TRUE(opt_problem.getOutputVariableDimension() == 1);

            V x(dim);
            x.setAllRandomly(gen_x);

            V grad_oracle = opt_problem.evaluateGradient(x);
            V grad_oracle_compute = dopt::diff_approximators::evaluateGradientNumerically(opt_problem, x, 0.00001);
            EXPECT_TRUE((grad_oracle - grad_oracle_compute).vectorL2Norm() < 0.01);

            M hessian_oracle = opt_problem.template evaluateHessian<true, kAlgorithm>(x);
            M hessian_compute = dopt::diff_approximators::evaluateHessianNumerically<M>(opt_problem, x, 0.00001);

            auto hessian_diff = (hessian_compute - hessian_oracle).frobeniusNorm();

            EXPECT_TRUE(hessian_diff < 0.01);

            // Check different flavors of function evaluation
            auto margin = opt_problem.evaluateClassificationMargin(x);
            auto margin_sigmoid = opt_problem.evaluateClassificationMarginSigmoid(margin);

            auto f1 = opt_problem.evaluateFunction(x);
            auto f2 = opt_problem.evaluateFunction(x);
            auto f3 = opt_problem.evaluateFunction(x, margin, margin_sigmoid);

            EXPECT_TRUE(fabs(f1 - f2) < 0.001);
            EXPECT_TRUE(fabs(f2 - f3) < 0.001);

            // Check different flavors of gradient evaluation
            auto g1 = opt_problem.evaluateGradient(x);
            auto g2 = opt_problem.evaluateGradient(x, margin);
            auto g3 = opt_problem.evaluateGradient(x, margin, margin_sigmoid);

            EXPECT_TRUE((g1 - g2).vectorL2Norm() < 0.01);
            EXPECT_TRUE((g2 - g3).vectorL2Norm() < 0.01);

            // Check different flavors of Hessian evaluation
            auto h1 = opt_problem.template evaluateHessian<true, kAlgorithm> (x);
            auto h2 = opt_problem.template evaluateHessian<true, kAlgorithm> (x, margin);
            auto h3 = opt_problem.template evaluateHessian<true, kAlgorithm> (x, margin, margin_sigmoid);
            auto h4 = opt_problem.template evaluateHessian<false, kAlgorithm>(x);

            EXPECT_TRUE((h1 - h2).frobeniusNorm() < 0.01);
            EXPECT_TRUE((h2 - h3).frobeniusNorm() < 0.01);
            EXPECT_TRUE(M::computeDifferenceWithUpperTriangularPart(h1, h4).frobeniusNorm() < 0.01);
        }
    }
}

TEST(dopt, L2RegulirizeLogisticRegressionGTest)
{
    executeL2RegulirizeLogisticRegressionTest<1>();
    executeL2RegulirizeLogisticRegressionTest<2>();
    executeL2RegulirizeLogisticRegressionTest<3>();
    executeL2RegulirizeLogisticRegressionTest<4>();
    executeL2RegulirizeLogisticRegressionTest<5>();
}

template<size_t kAlgorithm>
void executeL2RegulirizeLogisticPerfTest()
{
    using V = dopt::VectorNDStd_d;
    using M = dopt::MatrixNMD<V>;

    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenRealLinear gen(1.0, 100.0);
    dopt::RandomGenIntegerLinear labels(0, 1, 9090);

    size_t dim = 100;
    size_t m = 25000;

    M a = M(m, dim); // features
    V b = V(m);      // labels
    V x = V(dim);      // iterate

    // Initialize randomly
    //a.matrixByCols.setAllRandomly(gen);
    //x.setAllRandomly(gen);

    for (size_t i = 0; i < m; ++i)
    {
        if (labels.generateInteger() == 0)
        {
            b.set(i, 1.0);
        }
        else
        {
            b.set(i, -1.0);
        }
    }
    dopt::L2RegulirizeLogisticRegression opt_problem(a.getTranspose(), b, 0.01);

    dopt::HighPrecisionTimer timer_main;
    V grad_oracle = opt_problem.evaluateGradient(x);
    double deltaMsGrad = timer_main.getTimeMs();
    std::cout << "Time spent to gradient oracle for Logistic Regression: " << deltaMsGrad << " milliseconds" << "(dimension: " << dim << " samples: " << m << ")" << " [algorithm:" << kAlgorithm << "]" << '\n';

    timer_main.reset();
    M h = opt_problem.template evaluateHessian<true, kAlgorithm> (x);
    double deltaMsHessian = timer_main.getTimeMs();
    std::cout << "Time spent to hessian oracle for Logistic Regression: " << deltaMsHessian << " milliseconds" << "(dimension: " << dim << " samples: " << m << ")" << " [algorithm:" << kAlgorithm << "]" << '\n';
}

TEST(dopt, LogisticRegressionGPerfTest)
{
    executeL2RegulirizeLogisticPerfTest<1>();
    executeL2RegulirizeLogisticPerfTest<2>();
    executeL2RegulirizeLogisticPerfTest<3>();
    executeL2RegulirizeLogisticPerfTest<4>();
    executeL2RegulirizeLogisticPerfTest<5>();
}
