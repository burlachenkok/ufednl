#include "dopt/linalg_linsolvers/include/ElementarySolvers.h"
#include "dopt/linalg_linsolvers/include/IterativeSolvers.h"

#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"

#include "dopt/linalg_linsolvers/include/GaussEliminationSolvers.h"

#include "dopt/random/include/RandomGenRealLinear.h"
#include "gtest/gtest.h"

TEST(dopt, LinearEquationsGTest)
{
    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(2, 2);
        V b(2);

        a.setRow(0, {1.0, 2.0});
        a.setRow(1, {2.0, 4.0});
        b.set(0, 5.0);
        b.set(1, 7.0);

        EXPECT_FALSE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));
    }

    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(4, 4);
        V b(4);

        a.setRow(0, {1.0, -1.0, 1.0, -1.0});
        a.setRow(1, {1.0, +1.0, 2.0, +3.0});
        a.setRow(2, {2.0, +4.0, 5.0, 10.0});
        a.setRow(3, {2.0, -4.0, 1.0, -6.0});

        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));

        V x = dopt::gausEleminationSolver(a, b, (std::vector<V>*)nullptr, false, 1.0e-6);
        EXPECT_TRUE(x.size() == a.columns());
        auto residual_1 = (a * x - b).vectorL2Norm();
        EXPECT_TRUE(residual_1 < 1.0e-6);
        
        std::vector<V> nullspace;
        x = dopt::gausEleminationSolver(a, b, &nullspace, false, 1.0e-6);
        EXPECT_TRUE(nullspace.size() == 2);

        auto residual_2 = (a * x  + a * (nullspace[0] * 3.0 + nullspace[1] * 2.0) - b).vectorL2Norm();
        EXPECT_TRUE(residual_2 < 1.0e-6);
    }
    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(4, 4);
        V b(4);

        a.setRow(0, { 1.0, -1.0, 1.0, -1.0 });
        a.setRow(1, { 1.0, +1.0, 2.0, +3.0 });
        a.setRow(2, { 2.0, +4.0, 5.0, 10.0 });
        a.setRow(3, { 2.0, -4.0, 1.0, -6.0 });

        b[0] = 4.0;
        b[1] = 8.0;
        b[2] = 20.0;
        b[3] = 5.0;
        EXPECT_FALSE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));
    }
    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(4, 4);
        V b(4);

        a.setRow(0, { 1.0, -1.0, 1.0, -1.0 });
        a.setRow(1, { 1.0, +1.0, 2.0, +3.0 });
        a.setRow(2, { 2.0, +4.0, 5.0, 10.0 });
        a.setRow(3, { 2.0, -4.0, 1.0, -6.0 });

        b[0] = 4.0;
        b[1] = 8.0;
        b[2] = 20.0;
        b[3] = 4.0;
        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));

        V x = dopt::gausEleminationSolver(a, b, (std::vector<V>*)nullptr, false, 1.0e-6);
        EXPECT_TRUE(x.size() == a.columns());
        auto residual_1 = (a * x - b).vectorL2Norm();
        EXPECT_TRUE(residual_1 < 1.0e-6);

        std::vector<V> nullspace;
        x = dopt::gausEleminationSolver(a, b, &nullspace, false, 1.0e-6);
        EXPECT_TRUE(nullspace.size() == 2);

        auto residual_2 = (a * x + a * (nullspace[0] * 3.0 + nullspace[1] * 2.0) - b).vectorL2Norm();
        EXPECT_TRUE(residual_2 < 1.0e-6);
    }

    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(2, 4);
        V b(2);
        a.setRow(0, {1.0, 1.0, 1.0, 3.0});
        a.setRow(1, {1.0, 1.0, 2.0, 1.0});
        b[0] = 1.0;
        b[1] = 3.0;
        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));

        std::vector<V> nullspace;
        V x = dopt::gausEleminationSolver(a, b, &nullspace, false, 1.0e-6);
        EXPECT_TRUE(x.size() == a.columns());

        auto residual_1 = (a * x - b).vectorL2Norm();
        EXPECT_TRUE(residual_1 < 1.0e-6);

        auto residual_sol = (x - V({ -1.0, 0.0, 2.0, 0.0 })).vectorL2Norm();
        EXPECT_TRUE(residual_sol < 1.0e-6);
        EXPECT_TRUE(nullspace.size() == 2);

        auto vec1 = nullspace[0];
        auto vec2 = nullspace[1];

        EXPECT_TRUE( (a * (x + 0.2 * vec1 + 0.67 * vec2) - b).vectorL2Norm() < 1.0e-6 );
        EXPECT_TRUE(vec1 == V({ -1.0, 1.0, 0.0, 0.0 })) << "Take expected first vector from null space based on knowledge under the hood algorithm";
        EXPECT_TRUE(vec2 == V({ -5.0, 0.0, 2.0, 1.0 })) << "Take expected second vector from null space based on knowledge under the hood algorithm";
    }
    {
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(4, 4);
        V b(4);
        a.setRow(0, {2.0, -1.0, 0.0, 0.0});
        a.setRow(1, {9.0, +1.0, 2.0, 0.0});
        a.setRow(2, { 0.0, +4.0, 5.0, 10.0 });
        a.setRow(3, { 0.0, 0.0, 11.0, -6.0 });

        b[0] = 4.0;
        b[1] = 8.0;
        b[2] = 20.0;
        b[3] = 4.0;

        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));
        auto x3Diag = dopt::sweepSolver(a, b);
        EXPECT_TRUE((a* x3Diag - b).vectorL2Norm() < 1e-6);

        b = a.getTranspose() * b;
        a = a.getTranspose() * a;
        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-6));

        auto xGausElimWithSelection = dopt::gausEleminationSolver(a, b);
        EXPECT_TRUE((a* xGausElimWithSelection - b).vectorL2Norm() < 1e-6);

        {
            bool kJacobiIsApplied = dopt::linear_solvers_helpers::isMatrixHasDiagonalDominance(a);

            V xJacobi[2] = { V(b.size()), V(b.size()) };
            for (size_t i = 0; i < 500; ++i)
            {
                dopt::jacobiSolverIteration(xJacobi[(0 + i % 2) % 2], a, b, xJacobi[(1 + i % 2) % 2]);
                //std::cout << " jacobiSolverIteration error: " << (a * xJacobi[0] - b).vectorL2Norm() << '\n';
            }

            EXPECT_TRUE((a * xJacobi[0] - b).vectorL2Norm() < 1e-6);
            std::cout << " jacobiSolverIteration error: " << (a * xJacobi[0] - b).vectorL2Norm() << '\n';
        }

        {
            V xSeidel = V(b.size());
            for (size_t i = 0; i < 250; ++i)
            {
                dopt::seidelSolverIteration(xSeidel, a, b);
                // std::cout << " seidelSolverIteration error: " << (a * xSeidel - b).vectorL2Norm() << '\n';
            }
            EXPECT_TRUE((a * xSeidel - b).vectorL2Norm() < 1e-6);
            std::cout << " seidelSolverIteration error: " << (a * xSeidel - b).vectorL2Norm() << '\n';
        }
        {
            std::vector<V> nullspace;
            V xGauss = dopt::gausEleminationSolver(a, b, &nullspace);

            EXPECT_TRUE((a* xGauss - b).vectorL2Norm() < 1e-6);
            std::cout << " gausEleminationSolver error: " << (a * xGauss - b).vectorL2Norm() << '\n';
            EXPECT_TRUE(nullspace.empty());
        }
        {
            V xCG = V(b.size());

            double rawPrevOut = 0.0;
            V residual;
            V p;
            dopt::cgSolverInit(rawPrevOut, residual, p, a, b, xCG);
            
            for (size_t i = 0; i < 6; ++i)
            {
                dopt::cgSolverIteration(xCG, residual, p, rawPrevOut, a, b, i, 1.0e-9);                
            }
            std::cout << " cgSolverIteration error: " << (a * xCG - b).vectorL2Norm() << '\n';
            EXPECT_TRUE( (a * xCG - b).vectorL2Norm() < 1e-6 );
        }
    }

    {
        size_t dim = 10;
        using V = dopt::VectorNDRaw_d;
        using M = dopt::MatrixNMD<V>;

        M a(dim, dim);
        V b(dim);
        V xSolution(dim);

        dopt::RandomGenRealLinear rnd(0.0, 200.0);
        
        for (size_t i = 0; i < dim; ++i)
        {
            for (size_t j = 0; j < dim; ++j)
                a.set(i, j, rnd.generateReal());
        }

        for (size_t i = 0; i < dim; ++i)
            xSolution.set(i, rnd.generateReal());

        a = a.getTranspose() * a + 0.1 * M::getIdentitySquareMatrix(dim);
        b = a * xSolution;

        EXPECT_TRUE(dopt::linear_solvers_helpers::isSystemFeasible(a, b, 1.0e-9));

        {
            auto xGaus = dopt::gausEleminationSolver(a, b);
            EXPECT_TRUE((xGaus - xSolution).vectorL2Norm() < 1.0e-3);
            std::cout << " gausEleminationSolver error: " << (a * xGaus - b).vectorL2Norm() << '\n';
        }

        {
            bool kJacobiIsApplied = dopt::linear_solvers_helpers::isMatrixHasDiagonalDominance(a);
            if (kJacobiIsApplied)
            {
                V xJacobi[2] = { V(b.size()), V(b.size()) };
                for (size_t i = 0; i < 1000; ++i)
                {
                    dopt::jacobiSolverIteration(xJacobi[(0 + i % 2) % 2], a, b, xJacobi[(1 + i % 2) % 2]);
                    //std::cout << " jacobiSolverIteration error: " << (a * xJacobi[0] - b).vectorL2Norm() << '\n';
                }
                EXPECT_TRUE((a * xJacobi[0] - b).vectorL2Norm() < 1e-6);
                std::cout << " jacobiSolverIteration error: " << (a * xJacobi[0] - b).vectorL2Norm() << '\n';
            }
            else
            {
                std::cout << " jacobiSolverIteration: " << " is not applied " << '\n';
            }
        }

        {
            V xSeidel = V(b.size());
            for (size_t i = 0; i < 15000; ++i)
            {
                dopt::seidelSolverIteration(xSeidel, a, b);
            }
            EXPECT_TRUE((a * xSeidel - b).vectorL2Norm() < 1e-3);
            std::cout << " seidelSolverIteration error: " << (a * xSeidel - b).vectorL2Norm() << '\n';
        }
        {
            std::vector<V> nullspace;
            V xGauss = dopt::gausEleminationSolver(a, b, &nullspace);
            EXPECT_TRUE((a* xGauss - b).vectorL2Norm() < 1e-3);
            std::cout << " gausEleminationSolver error: " << (a * xGauss - b).vectorL2Norm() << '\n';
            EXPECT_TRUE(nullspace.empty());
        }
        {
            V xCG = V(b.size());

            double rawPrevOut = 0.0;
            V residual;
            V p;
            dopt::cgSolverInit(rawPrevOut, residual, p, a, b, xCG);

            for (size_t i = 0; i < 15; ++i)
            {
                dopt::cgSolverIteration(xCG, residual, p, rawPrevOut, a, b, i, 1.0e-9);
                //std::cout << " cgSolverIteration error: " << (a * xCG - b).vectorL2Norm() << '\n';
            }

            EXPECT_TRUE((a * xCG - b).vectorL2Norm() < 1e-3);
            std::cout << " cgSolverIteration error: " << (a * xCG - b).vectorL2Norm() << '\n';
        }

        {
            using V = dopt::VectorNDRaw_d;
            using M = dopt::MatrixNMD<V>;

            M a(4, 4);
            V b(4);
            a.setRow(0, { 2.0, -1.0, 22.0, 13.0 });
            a.setRow(1, { 0.0, +1.0, 2.0, 12.0 });
            a.setRow(2, { 0.0, 0.0, 5.0, 10.0 });
            a.setRow(3, { 0.0, 0.0, 0.0, -6.0 });

            b[0] = 4.0;
            b[1] = 8.0;
            b[2] = 20.0;
            b[3] = 4.0;

            EXPECT_TRUE(dopt::isBackwardSubstitutionFine(a));
            EXPECT_FALSE(dopt::isForwardSubstitutionFine(a));
            EXPECT_FALSE(dopt::isBackwardSubstitutionFine(a.getTranspose()));
            EXPECT_TRUE(dopt::isForwardSubstitutionFine(a.getTranspose()));

            V xBackSub = dopt::backwardSubstitution(a, b);
            EXPECT_TRUE( (a * xBackSub - b).vectorL2Norm() < 1e-6 );

            V xForSub = dopt::forwardSubstitution(a.getTranspose(), b);
            EXPECT_TRUE((a.getTranspose() * xForSub - b).vectorL2Norm() < 1e-6);

            V xBackSubWithAtr = dopt::backwardSubstitutionWithATranspose(a.getTranspose(), b);
            EXPECT_TRUE((xBackSubWithAtr - xBackSub).vectorL2Norm() < 1e-6);

            V xForSubWithAtr = dopt::forwardSubstitutionWithATranspose(a.getTranspose().getTranspose(), b);
            EXPECT_TRUE((xForSub - xForSubWithAtr).vectorL2Norm() < 1e-6);


        }
    }
}
