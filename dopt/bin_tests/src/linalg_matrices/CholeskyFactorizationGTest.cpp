#include "dopt/linalg_matrices/include/factorization/CholeskyFactorization.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"

#include "gtest/gtest.h"

TEST(dopt, CholeskyFactorizationGTest)
{
    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> testMatrix(3, 3);
        testMatrix.setRow(0, {1, 0, 0} );
        testMatrix.setRow(1, {4, 7, 0} );
        testMatrix.setRow(2, {1, 2, 5} );
        dopt::MatrixNMD <dopt::VectorNDStd_d> square = testMatrix * testMatrix.getTranspose();

        dopt::MatrixNMD <dopt::VectorNDStd_d> testMatrixMult(3, 3);
        testMatrixMult.setRow(0, { 1.0, 4.0, 1.0 });
        testMatrixMult.setRow(1, { 4.0, 65.0, 18.0 });
        testMatrixMult.setRow(2, { 1.0, 18.0, 30.0 });
        EXPECT_TRUE(testMatrixMult == square);
        
        dopt::MatrixNMD<dopt::VectorNDStd_d> testOut(3, 3);
        dopt::MatrixNMD<dopt::VectorNDStd_d> testOutTr(3, 3);
        
        bool test = dopt::cholFactorization::choleskyFactorization<decltype(testOut), false, false>(testOutTr, square, 0.0, 1e-6);
        testOut = testOutTr.getTranspose();
        
        EXPECT_TRUE(test) << "Test factorization. Eqialent to chol(testOut, 'lower') in Matlab";
        
        EXPECT_TRUE(testOut == testOutTr.getTranspose());
        EXPECT_FALSE(testOut == testOutTr);

        EXPECT_TRUE(testOut == testMatrix);
        EXPECT_TRUE(testOut.isLowerTriangular());
        EXPECT_FALSE(testOut.isUpperTriangular());

        auto testRFactor = dopt::cholFactorization::LLT_to_RTT(testOut);
        EXPECT_TRUE(testRFactor.isUpperTriangular());
        EXPECT_TRUE(testRFactor.getTranspose() == testMatrix);
    }
    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> test(2, 2);
        test.setRow(0, {1, 2});
        test.setRow(1,  { 4, 7 });
        
        dopt::MatrixNMD <dopt::VectorNDStd_d> testOutTr;
        EXPECT_FALSE(dopt::cholFactorization::choleskyFactorization(testOutTr, test, 0)) << "Try to feed not positive definite quadratic form";
        test.appendColumns(1);
        test.setRow(0, {1, 2, 3});
        test.setRow(1, { 1, 2, 3 });
        //EXPECT_FALSE(dopt::cholFactorization::choleskyFactorization(testOut, testOutTr, test)) << "Try to feed not square matrix";
    }
    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> test(2, 2);
        test.setRow(0, { 1, 2 });
        test.setRow(1, { 4, 7 });
        dopt::MatrixNMD <dopt::VectorNDStd_d> testOutTr;
        
        EXPECT_FALSE(dopt::cholFactorization::choleskyFactorization(testOutTr, test, 0)) << "Try to feed not positive definite quadratic form";
        test.appendColumns(1);
        test.setRow(0, { 1, 2, 3 });
        test.setRow(1, { 1, 2, 3 });
        //EXPECT_FALSE(dopt::cholFactorization::choleskyFactorization(testOut, testOutTr, test)) << "Try to feed not square matrix";
    }
    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> test(2, 2);
        dopt::MatrixNMD <dopt::VectorNDStd_d> testOutTr;
        test.setRow(0, { 1, 2 });
        test.setRow(1, { 2, 1 });
        EXPECT_FALSE(dopt::cholFactorization::choleskyFactorization(testOutTr, test, 0)) << "Try to feed not positive definite quadratic form";
    }
}
