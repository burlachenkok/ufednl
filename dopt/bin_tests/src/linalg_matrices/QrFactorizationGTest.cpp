#include "dopt/linalg_matrices/include/factorization/QrFactorization.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"

#include "gtest/gtest.h"

#include <math.h>

TEST(dopt, QrFactorizationGTest)
{
    dopt::MatrixNMD <dopt::VectorNDStd_d>* matNullPtr = (dopt::MatrixNMD <dopt::VectorNDStd_d>*)nullptr;

    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> a(3, 3);
        a.setRow(0, { 1, 0, -2 });
        a.setRow(1, { 2.5, 1, 0 });
        a.setRow(2, { 0, 3, 1 });
        dopt::MatrixNMD <dopt::VectorNDStd_d> q(10, 10), r(10, 10);

        dopt::qrFactorization::qrFactorizeFull( &q,
                                                matNullPtr,
                                                &r, 
                                                a);

        auto diff = q* r - a;
        EXPECT_TRUE( diff.frobeniusNorm() < 1e-6);

        EXPECT_TRUE(q.isOrthogonal(1e-6));
        EXPECT_TRUE(r.isUpperTriangular(1e-6));
        EXPECT_TRUE(r.get(0,0) > 0.0 && r.get(1, 1) > 0.0 && r.get(2,2) > 0.0);

        dopt::QrFactorizeInfo<decltype(q)> qrInfo(q, r);
        EXPECT_TRUE(qrInfo.getDimOfRangeA() == 3);
        EXPECT_TRUE(qrInfo.getOrthoBasisForRangeA().columns() == 3);
        dopt::MatrixNMD <dopt::VectorNDStd_d>::MatrixColumn testToMultiply = { 1.0, 2.0, 3.0 };
        EXPECT_TRUE( (qrInfo.evaluateAX(testToMultiply) - a * testToMultiply).vectorL2NormSquare() < 1e-4);
    }
    {
        dopt::MatrixNMD <dopt::VectorNDStd_d> a(3, 5);
        a.setRow(0, { 0.0, 1.0, 0.0, 0.0, 2.0 });
        a.setRow(1, { 0.0, 2.5, 1.0, 1.0, 5.0 });
        a.setRow(2, { 0.0, 0.0, 3.0, 3.0, 0.0 });
        dopt::MatrixNMD <dopt::VectorNDStd_d> q(10, 10), r(10, 10);
        dopt::qrFactorization::qrFactorizeFull(&q, matNullPtr, &r, a, 1e-6);
        
        auto diff = q * r - a;
        EXPECT_TRUE(diff.frobeniusNorm() < 1e-6);

        EXPECT_TRUE(q.isOrthogonal(1e-6));
        EXPECT_FALSE(q.isSquareMatrix());
        EXPECT_TRUE(q.columns() == 2);
        EXPECT_TRUE(r.rows() == 2);
        EXPECT_TRUE(r.columns() == 5);
        EXPECT_FALSE(r.isSquareMatrix());
        EXPECT_TRUE(r.isUpperTriangular(1e-6));
        EXPECT_TRUE(dopt::QrFactorizeInfo(q, r).getDimOfRangeA() == 2);

        dopt::QrFactorizeInfo qrInfo(q, r);
        dopt::MatrixNMD <dopt::VectorNDStd_d>::MatrixColumn testToMultiply = { 1.0, 2.0, 3.0, 2.0, 2.0 };

        EXPECT_TRUE((qrInfo.evaluateAX(testToMultiply) - a * testToMultiply).vectorL2NormSquare() < 1e-4);
        
        dopt::MatrixNMD <dopt::VectorNDStd_d> q1(10, 10), q2(10, 10), r1(10, 10);
        dopt::qrFactorization::qrFactorizeFull(&q1, &q2, &r1, a, 1e-6);

        EXPECT_TRUE(q1 == q);
        EXPECT_TRUE(r1 == r);
        EXPECT_TRUE(q1.rows() == q2.rows());
        EXPECT_TRUE(q1.columns() + q2.columns() == q1.rows());
        EXPECT_TRUE(q1.isOrthogonal(1e-6));
        EXPECT_TRUE(q2.isOrthogonal(1e-6));

        for (size_t i = 0; i < q1.columns(); ++i)
        {
            for (size_t j = 0; j < q2.columns(); ++j)
            {
                EXPECT_TRUE( fabs(q1.getColumn(i) & q2.getColumn(j)) <= 1e-6);
            }
        }

        dopt::MatrixNMD<dopt::VectorNDStd_d> nullspaceOfA = dopt::qrFactorization::orthoBasis4NullSpace(a, 1e-6);
        EXPECT_TRUE(nullspaceOfA.isOrthogonal(1e-6));
        EXPECT_TRUE(nullspaceOfA.columns() == 3);
    }
}
