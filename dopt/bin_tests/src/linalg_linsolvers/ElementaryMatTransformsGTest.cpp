#include "dopt/linalg_linsolvers/include/ElementaryMatTransforms.h"
#include "dopt/linalg_matrices/include/factorization/CholeskyFactorization.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"

#include "gtest/gtest.h"

#include <math.h>

TEST(dopt, ElementaryMatTransformsGTest)
{
    using M = dopt::MatrixNMD<dopt::VectorNDRaw_i>;

    {
        M a(3, 4);
        a.setRow(0, {1, 2, 3, 4});
        a.setRow(1, {9, 2, 1, 1});
        a.setRow(2, {1, 1, 8, 0});

        M b = a;
        EXPECT_TRUE(a == b);
        b.setRow(0, { 1, 2, 3, 4 });
        b.setRow(1, {1, 1, 8, 0});
        b.setRow(2, { 9, 2, 1, 1 });
        EXPECT_FALSE(a == b);

        dopt::matSwapTwoRows(a, 1, 2);
        EXPECT_TRUE(a == b) << "Simple test of swap two rows";

        b.setRow(0, { 3, 2, 1, 4 });
        b.setRow(1, { 8, 1, 1, 0 });
        b.setRow(2, { 1, 2, 9, 1 });
        dopt::matSwapTwoColumns(a, 0, 2);
        EXPECT_TRUE(a == b) << "Simple test of swap two columns";

        a = b;
        b.setRow(0, { 3, 2, 1, 4 });
        b.setRow(1, { 8, 1, 1, 0 });
        b.setRow(2, { 1, 2, 9, 1 });

        b.setRow(0, { 3*2, 2*2, 1*2, 4*2 });
        b.setRow(1, { 8*3, 1*3, 1*3, 0*3 });
        b.setRow(2, { 1*4, 2*4, 9*4, 1*4 });

        EXPECT_FALSE(a == b) << "Simple test by scaling row";
        dopt::matMultiplyRowByVal(a, 0, 2);
        dopt::matMultiplyRowByVal(a, 1, 3);
        dopt::matMultiplyRowByVal(a, 2, 4);
        EXPECT_TRUE(a == b) << "Simple test of multiplyRowByValue";

        a.setRow(0, { 3, 2, 1, 4 });
        a.setRow(1, { 8, 1, 1, 0 });
        a.setRow(2, { 1, 2, 9, 1 });
        dopt::matAppendKRowToIRow(a, 1, 2, 3);
        b.setRow(0, { 3, 2, 1, 4 });
        b.setRow(1, { 8 + 1 * 3, 1 + 2 * 3, 1 + 9 * 3, 0 + 1 * 3 });
        b.setRow(2, { 1, 2, 9, 1 });
        EXPECT_TRUE(a == b) << "Check simple append k tow to i row";
    }
    {
        M a(3, 4);
        a.setRow(0, { 1, 2, 3, 4 });
        a.setRow(1, { 9, 2, 1, 1 });
        a.setRow(2, { 1, 1, 8, 0 });

        M b = a;
        EXPECT_TRUE(a == b);
        b.setRow(0, { 1, 3, 2, 4 });
        b.setRow(1, { 9, 1, 2, 1 });
        b.setRow(2, { 1, 8, 1, 0 });
        EXPECT_FALSE(a == b);

        dopt::matSwapTwoColumns(a, 1, 2);
        EXPECT_TRUE(a == b) << "Simple test of swap two column";

        b.setRow(0, { 2, 3, 1, 4 });
        b.setRow(1, { 2, 1, 9, 1 });
        b.setRow(2, { 1, 8, 1, 0 });
        dopt::matSwapTwoColumns(a, 0, 2);
        EXPECT_TRUE(a == b) << "Simple test of swap two columns";
        a = b;
        b.setRow(0, { 2, 3, 1*3, 4 });
        b.setRow(1, { 2, 1, 9*3, 1 });
        b.setRow(2, { 1, 8, 1*3, 0 });
        EXPECT_FALSE(a == b) << "Simple test by scaling row";
        dopt::matMultiplyColByVal(a, 2, 3);
        EXPECT_TRUE(a == b) << "Simple test of multiplyColumnByValue";

        a.setRow(0, { 3, 2, 1, 4 });
        a.setRow(1, { 8, 1, 1, 0 });
        a.setRow(2, { 1, 2, 9, 1 });
        dopt::matAppendKColToJCol(a, 2, 1, 4);
        b.setRow(0, { 3, 2, 1 + 2 * 4, 4});
        b.setRow(1, { 8, 1, 1 + 1 * 4, 0 });
        b.setRow(2, { 1, 2, 9 + 2 * 4, 1 });
        EXPECT_TRUE(a == b) << "Check simple append k column to j column";
    }
}
