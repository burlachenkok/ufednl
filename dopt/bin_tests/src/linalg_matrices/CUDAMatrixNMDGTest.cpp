#if DOPT_CUDA_SUPPORT

#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include "dopt/random/include/RandomGenRealLinear.h"
#include "dopt/timers/include/HighPrecisionTimer.h"
#include "dopt/system/include/FloatUtils.h"

#include "dopt/gpu_compute_support/include/linalg_matrices/MatrixNMD_CUDA.h"
#include "dopt/gpu_compute_support/include/linalg_vectors/VectorND_CUDA_Raw.h"

#include "gtest/gtest.h"

#include <stdint.h>

#include <iostream>
#include <math.h>

namespace
{
    constexpr double kPi  = 3.14159265359;
    constexpr double kTolEps = 1e-3;
}

template <template <class> class VecType>
void cudaMatrixNMDGTest()
{
    using namespace dopt;

    {
        auto m = MatrixNMD_CUDA<VecType<float>>::getIdentitySquareMatrix(4);
        EXPECT_TRUE(m.isDiagonal());
        EXPECT_TRUE(m.isSymmetric());
        EXPECT_TRUE(m.isOrthogonal(float(kTolEps)));
    }

    {
        MatrixNMD_CUDA<VecType<float>> m(5, 5);
        m.set(0,1, 0.9f);
        m.set(1,2, 0.36f);  m.set(1,3, 0.36f);  m.set(1,4, 0.18f);
        m.set(2,3, 0.90f);
        m.set(3,0, 0.90f);
        m.set(4,0, 0.47f); m.set(4,2, 0.47f);

        EXPECT_EQ(0, m.compress());
        EXPECT_EQ(5, m.rows());
        EXPECT_EQ(5, m.columns());

        VecType<float> v(5);
        v.set(0, 0.05f);
        v.set(1, 0.04f);
        v.set(2, 0.36f);
        v.set(3, 0.37f);
        v.set(4, 0.19f);

        EXPECT_FLOAT_EQ(0.9f, m.get(0, 1));
        EXPECT_FLOAT_EQ(float(), m.get(0, 0));

        {
            VecType<float> res = m * v;
            EXPECT_FLOAT_EQ(0.036f, res[0]);
            EXPECT_FLOAT_EQ(0.297f, res[1]);
            EXPECT_FLOAT_EQ(0.333f, res[2]);
            EXPECT_FLOAT_EQ(0.045f, res[3]);
            EXPECT_FLOAT_EQ(0.1927f, res[4]);
        }
        {
            VecType<float> res = m.matrixVectorMultiply(m, v);
            EXPECT_FLOAT_EQ(0.036f, res[0]);
            EXPECT_FLOAT_EQ(0.297f, res[1]);
            EXPECT_FLOAT_EQ(0.333f, res[2]);
            EXPECT_FLOAT_EQ(0.045f, res[3]);
            EXPECT_FLOAT_EQ(0.1927f, res[4]);
        }
    }

    {
       MatrixNMD_CUDA<VecType<int32_t>> x(2, 2);
        x.set(0,0, 1);  x.set(0,1, 0);
        x.set(1,0, 0);  x.set(1,1, 12);

        EXPECT_TRUE( (x - x) == MatrixNMD_CUDA<VecType<int32_t>>(2, 2));
        EXPECT_TRUE( (x + x) == 2 * x);

        EXPECT_TRUE(x.isDiagonal());
        x.compress();
        EXPECT_TRUE(x.isDiagonal());
        EXPECT_TRUE(x.isLowerTriangular());
        x.set(1,0, 10);
        EXPECT_FALSE(x.isDiagonal());
        EXPECT_TRUE(x.isLowerTriangular());
        EXPECT_FALSE(x.isUpperTriangular());
    }

    {
       MatrixNMD_CUDA<VecType<int32_t>> a(2, 2);
       EXPECT_TRUE(MatrixNMD_CUDA<VecType<int32_t>>::getDiagonalMatrix(2, 1).isDiagonal());
       MatrixNMD_CUDA<VecType<int32_t>> b(2, 2);
       EXPECT_TRUE(MatrixNMD_CUDA<VecType<int32_t>>::getDiagonalMatrix(2, 2).isDiagonal());

       a = MatrixNMD_CUDA<VecType<int32_t>>::getDiagonalMatrix(2);
       b = MatrixNMD_CUDA<VecType<int32_t>>::getDiagonalMatrix(2, 2);
       EXPECT_TRUE(a * b == b);

       {
          MatrixNMD_CUDA<VecType<int32_t>> a(2, 10);
          MatrixNMD_CUDA<VecType<int32_t>> b(10, 3);
          EXPECT_TRUE((a*b).rows() == 2);
          EXPECT_TRUE((a*b).columns() == 3);
       }
       EXPECT_TRUE(a + b == b + a);
       EXPECT_TRUE(a * 2 == b);
       b.set(1,0, 10);
       EXPECT_TRUE(a != b);

       {
           int32_t data[4] = {};
           (a + b).dumpByRows(data, 0, 1, 0, 1);
           EXPECT_TRUE(data[0] == 3 && data[1] == 0 && data[2] == 10 && data[3] == 3);
       }
       {
           int32_t data[4] = {};
            (a + b + b).dumpByRows(data, 0, 1, 0, 1);
            EXPECT_TRUE(data[0] == 5 && data[1] == 0 && data[2] == 20 && data[3] == 5);
        }

        {
            int32_t evaluated[4] = {};
            (a - b + b).dumpByRows(evaluated, 0, 1, 0, 1);
            int32_t aitems[4] = {};
            (a).dumpByRows(aitems, 0, 1, 0, 1);
            EXPECT_TRUE(aitems[0] == evaluated[0] && aitems[1] == evaluated[1] && aitems[2] == evaluated[2] && aitems[3] == evaluated[3]);
            EXPECT_TRUE(a - b + b == a) << "Create zero items in matrix. Check that it did not break internal logic";
            auto c = (a + b);
            EXPECT_TRUE(c.getTranspose().getTranspose() == a + b);
        }
    }

    {
       MatrixNMD_CUDA<VecType<int32_t>> a(3, 4);
       MatrixNMD_CUDA<VecType<int32_t>> b(3, 4);
       MatrixNMD_CUDA<VecType<int32_t>> c(3, 4);

       a.setRow(0, { 1, 2, 3, 0  });
       a.setRow(1, { 2, 1, -1, 1});
       a.setRow(2, { 0, 5, 2, 3  });

       b.setRow(0, { 0, 1, 3, 3 });
       b.setRow(1, { 4, 3, 0, 1 });
       b.setRow(2, { 2, 3, 1, -1 });

       c.setRow(0, { 1, 3, 6, 3 });
       c.setRow(1, { 6, 4, -1, 2 });
       c.setRow(2, { 2, 8, 3, 2 });
       VecType<int32_t> aCmp1 = { 2, 1, 5 };
       VecType<int32_t> aCmp0 = { 1, 2, 0 };

       EXPECT_TRUE(a.getColumn(1) == aCmp1);
       EXPECT_TRUE(a.getColumn(0) == aCmp0);

        {
           MatrixNMD_CUDA<VecType<int32_t>> aCopy = a;
           aCopy.setColumn(1, aCopy.getColumn(1));
           EXPECT_TRUE(aCopy == a);
        }

        EXPECT_TRUE(a + b == c);
        EXPECT_TRUE(a * 2 == 2 * a);
        EXPECT_TRUE(a * 2 / 2 == a);
        EXPECT_TRUE(a + (b + c) == (a + b) + c);
        EXPECT_TRUE((a - a) ==MatrixNMD_CUDA<VecType<int32_t>>::getZeroMatrix(3, 4));

        EXPECT_TRUE(a.getTranspose().rows() == a.columns());
        EXPECT_TRUE(a.getTranspose().columns() == a.rows());
    }

    {
       MatrixNMD_CUDA<VecType<int32_t>> a(2, 2);
       MatrixNMD_CUDA<VecType<int32_t>> b(2, 2);
       MatrixNMD_CUDA<VecType<int32_t>> c(2, 2);

       a.setRow(0, { 1, 2 });
       a.setRow(1, { 3, 4 });

       b.setRow(0, { -1, 0 });
       b.setRow(1, { 2, -3 });

       c.setRow(0, { 3, -6 });
       c.setRow(1, { 5, -12 });

       EXPECT_TRUE(a * b *MatrixNMD_CUDA<VecType<int32_t>>::getIdentitySquareMatrix(2) == c);
    }

    {
       MatrixNMD_CUDA<VecType<int32_t>> a(2, 4);
       MatrixNMD_CUDA<VecType<int32_t>> b(1, 4);
       MatrixNMD_CUDA<VecType<int32_t>> c(2, 1);

        a.setRow(0, { 1, 2, 3, 4 });
        a.setRow(1, { -1, 1, int32_t(-2), 2 });

        a = a * 2;
        a = 2 * a / 4;

        b.setRow(0, { 2, 0, 3, 1 });
        b = b.getTranspose();

        c.setRow(0, { 15 });
        c.setRow(1, { -6 });
        
        EXPECT_TRUE(a*b == c);
        EXPECT_FALSE(a.isSquareMatrix());
    }
    {
       MatrixNMD_CUDA<VecType<int32_t>> a(3, 3);
        a.setRow(0, { 1, 2, 3 });
        a.setRow(1, { 0, 1, 2 });
        a.setRow(2, { 0, 0, 0 });
        EXPECT_FALSE(a.isDiagonal());
        EXPECT_TRUE(a.isUpperTriangular());
        EXPECT_FALSE(a.isLowerTriangular());
        EXPECT_FALSE(a.isSymmetric());
        EXPECT_TRUE(a.getTranspose().isLowerTriangular());
        EXPECT_FALSE(a.isSymmetric());

        a.setRow(0, { 1, 0, 0 });
        a.setRow(1, { 0, 2, 0 });
        a.setRow(2, { 0, 0, 0 });
        EXPECT_TRUE(a.isDiagonal());
        EXPECT_TRUE(a.isUpperTriangular());
        EXPECT_TRUE(a.isLowerTriangular());
        EXPECT_TRUE(a.isSymmetric());
        a.appendColumns(2);
        EXPECT_TRUE(a.columns() == 5);
        EXPECT_TRUE(a.rows() == 3);
    }

    {
        EXPECT_TRUE((MatrixNMD_CUDA<VecType<double>>::getIdentitySquareMatrix(5)).isOrthogonal(kTolEps));
        EXPECT_FALSE((MatrixNMD_CUDA<VecType<double>>::getDiagonalMatrix(5, 2.0)).isOrthogonal(kTolEps));
        MatrixNMD_CUDA<VecType<double>> a(2, 2);
        double phi = kPi / 6;
        a.setRow(0, { cos(phi), -sin(phi) });
        a.setRow(1, { sin(phi), cos(phi) });
        EXPECT_TRUE(a.isOrthogonal(kTolEps));
        a.setRow(1, {a.get(0,0), a.get(0,1)});
        EXPECT_FALSE(a.isOrthogonal(kTolEps));
    }
    {
       MatrixNMD_CUDA<VecType<double>> a(3, 3);
        a.setRow(0, { 1, 2, 3 });
        a.setRow(1, { 2, 5, 9 });
        a.setRow(2, { 3, 4, 8 });
        //EXPECT_TRUE(a * (a^-1) == MatrixNMD_CUDA<double>::getIdentitySquareMatrix(3));
        //EXPECT_TRUE((a*a) * (a^-2) == MatrixNMD_CUDA<double>::getIdentitySquareMatrix(3));
        EXPECT_TRUE((a*a*a) == (a ^ 3));
    }

    {
       MatrixNMD_CUDA<VecType<float>> a(2, 2);
        a.setRow(0, { 0.0f, 0.0f});
        a.setRow(1, { 0.0f, 0.0f});
        EXPECT_TRUE(a.isZeroMatrix());
        EXPECT_TRUE(MatrixNMD_CUDA<VecType<float>>::getZeroMatrix(2, 3).isZeroMatrix());
    }

    {
        MatrixNMD_CUDA<VecType<float>> expected(2, 2);
        expected.setRow(0, { 1.0f*3.0f, 1.0f*4.0f});
        expected.setRow(1, { 2.0f*3.0f, 2.0f*4.0f});
        VecType<float> vec1 = { 1.0, 2.0 };
        VecType<float> vec2 = {3.0, 4.0};
        EXPECT_TRUE(expected == MatrixNMD_CUDA<VecType<float>>::outerProduct(vec1,vec2));
    }

    {
        MatrixNMD_CUDA<VecType<int32_t>> a(3, 3);
        a.setRow(0, { 1, 2, 3 });
        a.setRow(1, { 3, -1, 2 });
        a.setRow(2, { 10, 11, 12 });
        EXPECT_FALSE(a.isDiagonal());

        MatrixNMD_CUDA<VecType<int32_t>> d(3, 3);
        d.setRow(0, { 11, 0, 0 });
        d.setRow(1, { 0, 2, 0  });
        d.setRow(2, { 0, 0, 4  });
        EXPECT_TRUE(d.isDiagonal());
        EXPECT_EQ(3, d.nnz()); 

        d.zeroOutItems(12);
        EXPECT_EQ(0, d.nnz());
    }
    {
        MatrixNMD_CUDA<VecType<int32_t>> a(3, 3);
        a.setRow(0, { 11, 2, 3 });
        a.setRow(1, { 32, -1, 2 });
        a.setRow(2, { 10, 11, 12 });
        EXPECT_FALSE(a.isDiagonal());

        MatrixNMD_CUDA<VecType<int32_t>> d(3, 3);
        d.setRow(0, { 11, 0, 0 });
        d.setRow(1, { 0, 2, 0 });
        d.setRow(2, { 0, 0, 4 });

        VecType<int32_t> diag({ 11,2,4 });
        EXPECT_TRUE(d * a == MatrixNMD_CUDA<VecType<int32_t>>::multiplyDiagonalByDense(diag, a) );
        EXPECT_TRUE(a * d == MatrixNMD_CUDA<VecType<int32_t>>::multiplyDenseByDiagonal(a, diag));
    }
}

template<template <class> class VecType, bool kTestCorrectNess = true>
void cudaCheckForMatrixShape(size_t n, size_t m, size_t k)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    dopt::MatrixNMD_CUDA<VecType<double>> a(n, k);
    dopt::MatrixNMD_CUDA<VecType<double>> b(k, m);
    a.setAllRandomly(gen);
    b.setAllRandomly(gen);

    for (size_t j = 0; j < a.columns(); ++j)
    {
        EXPECT_TRUE(a.getColumn(j).vectorL2NormSquare() > 0.0);
    }

    dopt::MatrixNMD_CUDA<VecType<double>> c = a * b;

    EXPECT_TRUE(a.rows() == n);
    EXPECT_TRUE(b.columns() == m);
    ASSERT_TRUE(a.columns() == b.rows());

    dopt::MatrixNMD<dopt::VectorNDRaw<double>> aHost;
    a.store(aHost);

    dopt::MatrixNMD<dopt::VectorNDRaw<double>> bHost;
    b.store(bHost);

    dopt::MatrixNMD<dopt::VectorNDRaw<double>> cHost;
    c.store(cHost);

    if (kTestCorrectNess)
    {
        for (size_t i_c = 0; i_c < c.rows(); ++i_c)
        {
            for (size_t j_c = 0; j_c < c.columns(); ++j_c)
            {
                double cItem = cHost.get(i_c, j_c);

                for (size_t k = 0; k < b.rows(); ++k)
                {
                    cItem -= aHost.get(i_c, k) * bHost.get(k, j_c);
                }
                
                EXPECT_TRUE(fabs(cItem) < 1.0e-4);
                
                if (fabs(cItem) >= 1e-6)
                {
                    std::cout << "  Discrepancy with true value: " << fabs(cItem) << '\n';
                }
            }
        }
    }
}

TEST(dopt, CUDAMatrixNMDGTest)
{
    cudaMatrixNMDGTest<dopt::VectorND_CUDA_Raw>();
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(20, 40, 20);

    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(111, 909, 103);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(31, 909, 103);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(67, 30, 103);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(32, 32, 33);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(32, 33, 32);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(33, 32, 32);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw>(31, 23, 21);
}

TEST(dopt, CUDAMatrixNMDGTestIndicies)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(101, 23);
    a.setAllRandomly(gen);

    size_t flattering_index = 0;
    size_t flattering_index_start = 0;

    for (size_t j = 0; j < a.columns(); ++j, flattering_index_start += a.LDA)
    {
        flattering_index = flattering_index_start;

        for (size_t i = 0; i < a.rows(); ++i, ++flattering_index)
        {
            const size_t index = a.getFlattenIndexFromPosition(i, j);

            EXPECT_TRUE(flattering_index == index);

            size_t iOut = 0;
            size_t jOut = 0;
            a.getPositionFromFlatternIndex(iOut, jOut, index);
            
            EXPECT_TRUE(iOut == i);
            EXPECT_TRUE(jOut == j);

            size_t indexTranspose = a.getFlattenIndexFromPosition(jOut, iOut);
            
            size_t LDA = a.LDA;
            EXPECT_TRUE(indexTranspose == a.getTranspoedIndexFromFlatternIndex(index));

            if (i == j)
            {
                EXPECT_TRUE(indexTranspose == index);
            }
            else
            {
                EXPECT_FALSE(indexTranspose == index);
            }

            if (j >= i)
            {
                EXPECT_TRUE(a.isIndexFromUpperTriangularPart(index));
            }
            else
            {
                EXPECT_FALSE(a.isIndexFromUpperTriangularPart(index));
            }
        }
    }
}
TEST(dopt, CUDAMatrixNMDGPerf)
{
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw, false>(500, 500, 500);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw, false>(500, 500, 100);
    cudaCheckForMatrixShape<dopt::VectorND_CUDA_Raw, false>(100, 300, 200);
}

TEST(dopt, CUDAMatrixNMDGTestSpecialFunctions)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(101, 23);
        a.setAllRandomly(gen);

        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> b(101, 23);
        b.setAllRandomly(gen);

        double l2NormOfDifference = 0.0;
        auto diff_1 = a - b;
        auto diff_2 = dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d>::computeDifferenceAndEvalL2Norm(a, b, l2NormOfDifference);

        auto err_1 = (diff_1 - diff_2).frobeniusNorm();
        EXPECT_TRUE(err_1 < kTolEps);

        auto err_2 = fabs(diff_1.frobeniusNorm() - l2NormOfDifference);
        EXPECT_TRUE(err_2 < kTolEps);
    }
    
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_f> a(101, 23);
        a.setAllRandomly(gen);

        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_f> b(101, 23);
        b.setAllRandomly(gen);

        float l2NormOfDifference = 0.0;
        auto diff_1 = a - b;
        auto diff_2 = dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_f>::computeDifferenceAndEvalL2Norm(a, b, l2NormOfDifference);

        auto err_1 = (diff_1 - diff_2).frobeniusNorm();
        EXPECT_TRUE(err_1 < kTolEps);

        auto err_2 = fabs(diff_1.frobeniusNorm() - l2NormOfDifference);
        EXPECT_TRUE(err_2 < kTolEps);
    }
}

template<class Vec>
void cudaCheckMatrixVectorAcceleratedOps()
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100};
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105};

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<Vec> a(rows[i], cols[i]);
        a.setAllRandomly(gen);

        auto aTrNaive = a.getTransposeNaive();
        auto aTr = a.getTranspose();

        EXPECT_TRUE(aTrNaive.rows() == aTr.rows());
        EXPECT_TRUE(aTrNaive.columns() == aTr.columns());
        EXPECT_TRUE((aTrNaive - aTr).frobeniusNorm() == 0.0);

        Vec x(a.columns());
        x.setAllRandomly(gen);

        {
            Vec xres = a.matrixVectorMultiply(a, x);

            dopt::VectorNDRaw<typename Vec::TElementType> xresHost;
            xres.store(xresHost);
            
            EXPECT_TRUE(xres.size() == a.rows());

            for (size_t i = 0; i < a.rows(); ++i)
            {
                typename Vec::TElementType xres_i = a.getRow(i) & x;
                EXPECT_TRUE(fabs(xresHost[i] - xres_i) < 0.1 * fabs(xres[i]));
            }

            Vec xres_two = a.matrixVectorMultiply(a, x, 0.0, Vec::eye(a.rows()));
            EXPECT_TRUE((xres_two - xres).vectorL2Norm() < 0.1);
        }

        {
            //aTr.dbgPrintInMatlabStyle(std::cout, "a");
            //x.dbgPrintInMatlabStyle(std::cout, "x");
            Vec xres = a.matrixVectorMultiplyWithPreTranspose(aTr, x);
            //xres.dbgPrintInMatlabStyle(std::cout, "xres");
            EXPECT_TRUE(xres.size() == a.rows());

            for (size_t i = 0; i < a.rows(); ++i)
            {
                typename Vec::TElementType xres_i = a.getRow(i) & x;
                EXPECT_TRUE(fabs(xres[i] - xres_i) < 0.1 * fabs(xres[i]));
            }

            Vec xres_two = a.matrixVectorMultiplyWithPreTranspose(aTr, x, 0.0, Vec::eye(a.rows()));
            EXPECT_TRUE((xres_two - xres).vectorL2Norm() < 0.1);
        }

        Vec y(a.rows());
        y.setAllRandomly(gen);
        {
            //a.dbgPrintInMatlabStyle(std::cout, "a");
            //x.dbgPrintInMatlabStyle(std::cout, "x");

            typename Vec::TElementType beta = 0.0;
            
            Vec xres = a.matrixVectorMultiply(a, x, beta, y);
            EXPECT_TRUE(xres.size() == a.rows());

            for (size_t i = 0; i < a.rows(); ++i)
            {
                typename Vec::TElementType xres_i_one = a.getRow(i) & x;
                typename Vec::TElementType xres_i_two = (beta * y).get(i);

                typename Vec::TElementType xres_i = xres_i_one + xres_i_two;
                typename Vec::TElementType mv_res_i = xres[i];

                EXPECT_TRUE(fabs(mv_res_i - xres_i) < 0.1 * fabs(xres_i));
            }
        }

        dopt::MatrixNMD_CUDA<Vec> a_sym = aTr * a;
        EXPECT_TRUE(a_sym.isSquareMatrix());
        
        auto fn1 = a_sym.frobeniusNorm();
        auto fn2 = a_sym.frobeniusNormForSymmetricMatrixFromUpPart();
        auto fn3 = std::sqrt(a_sym.frobeniusNormSquareForSymmetricMatrixFromUpPart());
        double fn4 = 0.0;

        for (size_t i = 0; i < a_sym.rows(); ++i)
        {
            for (size_t j = 0; j < a_sym.columns(); ++j)
            {
                fn4 += a_sym.get(i, j) * a_sym.get(i, j);
            }
        }

        fn4 = sqrt(fn4);

        EXPECT_TRUE(fabs(fn1 - fn2) < (fn1 * 0.001 + 1e-6));
        EXPECT_TRUE(fabs(fn2 - fn3) < (fn1 * 0.001 + 1e-6));
        EXPECT_TRUE(fabs(fn3 - fn4) < (fn1 * 0.001 + 1e-6));

    }
}

TEST(dopt, CUDAMatrixNMDGTestAccelearatedCPUOperations)
{
    cudaCheckMatrixVectorAcceleratedOps<dopt::VectorND_CUDA_Raw_d>();
    cudaCheckMatrixVectorAcceleratedOps<dopt::VectorND_CUDA_Raw_f>();
}

TEST(dopt, CUDAMatrixNMDTestGPerfTestMatrixVectorMultiply)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105, 202, 500, 759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    dopt::HighPrecisionTimer tm1;
    tm1.reset();
    tm1.pause();
    
    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);

        dopt::VectorND_CUDA_Raw_d x(a.columns());
        x.setAllRandomly(gen);

        dopt::VectorND_CUDA_Raw_d y(a.rows());
        y.setAllRandomly(gen);

        tm1.resume();
        
        for (size_t repeats = 0; repeats < 10; ++repeats)
        {
            dopt::VectorND_CUDA_Raw_d xres = a.matrixVectorMultiply(a, x);
            dopt::VectorND_CUDA_Raw_d xres_two = a.matrixVectorMultiply(a, x, 0.0, y);
        }
        
        tm1.pause();
    }
    double eclapsedTime = tm1.getTimeSec();
    std::cout << "Time for for matrix - vector multiply benchmark: " << eclapsedTime << '\n';
}

TEST(dopt, CUDAMatrixNMDTransposeGTest)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);
    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100, 200 };
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105, 202 };
    
    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);
        
        EXPECT_TRUE((a.getTransposeNaive() - a.getTranspose()).frobeniusNorm() < 1e-12);
        EXPECT_TRUE((a.getTransposeNaive() - a.getTransposeCO()).frobeniusNorm() < 1e-12);
    }
}


TEST(dopt, CUDAMatrixNMDTestGPerfTestMatrixTransposeNaive)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105, 202, 500, 759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    dopt::HighPrecisionTimer tm1;
    tm1.reset();
    tm1.pause();

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);
        tm1.resume();

        for (size_t repeats = 0; repeats < 10; ++repeats)
        {
            dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> aTr = a.getTransposeNaive();
        }

        tm1.pause();
    }
    
    double eclapsedTime = tm1.getTimeSec();
    std::cout << "Time for for matrix transpose benchmark: " << eclapsedTime << '\n';
}

TEST(dopt, CUDAMatrixNMDTestGPerfTestMatrixTransposeStd)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105, 202, 500, 759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    dopt::HighPrecisionTimer tm1;
    tm1.reset();
    tm1.pause();

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);
        tm1.resume();

        for (size_t repeats = 0; repeats < 10; ++repeats)
        {
            dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> aTr = a.getTranspose();
        }

        tm1.pause();
    }

    double eclapsedTime = tm1.getTimeSec();
    std::cout << "Time for for matrix transpose benchmark: " << eclapsedTime << '\n';
}


TEST(dopt, CUDAMatrixNMDTestGPerfTestMatrixTransposeCacheOblivious)
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 1, 3, 5, 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 1, 2, 10, 12, 16, 55, 105, 202, 500, 759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    dopt::HighPrecisionTimer tm1;
    
    tm1.reset();
    tm1.pause();

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);
        tm1.resume();

        for (size_t repeats = 0; repeats < 10; ++repeats)
        {
            dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> aTr = a.getTransposeCO();
        }

        tm1.pause();
    }
    double eclapsedTime = tm1.getTimeSec();
    std::cout << "Time for for matrix transpose benchmark: " << eclapsedTime << '\n';

    //======================================================================================
    
    tm1.reset();
    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);
        
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> b = a.getTranspose() * a;
        tm1.resume();

        volatile double norm = 0.0;
        
        for (size_t repeats = 0; repeats < 100; ++repeats)
        {
            norm = norm + b.frobeniusNorm();
        }
        tm1.pause();
    }
    double eclapsedTime_fr_norm = tm1.getTimeSec();
    std::cout << "Time for for frobenius norm benchmark: " << eclapsedTime_fr_norm << '\n';
    
    //======================================================================================
    
    tm1.reset();
    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> a(rows[i], cols[i]);
        a.setAllRandomly(gen);

        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d> b = a.getTranspose() * a;
        tm1.resume();

        volatile double norm = 0.0;

        for (size_t repeats = 0; repeats < 100; ++repeats)
        {
            norm = norm + b.frobeniusNormForSymmetricMatrixFromUpPart();
        }
        tm1.pause();
    }
    double eclapsedTime_fr_norm_sym = tm1.getTimeSec();
    std::cout << "Time for for frobenius norm [for symmetric matrix] benchmark: " << eclapsedTime_fr_norm_sym << '\n';
    
    //======================================================================================    
}

template<class VectorType>
void cuExecMatrixNMDTestGPerfTestDifferenceWithUpperPart()
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 10, 12, 16, 55, 105,  202,   759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    dopt::HighPrecisionTimer tm1;
    tm1.reset();
    tm1.pause();

    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<VectorType> a(rows[i], cols[i]);
        a.setAllRandomly(gen);

        dopt::MatrixNMD_CUDA<VectorType> b(rows[i], cols[i]);
        b.setAllRandomly(gen);
        tm1.resume();

        for (size_t repeats = 0; repeats < 10; ++repeats)
        {
            dopt::MatrixNMD_CUDA<VectorType> c = dopt::MatrixNMD_CUDA<VectorType>::computeDifferenceWithUpperTriangularPart(a, b);
        }
        tm1.pause();
    }

    double eclapsedTime = tm1.getTimeSec();
    std::cout << "Time for for computeDifferenceWithUpperTriangularPart benchmark: " << eclapsedTime << '\n';
    //=======================================================================================================================================//
}

TEST(dopt, CUDAMatrixNMDTestGPerfTestDifferenceWithUpperPart)
{
    cuExecMatrixNMDTestGPerfTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_d> ();
    cuExecMatrixNMDTestGPerfTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_f> ();
    cuExecMatrixNMDTestGPerfTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_ui> ();
}

template<class VectorType>
void cuExecuteMatrixNMDTestGTestDifferenceWithUpperPart()
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rows[] = { 10, 18, 50, 100, 200, 500, 1000 };
    size_t cols[] = { 10, 12, 16, 55, 105,  202,   759 };

    constexpr size_t kTests = sizeof(rows) / sizeof(rows[0]);

    for (size_t i = 0; i < kTests; ++i)
    {
            dopt::MatrixNMD_CUDA<VectorType> a(rows[i], cols[i]);
            a.setAllRandomly(gen);

            dopt::MatrixNMD_CUDA<VectorType> b(rows[i], cols[i]);
            b.setAllRandomly(gen);

            dopt::MatrixNMD_CUDA<VectorType> c = dopt::MatrixNMD_CUDA<VectorType>::computeDifferenceWithUpperTriangularPart(a, b);
            
            dopt::MatrixNMD<dopt::VectorNDRaw<typename VectorType::TElementType>> aHost, bHost, cHost;
            a.store(aHost);
            b.store(bHost);
            c.store(cHost);
            
            for (size_t i = 0; i < c.rows(); ++i)
            {
                for (size_t j = 0; j < c.columns(); ++j)
                {
                    if (i > j)
                    {
                        // Bottom part
                        EXPECT_TRUE(fabs(cHost.get(i, j)) <= kTolEps);
                    }
                    else
                    {
                        // Top part
                        EXPECT_TRUE(fabs(cHost.get(i, j) - (aHost.get(i, j) - bHost.get(i, j))) <= kTolEps);
                    }
                }
            }
    }
}


TEST(dopt, CUDAMatrixNMDTestGTestDifferenceWithUpperPart)
{
    cuExecuteMatrixNMDTestGTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_d>();
    cuExecuteMatrixNMDTestGTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_f>();
    cuExecuteMatrixNMDTestGTestDifferenceWithUpperPart<dopt::VectorND_CUDA_Raw_ui>();
}

template<class VectorType>
void cuExecuteMatrixNMDSymmetrizeGTest()
{
    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    size_t rowsAndCols[] = { 1, 3, 10, 18, 50, 100};

    constexpr size_t kTests = sizeof(rowsAndCols) / sizeof(rowsAndCols[0]);
    dopt::HighPrecisionTimer tm1;
    tm1.pause();
    
    dopt::HighPrecisionTimer tm2;
    tm2.pause();
    
    for (size_t i = 0; i < kTests; ++i)
    {
        dopt::MatrixNMD_CUDA<VectorType> a(rowsAndCols[i], rowsAndCols[i]);
        a.setAllRandomly(gen);
        EXPECT_TRUE(a.isSquareMatrix());

        if (rowsAndCols[i] > 1)
        {
            EXPECT_FALSE(a.isUpperTriangular());
        }
        for (size_t j = 0; j < a.columns(); ++j)
        {
            for (size_t i = j + 1; i < a.rows(); ++i)
            {
                a.set(i, j, 0);
            }
        }
        EXPECT_TRUE(a.isUpperTriangular());
        dopt::MatrixNMD_CUDA<VectorType> aSymmetrizeManually(rowsAndCols[i], rowsAndCols[i]);
        tm1.resume();
        for (size_t j = 0; j < a.columns(); ++j)
        {
            for (size_t i = 0; i <= j; ++i)
            {
                auto item = a.get(i, j);
                aSymmetrizeManually.set(i, j, item);
                aSymmetrizeManually.set(j, i, item);
            }
        }
        tm1.pause();

        dopt::MatrixNMD_CUDA<VectorType> aSymmetrizeAutom = a;
        
        tm2.resume();
        aSymmetrizeAutom.symmetrizeLowerTriangInPlace();
        tm2.pause();
        
        auto descripancy = (aSymmetrizeAutom - aSymmetrizeManually).frobeniusNorm();
        
        EXPECT_TRUE(descripancy < kTolEps);
    }    
    
    std::cout << " Spend time for manual symmetrization: " << tm1.getTimeSec() << " sec" << '\n';
    std::cout << " Spend time for automatic symmetrization: " << tm2.getTimeSec() << " sec" << '\n';
}

TEST(dopt, CUDAMatrixNMDSymmetrizeGTest)
{
    cuExecuteMatrixNMDSymmetrizeGTest<dopt::VectorND_CUDA_Raw_d>();
    cuExecuteMatrixNMDSymmetrizeGTest<dopt::VectorND_CUDA_Raw_f>();    
}

template <class Mat, size_t dim, size_t kIterations = 10>
void cuMatrixNMDTestNaturalCompressionTest()
{
    dopt::RandomGenRealLinear gen_compressor;
    gen_compressor.setSeed(908);

    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    Mat a(dim, dim);
    a.setAllRandomly(gen);
    a.symmetrizeLowerTriangInPlace();

    double variance_est = 0.0;
    Mat aAvg = a;
    aAvg.setAllToDefault();

    for (size_t i = 0; i < kIterations; ++i)
    {
        Mat aTest = a;
        aTest.applyNaturalCompressor();
        
        {
            for (size_t j = 0; j < aTest.columns(); ++j)
            {
                for (size_t i = 0; i < aTest.rows(); ++i)
                {
                    auto item = aTest.get(i, j);
                    EXPECT_TRUE( (dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item).components.mantissa == 0) );
                }
            }
        }
        
        // aTest.symmetrizeLowerTriangInPlace();

        aAvg += aTest;

        auto diff = (aTest - a).frobeniusNorm();
        variance_est += diff * diff;
    }

    aAvg /= kIterations;
    variance_est /= kIterations;

    std::cout << "  Natural Compressor / disrepancy with input in expectation [abs]: " << (aAvg - a).frobeniusNorm() << "\n";
    std::cout << "  Natural Compressor / disrepancy with input in expectation [rel]: " << (aAvg - a).frobeniusNorm() / a.frobeniusNorm() << "\n";
    std::cout << "  Natural Compressor W estimated / instance: " << variance_est / (a.frobeniusNorm() * a.frobeniusNorm()) << "\n";

    {
        Mat aTest = a;
        aTest.applyNaturalCompressorNaive();
        //aTest.symmetrizeLowerTriangInPlace();

        for (size_t j = 0; j < a.columns(); ++j)
        {
            for (size_t i = 0; i < a.rows(); ++i)
            {
                auto item = aTest.get(i, j);
                auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item);
                EXPECT_TRUE(pack.components.mantissa == 0);
            }
        }
    }
}

TEST(dopt, CUDAMatrixNMDTestNaturalCompression)
{
    cuMatrixNMDTestNaturalCompressionTest <dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d>, 100> ();
    cuMatrixNMDTestNaturalCompressionTest <dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_f>, 100>();    
}

template <class Mat, size_t dim, size_t kIterations = 10>
void cuMatrixNMDTestNaturalCompressionNaiveTest()
{
    dopt::RandomGenRealLinear gen_compressor;
    gen_compressor.setSeed(908);

    dopt::RandomGenRealLinear gen(-10.0, +11.0);
    gen.setSeed(123);

    Mat a(dim, dim);
    a.setAllRandomly(gen);
    a.symmetrizeLowerTriangInPlace();

    double variance_est = 0.0;
    Mat aAvg = a;
    aAvg.setAllToDefault();

    for (size_t i = 0; i < kIterations; ++i)
    {
        Mat aTest = a;
        aTest.applyNaturalCompressorNaive();
        // aTest.symmetrizeLowerTriangInPlace();

        aAvg += aTest;

        auto diff = (aTest - a).frobeniusNorm();
        variance_est += diff * diff;
    }

    aAvg /= kIterations;
    variance_est /= kIterations;

    std::cout << "  Natural Compressor / disrepancy with input in expectation [abs]: " << (aAvg - a).frobeniusNorm() << "\n";
    std::cout << "  Natural Compressor / disrepancy with input in expectation [rel]: " << (aAvg - a).frobeniusNorm() / a.frobeniusNorm() << "\n";
    std::cout << "  Natural Compressor W estimated / instance: " << variance_est / (a.frobeniusNorm() * a.frobeniusNorm()) << "\n";

    {
        Mat aTest = a;
        aTest.applyNaturalCompressorNaive();
        // aTest.symmetrizeLowerTriangInPlace();
        
        for (size_t j = 0; j < a.columns(); ++j)
        {
            for (size_t i = 0; i < a.rows(); ++i)
            {
                auto item = aTest.get(i, j);
                auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item);
                EXPECT_TRUE(pack.components.mantissa == 0);
            }
        }
    }    
}

TEST(dopt, CUDAMatrixNMDTestNaturalCompressionNaive)
{
    cuMatrixNMDTestNaturalCompressionNaiveTest <dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_d>, 100>();
    cuMatrixNMDTestNaturalCompressionNaiveTest <dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw_f>, 100>();
}
#endif
