#pragma once

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_vectors/include/LightVectorND.h"

#include <math.h>
#include <stddef.h>

namespace dopt
{
    namespace cholFactorization
    {
#if 0
        template<class Mat>
        bool choleskyFactorizationCrout(Mat& L, Mat& in, typename Mat::TElementType nullValue = typename Mat::TElementType())
        {
            typedef typename Mat::TElementType T;

            if (!in.isSquareMatrix()) [[unlikely]]
                return false; // Cholesky factorization is defined only for square matrices

            size_t d = in.rows();

            L = Mat(d, d);

            for (size_t j = 0; j < d; ++j)
            {
                T squares = T();

                for (size_t k = 0; k < j; ++k)
                {
                    squares += (L.get(j, k)) * (L.get(j, k));
                }

                {
                    T result = (in.get(j, j) - squares);
                    if (result < 0)
                    {
                        if (result > -nullValue)
                            result = T();
                        else
                            return false;
                    }
                    result = sqrt(result);
                    L.set(j, j, result);
                }

                for (size_t i = j + 1; i < d; ++i)
                {
                    squares = T();
                    for (size_t k = 0; k < j; ++k)
                    {
                        squares += (L.get(i, k)) * (L.get(j, k));
                    }

                    if (dopt::abs(L.get(j, j)) < nullValue)
                    {
                        return false;
                    }
                    else
                    {
                        T result = (T(1) / L.get(j, j)) * (in.get(i, j) - squares);
                        L.set(i, j, result);
                    }
                }
            }

            return true;
        }
#endif

#if 0
        /*  Cholesky - Banachiewicz algorithm.
        * @param out output matrix S. Such that "L * transpose(L) = in"
        * @param in input square symmetric matrix
        * @return true if Cholesky factorization exist.
        * @remark Th1. Every positive definite matrix A has a unique Cholesky decomposition
        * @remark Th2. The decomposition need not be unique when A is positive semi-definite
        * @remark Th3. If the matrix A is positive semi-definite and symmetric then it still has a decomposition of the form A = LL' if the diagonal entries of L are allowed to be zero
        * (http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf)
        * @remark Th4. Cholesky decomposition is extremely stable numerically, without any pivoting at all. (Also from "Numerical_Recipes 3rd edition" , p.100)
        */
        template<class Mat>
        bool choleskyFactorization(Mat& L, 
                                   const Mat& in, 
                                   typename Mat::TElementType nullValue = typename Mat::TElementType())
        {
            //return choleskyFactorizationCrout(L, in, nullValue);

            // Algorithm have been written based on algorithm mentioned in wiki https://en.wikipedia.org/wiki/Cholesky_decomposition

            // "Numerical_Recipes 3rd edition" , p.100
            // https://en.wikipedia.org/wiki/Cholesky_decomposition

            typedef typename Mat::TElementType T;

            if (!in.isSquareMatrix()) [[unlikely]]
                return false; // Cholesky factorization is defined only for square matrices

            L = Mat(in.rows(), in.columns());

            size_t inRows = in.rows();

            for (size_t i = 0; i < inRows; ++i)
            {
                for (size_t j = 0; j <= i; ++j)
                {
                    T squares = T();
                    for (size_t k = 0; k < j; ++k)
                        squares += (L.get(i,k)) * (L.get(j, k));

                    T result = T();
                    if (i == j)
                    {
                        result = (in.get(i,i) - squares);
                        if (result < 0)
                        {
                            if (result > -nullValue)
                                result = T();
                            else
                                return false;
                        }

                        result = sqrt(result);
                        L.set(i, j, result);
                    }
                    else
                    {
                        if (dopt::abs(L.get(j,j)) < nullValue)
                            return false;
                        else
                            result = (T(1) / L.get(j,j)) * (in.get(i,j) - squares);
                        L.set(i, j, result);
                    }
                }
            }

            return true;
        }
#endif

        /*  Adaptation of Cholesky - Banachiewicz algorithm.
        * @param out output matrix S. Such that "L * transpose(L) = in"
        * @param in input square symmetric matrix
        * @return true if Cholesky factorization exist.
        * @remark Th1. Every positive definite matrix A has a unique Cholesky decomposition
        * @remark Th2. The decomposition need not be unique when A is positive semi-definite
        * @remark Th3. If the matrix A is positive semi-definite and symmetric then it still has a decomposition of the form A = LL' if the diagonal entries of L are allowed to be zero
        * (http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf)
        * @remark Th4. Cholesky decomposition is extremely stable numerically, without any pivoting at all. (Also from "Numerical_Recipes 3rd edition" , p.100)
        * @tparam in_input_symmetric if true then all input elements will be fetched from poistion [j, i] where where j <= i, "j" is a number of row, and "i" is a number of column. i.e. from upper triangular part of matrix. If false - elements will be fetched from lower triangular part of matrix.
        */
        template<class Mat, 
                 bool fill_L_factor_in_lower_part = false, 
                 bool in_input_symmetric = false>
                 // typename Mat::TElementType nullValue = typename Mat::TElementType()
        bool choleskyFactorization(Mat& Ltr,
                                   const Mat& in,
                                   typename Mat::TElementType  extraDiagAdditiveItem, 
                                   typename Mat::TElementType nullValue = typename Mat::TElementType())
        {
            //return choleskyFactorizationCrout(L, in, nullValue);
            // Algorithm have been written based on algorithm mentioned in wiki https://en.wikipedia.org/wiki/Cholesky_decomposition
            // "Numerical_Recipes 3rd edition" , p.100
            // https://en.wikipedia.org/wiki/Cholesky_decomposition

            typedef typename Mat::TElementType T;
            typedef typename Mat::MatrixRow TVec;

            assert(in.isSquareMatrix()); // Cholesky factorization is defined only for square matrices

            size_t d = in.rows();
    
            if (Ltr.rows() != d || Ltr.columns() != d)
            {
                Ltr = Mat(d, d);
            }
            
            T* restrict_ext LtrData = Ltr.matrixByCols.data();
            const T* restrict_ext InData = in.matrixByCols.dataConst();

            const size_t Ltr_LDA = Ltr.LDA;
            const size_t Ltr_LDA_plus_one = Ltr_LDA + 1;
            const size_t in_LDA_plus_one = Ltr_LDA_plus_one;

            assert(in.LDA == Ltr.LDA);

            for (size_t i = 0, i_th_column_offset = 0, i_in_diag_offset = 0;
                i < d; 
                ++i, i_th_column_offset += Ltr_LDA, i_in_diag_offset += in_LDA_plus_one)
            {
                // dopt::LightVectorND<TVec> LtrColumnI(LtrData + Ltr.template getFlattenIndexFromColumn<0>(i), 0);
                dopt::LightVectorND<TVec> LtrColumnI(LtrData + i_th_column_offset, 0);

                // In the end of the loop: Ltr.set(j, i) [j+1 element of the i-th column]
                // => in a next loop: Ltr.set(j-1, i), Ltr.set(j-2, i) has been executed. [OK]
                // => Ltr column "i" with "j" elements can be safely used in the current loop iteration.
                // => Ltr column "j" with "j" elements can be safely used in the current loop iteration. 
                //    In fact all "d" element of this column "j" have been processed, because [j < i] always.
                for (size_t j = 0, j_th_column_offset = 0, j_diag_offset = 0; 
                    j < i; 
                    ++j, j_th_column_offset += Ltr_LDA, j_diag_offset += Ltr_LDA_plus_one)
                {
                    // dopt::LightVectorND<TVec> LtrColumnI(LtrData + Ltr.getFlattenIndexFromColumn<0>(i), j);
                    LtrColumnI.componentsCount = j;

                    // dopt::LightVectorND<TVec> LtrColumnJ(LtrData + Ltr.template getFlattenIndexFromColumn<0>(j), j);
                    dopt::LightVectorND<TVec> LtrColumnJ(LtrData + j_th_column_offset, j);

                    T squares = LtrColumnI.dotProductForAlignedMemory(LtrColumnJ);

                    // auto Ljj   = Ltr.get(j, j);
                    auto Ljj = *(LtrData + j_diag_offset);

                    assert(j != i);

//                    auto In_ij = (in_input_symmetric ? 
//                                  in.get(j, i) 
//                                 : 
//                                  in.get(i, j)
//                                 );

                    auto In_ij = (in_input_symmetric ? 
                                  *(InData + i_th_column_offset + j)
                                 : 
                                  *(InData + j_th_column_offset +i)
                                 );

                    if (dopt::abs(Ljj) < nullValue)
                    {
                        return false;
                    }
                    else
                    {
                        T result = (T(1) / Ljj) * (In_ij - squares);
                        *(LtrData + i_th_column_offset + j) = result;
                        // Ltr.set(j, i, result);

                        if (fill_L_factor_in_lower_part)
                        {
                            *(LtrData + j_th_column_offset + i) = result;
                            // Ltr.set(i, j, result);
                        }
                    }
                }

                {
                    // All first i elements of the i-th column has been setup
                    //dopt::LightVectorND<TVec> LtrColumnI(LtrData + Ltr.getFlattenIndexFromColumn<0>(i), i);
                    LtrColumnI.componentsCount = i;
                    T squares = LtrColumnI.vectorL2NormSquareForAlignedMemory();

                    // auto In_ii = in.get(i, i) + extraDiagAdditiveItem;
                    auto In_ii = *(InData + i_in_diag_offset) + extraDiagAdditiveItem;

                    T result = (In_ii - squares);

                    if (result < 0)
                    {
                        if (result > -nullValue)
                            result = T();
                        else
                            return false;
                    }
                    result = sqrt(result);
                    *(LtrData + i_in_diag_offset) = result;
                    // Ltr.set(i, i, result);
                }
            }

            return true;
        }
        

        /*  Cholesky Factorization has two equivalent forms of writing:
        *  1) L * L' = in (math literature)
        *  2) R' * R = in (matlab implementation)
        *  R = L'
        * @param L lower triangular matrix from cholesky factorization
        * @return R upper triangular matrix into which we performs conversion of L factor
        */
        template<class Mat>
        Mat LLT_to_RTT(const Mat& L) {
            return L.getTranspose();
        }
    }
}
