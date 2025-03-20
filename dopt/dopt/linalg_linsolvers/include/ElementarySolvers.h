#pragma once

#include "dopt/linalg_linsolvers/include/ElementaryMatTransforms.h"
#include "dopt/linalg_vectors/include/LightVectorND.h"
#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include <stddef.h>

namespace dopt
{
    /* Check that Backward Substitution can be used for solving Ax=b in x
    * @param a matrix of linear system
    * @remark Backward substitution solver only works for square upper triangular matrices of the system
    * @return true if backward substitution is feasible
    */
    template<class Mat>
    bool isBackwardSubstitutionFine(const Mat& a)
    {
        if (a.size() == 0)
            return false;
        
        if (!a.isSquareMatrix())
            return false;

        if (!a.isUpperTriangular())
            return false;

        return true;
    }

    /* Use Backward Substitution for solving Ax=b in x
    * @param a matrix of linear system
    * @param b right hand side part of linear system
    * @return solution vector x
    * @sa isBackwardSubstitutionFine
    * @sa backwardSubstitutionWithATranspose
    */
    template<class Mat, class Vec>
    Vec backwardSubstitution(const Mat& a, const Vec& b)
    {
        assert(b.size() > 0);
        assert(a.rows() == a.columns());

        // Convex Optimization Book by S.Boyd, p. 665
        Vec x = Vec::getUninitializedVector(b.size());
        int aRowsAndCols = a.rows();

        for (int i = aRowsAndCols - 1; i >= 0; --i)
        {
            auto accum = b.get(i);
            
            for (int j = i + 1; j < aRowsAndCols; ++j)
            {
                accum -= a.get(i, j) * x.get(j);
            }

            x.set(i, accum / a.get(i, i));
        }

        return x;
    }

    /* Use Backward Substitution for solving Ax=b in x. But A for purpose of compute optimization is provided in a transposed form.
    * @param atr matrix of linear system A in a transposed form
    * @param b right hand side part of linear system
    * @return solution vector x
    * @sa isBackwardSubstitutionFine
    * @sa backwardSubstitution
    */
    template<class Mat, class Vec>
    Vec backwardSubstitutionWithATranspose(const Mat& atr, const Vec& b)
    {
        assert(b.size() > 0);
        assert(atr.rows() == atr.columns());
        
        // Convex Optimization Book by S.Boyd, p. 665
        Vec x = Vec::getUninitializedVector(b.size());

        const size_t aRowsAndCols = atr.rows();

        // Main Case
        const size_t LDA = atr.LDA;
        const size_t LDA_plus_one = LDA + 1;
        
        dopt::LightVectorND<Vec> x_i_plus_one(x.data() + aRowsAndCols, 0);
        
        // Setup pointer to [aRowsAndCols - 1, aRowsAndCols] in not transpose matrix
        dopt::LightVectorND<Vec> atr_column_i_plus_one(const_cast<Mat&>(atr).matrixByCols.data() +
                                                       atr.getFlattenIndexFromPosition(aRowsAndCols,
                                                                                       aRowsAndCols - 1),
                                                       0);
            
        for (int i = aRowsAndCols - 1; i >= 0; --i, 
                                               x_i_plus_one.componentsCount++,
                                               atr_column_i_plus_one.componentsCount++,
                                               x_i_plus_one.components--, 
                                               atr_column_i_plus_one.components -= LDA_plus_one)
        {

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD
            typedef typename dopt::VectorSimdTraits<typename Vec::TElementType, dopt::cpu_extension>::VecType VecType;

            constexpr size_t kVecBatchSizeInBytes = dopt::getVecBatchSize<VecType>() * 
                                                    sizeof(typename Vec::TElementType);

            bool isAligned = isAddressAligned<kVecBatchSizeInBytes> (x_i_plus_one.components) &&
                             isAddressAligned<kVecBatchSizeInBytes> (atr_column_i_plus_one.components);

            if (isAligned)
            {
                auto accum = b.get(i) - atr_column_i_plus_one.dotProductForAlignedMemory(x_i_plus_one);
                x.set(i, accum / atr.get(i, i));
            }
            else
            {
                auto accum = b.get(i) - (atr_column_i_plus_one & x_i_plus_one);
                x.set(i, accum / atr.get(i, i));
            }
#else
            // correspond to "i" row and "i+1" column items in direction from LEFT to RIGHT 
            // and pick all elements to the end of row
            auto accum = b.get(i) - (atr_column_i_plus_one & x_i_plus_one);
            x.set(i, accum / atr.get(i, i));
#endif
        }
        return x;
    }

    /* Check that Forward Substitution can be used for solving Ax=b in x
    * @param a matrix of linear system
    * @remark Forward substitution solver only works for square lower triangular matrices of the system
    * @return true if backward substitution is feasible
    */
    template<class Mat>
    bool isForwardSubstitutionFine(const Mat& a)
    {
        if (a.size() == 0)
            return false;
        
        if (!a.isSquareMatrix())
            return false;

        if (!a.isLowerTriangular())
            return false;

        return true;
    }

    /* Use Forward Substitution for solving Ax=b in x
    * @param a matrix of linear system
    * @param b right hand side part of linear system
    * @return solution vector x
    * @sa isForwardSubstitutionFine
    * @sa forwardSubstitutionWithATranspose
    */
    template<class Mat, class Vec>
    Vec forwardSubstitution(const Mat& a, const Vec& b)
    {
        assert(b.size() > 0);
        assert(a.rows() == a.columns());
        
        // Convex Optimization Book by S.Boyd, p. 665
        Vec x = Vec::getUninitializedVector(b.size());

        size_t aRows = a.rows();

        for (size_t i = 0; i < aRows; ++i)
        {
            auto accum = b.get(i);
            for (size_t j = 0; j < i; ++j)
                accum -= a.get(i, j) * x.get(j);
            x.set(i, accum / a.get(i, i));
        }

        return x;
    }

    /* Use Forward Substitution for solving Ax=b in x
    * @param atr matrix of linear system A presented in transposed way
    * @param b right hand side part of linear system
    * @return solution vector x
    * @sa isForwardSubstitutionFine
    * @sa forwardSubstitution
    */
    template<class Mat, class Vec>
    Vec forwardSubstitutionWithATranspose(const Mat& atr, const Vec& b)
    {
        assert(b.size() > 0);
        assert(atr.rows() == atr.columns());
        
        // Convex Optimization Book by S.Boyd, p. 665
        Vec x = Vec::getUninitializedVector(b.size());

        size_t aRowsAndCols = atr.rows();

        // x_0i vector
        dopt::LightVectorND<Vec> x_0i(&x[0], 0);
        dopt::LightVectorND<Vec> atr_column_i(const_cast<Mat&>(atr).matrixByCols.data(), 0);

        x.set(0, b.get(0) / atr.matrixByCols[0]);

        size_t LDA = atr.LDA;
        size_t diag_current = (LDA + 1);

        for ( size_t i = 1; i < aRowsAndCols; ++i, diag_current += (LDA + 1) )
        {
            // Update number of components in a "light vector" and in "x"
            x_0i.componentsCount = i;
            atr_column_i.componentsCount = i;

            // No update in x_0i.components -- same
            // Update in atr_column_i.components to point to a next Column in atr
            atr_column_i.components += LDA;

            // compute accum
            auto accum = b.get(i) - (atr_column_i.dotProductForAlignedMemory(x_0i));

            // update x
            x.set(i, accum / atr.matrixByCols[diag_current]);
        }

        return x;
    }

    /* Check that Diagonal solver can be used for solving Ax=b in x
    * @param a matrix of linear system
    * @remark Diagonal substitution solver only works for square and diagonal matrices of the system
    * @return true if backward substitution is feasible
    */
    template<class Mat>
    bool isDiagonalSolverFine(const Mat& a)
    {
        if (!a.isSquareMatrix())
            return false;

        if (!a.isDiagonal())
            return false;

        return true;
    }

    /* Use Diagonal solver for solving Ax=b in x
    * @param a matrix of linear system
    * @param b right hand side part of linear system
    * @return solution vector x
    * @sa isForwardSubstitutionFine
    */
    template<class Mat, class Vec>
    Vec diagonalSolver(const Mat& a, const Vec& b)
    {
        Vec x = Vec(b.size());
        size_t aRows = a.rows();

        for (size_t i = 0; i < aRows; ++i)
            x.set(i, b.get(i) / a.get(i, i));

        return true;
    }

    /**
     * @brief Check that Sweep Solver can be used for solving Ax=b in x.
     * @param a The matrix of the linear system.
     * @remark Sweep solver only works for square three-diagonal matrices of the system.
     * @return true if sweep solver is feasible.
     */
    template<class Mat>
    bool isSweepSolverFine(const Mat& a)
    {
        if (!a.isSquareMatrix())
            return false;

        if (!a.isThreeDiagonal())
            return false;

        return true;
    }

    /**
     * Solves a tridiagonal system of linear equations using the sweep (Thomas) algorithm.
     *
     * @param a The coefficient matrix of the system, which must be tridiagonal.
     * @param b The right-hand side vector of the system.
     * @return A vector containing the solution to the system.
     */
    template<class Mat, class Vec>
    Vec sweepSolver(const Mat& a, const Vec& b)
    {
        Vec x = Vec(b.size());

        size_t aRows = a.rows();
        size_t aCols = a.columns();

        if (aRows == 0)
            return true;

        Vec alpha(aRows - 1);
        Vec beta(aRows);

        //========== Evaluate alpha, beta  ===========
        alpha[0] = a.get(0, 1) / a.get(0, 0);
        beta[0] = b.get(0) / a.get(0, 0);

        for (size_t i = 1; i < aRows - 1; ++i)
        {
            auto divider = a.get(i, i) - a.get(i, i - 1) * alpha.get(i - 1);
            alpha[i] = a.get(i, i + 1) / divider;
            beta[i]  = (b[i] - a.get(i, i - 1) * beta[i - 1]) / divider;
        }
        {
            size_t ii = aRows - 1;
            auto divider = (a.get(ii, ii) - a.get(ii, ii - 1) * alpha[ii - 1]);
            beta[ii] = (b[ii] - a.get(ii, ii - 1) * beta[ii - 1]) / divider;
        }
        //========== Evaluate alpha, beta ===========

        //========== Gauss Reverse Start ============
        {
            size_t j = aCols - 1;

            x[j] = beta[j];

            for (; j > 0;)
            {
                --j;
                x[j] = beta[j] - alpha[j] * x[j + 1];
            }
        }
        //========== Gauss Reverse End ==============

        return x;
    }

    namespace linear_solvers_helpers
    {
        /** Does it exist at least one solution for ax = b
        * @param a system matrix of linear equations ax = b
        * @param b right hand side for system of linear equations ax = b
        * @param epsTolerance numerical tolerance to define zero
        * @return true if system ax=b has at least one solution
        */
        template<class Mat, class Vec, class TItem = typename Vec::TElementType>
        bool isSystemFeasible(const Mat & a, const Vec & b, TItem epsTolerance)
        {
            if (b.vectorL2Norm() < epsTolerance)
                return true;

            Mat ab(a);
            ab.appendColumns(1);
            for (size_t i = 0; i < ab.rows(); ++i)
                ab.set(i, ab.columns() - 1, b.get(i));

            size_t abRank = findRankViaStepTransform(ab, epsTolerance);
            size_t aRank = findRankViaStepTransform(a, epsTolerance);

            // Use theorem of Kronecker(1823-1891), German - Kapelli(1855-1910)
            return abRank == aRank;
        }

        /**
         * Checks if a matrix has diagonal dominance.
         *
         * A matrix is diagonally dominant if, for every row of the matrix,
         * the absolute value of the diagonal entry in that row is larger
         * than or equal to the sum of the absolute values of all the other entries
         * in that row.
         *
         * @param[in] a The matrix to check for diagonal dominance.
         * @return true if the matrix has diagonal dominance, false otherwise.
         */
        template<class Mat>
        bool isMatrixHasDiagonalDominance(const Mat& a)
        {
            using TElementType = typename Mat::TElementType;
            size_t aRows = a.rows();

            for (size_t i = 0; i < aRows; ++i)
            {
                auto ai = a.getRow(i);

                TElementType aii_abs = a.get(i, i);

                if (aii_abs < TElementType(0))
                    aii_abs = -aii_abs;

                if (aii_abs < (ai.vectorL1Norm() - aii_abs) )
                {
                    return false;
                }
            }

            return true;
        }
    }
}
