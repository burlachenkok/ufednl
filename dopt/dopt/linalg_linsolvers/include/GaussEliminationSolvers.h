#pragma once

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_linsolvers/include/ElementaryMatTransforms.h"
#include "dopt/copylocal/include/Copier.h"

#include <vector>
#include <stddef.h>

namespace dopt
{
    /**
     * Solves a system of linear equations using Gaussian elimination.
     *
     * This function transforms the input matrix `aMatrix` into its reduced row-echelon form
     * and solves the linear equations defined by `aMatrix * x = bVector`. Optionally, it can
     * also compute a basis for the null space of `aMatrix`.
     *
     * @param aMatrix The matrix representing the coefficients of the system.
     * @param bVector The vector representing the right-hand side constants.
     * @param nullspace Optional pointer to a vector of vectors, which will be filled with the basis vectors of the null space of `aMatrix`. Default is nullptr.
     * @param allowModifyReferencedMatrices Flag that indicates whether the function is allowed to modify `aMatrix` and `bVector` directly. Default is false.
     * @param epsTolerance A tolerance value used to determine when a pivot element is considered zero. Default is the default constructor of `TTtem`.
     * @return A vector representing the solution to the system of equations.
     */
    template<class Mat, class Vec, class TTtem = typename Vec::TElementType>
    Vec gausEleminationSolver(Mat& aMatrix, Vec& bVector,
                              std::vector<Vec>* nullspace = nullptr,
                              bool allowModifyReferencedMatrices = false,
                              TTtem epsTolerance = TTtem())
    {
        // Initialization to allow in place modification of A and B.
        Mat aCopy = Mat();
        Vec bCopy = Vec();

        Vec* bPtr = nullptr;
        Mat* aPtr = nullptr;

        if (allowModifyReferencedMatrices)
        {
            bPtr = const_cast<Vec*>(&(bVector));
            aPtr = const_cast<Mat*>(&(aMatrix));
        }
        else
        {
            aCopy = aMatrix;
            bCopy = bVector;
            bPtr = &(bCopy);
            aPtr = &(aCopy);
        }
        Mat& a = *aPtr;
        Vec& b = *bPtr;

        /** Gaus elemination via transform A for AX=B in reduced row-echelon.
        * A is called "reduced row-echelon"  if:
        * - The first nonzero entry in every nonzero row is a one. The corresponding xj variables are called pivot variables.
        * - Every leading one is the only nonzero element in its column
        * - Any all-zero rows are grouped at the bottom of the matrix
        *  (https://www-old.math.gatech.edu/academic/courses/core/math2601/Web-notes/3.pdf)
        */

        typedef typename Vec::TElementType TItem;

        size_t curColumn = 0, curRow = 0;
        std::vector<size_t> pivotVariables;
        pivotVariables.reserve(a.columns());

        // Pivoting strategy - max in current rows
        for (; curRow < a.rows() && curColumn < a.columns();)
        {
            //------------------------------PIVOTING START-----------------------------------------//
            {
                TItem maxPivot = dopt::abs(a.get(curRow, curColumn));
                size_t maxPivotRow = curRow;

                for (size_t i = curRow + 1; i < a.rows(); ++i)
                {
                    auto potentialPivot = dopt::abs(a.get(i, curColumn));
                    if (potentialPivot > maxPivot)
                    {
                        maxPivotRow = i;
                        maxPivot = potentialPivot;
                    }
                }

                if (maxPivotRow != curRow)
                {
                    dopt::matSwapTwoRows(a, curRow, maxPivotRow);
                    dopt::CopyHelpers::swapDifferentObjects(b[curRow], b[maxPivotRow]);
                }

                if (maxPivot < epsTolerance)
                {
                    // SKIP CURRENT COLUMN
                    curColumn += 1;
                    continue;
                }
                //------------------------------PIVOTING END-----------------------------------------//

                b[curRow] /= a.get(curRow, curColumn);
                dopt::matMultiplyRowByVal(a, curRow, 1.0 / a.get(curRow, curColumn));

                for (size_t i = 0; i < a.rows(); ++i)
                {
                    if (i == curRow)
                        continue;
                    b[i] -= b[curRow] * a.get(i, curColumn);
                    dopt::matAppendKRowToIRow(a, i, curRow, -a.get(i, curColumn));
                    a.set(i, curColumn, TTtem());
                }

                pivotVariables.push_back(curColumn);
                ++curRow;
                ++curColumn;
            }
        }

        // Find some answer for AX = B
        Vec x = Vec(a.columns());

        {
            size_t pivotVariablesSize = pivotVariables.size();

            for (size_t i = 0; i < pivotVariablesSize; ++i)
                x[pivotVariables[i]] = b[i];
        }

        // Find linear space of solutions for AX = 0, i.e. find null space of A
        if (nullspace)
        {
            for (size_t j = 0; j < a.columns(); ++j)
            {
                //======================================================================//
                bool ignoreColumn = false;

                size_t pivotVariablesSize = pivotVariables.size();

                for (size_t i = 0; i < pivotVariablesSize; ++i)
                {
                    if (pivotVariables[i] == j)
                    {
                        ignoreColumn = true;
                        break;
                    }
                }

                if (ignoreColumn)
                    continue;
                //======================================================================//

                auto jColumn = a.getColumn(j);

                // create basis vector with all zeros
                Vec basisVector(a.columns());

                // setup to 1 variable correspond to this free (basis) variable
                basisVector[j] = TItem(1);

                for (size_t i = 0; i < pivotVariablesSize; ++i)
                {
                    basisVector[pivotVariables[i]] = -jColumn[i];
                }

                if (ignoreColumn)
                    continue;

                // Number of pivot-variables or basis variables or dependent variables is equal to r, rank(A)=r
                // Residual variables are free or not-dependent
                // FundamentalSolutionSystem in Russian literature is called basis for solve AX=0, i.e. it is null space.
                nullspace->push_back(basisVector);
            }
        }
        // To create basis for range of A it is possible to collect columns of A which corresponds to pivotVariables

        return x;
    }
}
