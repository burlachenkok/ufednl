#pragma once

#include "dopt/copylocal/include/Copier.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include <stddef.h>

namespace dopt
{
    /** Swap two rows in matrix m in place
    * @param m matrix under which we perform rows swapping
    * @param iRow one row in the matrix
    * @param jRow second row in the matrix
    * @return true if all is ok
    */
    template<class Mat>
    inline bool matSwapTwoRows(Mat& m, size_t iRow, size_t jRow)
    {
        if (iRow >= m.rows() || jRow >= m.rows())
        {
            return false;
        }

        if (iRow == jRow)
            return true;

        size_t mColumns = m.columns();
        for (size_t j = 0; j < mColumns; ++j)
        {
            dopt::CopyHelpers::swapDifferentObjects(m.getRaw(iRow, j), m.getRaw(jRow, j));
        }
        return true;
    }

    /** Swap two column in matrix m in place
    * @param m matrix under which we perform columns swapping
    * @param iColumn first column in the matrix
    * @param jColumn second column in the matrix
    * @return true if all is ok
    */
    template<class Mat>
    inline bool matSwapTwoColumns(Mat& m, size_t iColumn, size_t jColumn)
    {
        if (iColumn >= m.columns() || jColumn >= m.columns())
        {
            return false;
        }

        if (iColumn == jColumn)
            return true;

        size_t nRows = m.rows();
        for (size_t i = 0; i < nRows; ++i)
        {
            dopt::CopyHelpers::swapDifferentObjects(m.getRaw(i, iColumn), m.getRaw(i, jColumn));
        }
        return true;
    }

    /** Multiply row iRow by value
    * @param m matrix under which we perform modifications
    * @param iRow number of row
    * @param value multiplier
    * @return true if all is ok
    */
    template<class Mat, class TValue>
    inline bool matMultiplyRowByVal(Mat& m, size_t iRow, TValue value)
    {
        if (iRow >= m.rows())
        {
            return false;
        }
        size_t mColumns = m.columns();
        for (size_t j = 0; j < mColumns; ++j)
        {
            m.getRaw(iRow,j) *= value;
        }
        return true;
    }

    /** Multiply column jColumn by value
    * @param m matrix under which we perform modifications
    * @param jColumn number of column
    * @param value multiplier
    * @return true if all is ok
    */
    template<class Mat, class TValue>
    inline bool matMultiplyColByVal(Mat& m, size_t jColumn, TValue value)
    {
        if (jColumn >= m.columns())
        {
            return false;
        }
        size_t nRows = m.rows();
        for (size_t i = 0; i < nRows; ++i)
        {
            m.getRaw(i, jColumn) *= value;
        }
        return true;
    }

    /** Append row "k" to row "i"
    * @param m matrix under which we perform modifications
    * @param iRow number of row "i"
    * @param kRow number of row "k"
    * @param kMultiplier multiplier for row "k"
    * @return true if all is ok
    */
    template<class Mat, class TValue>
    inline bool matAppendKRowToIRow(Mat& m, size_t iRow, size_t kRow, TValue kMultiplier)
    {
        if (iRow >= m.rows() || kRow >= m.rows())
        {
            return false;
        }
        size_t mColumns = m.columns();
        for (size_t j = 0; j < mColumns; ++j)
            m.getRaw(iRow, j) += m.getRaw(kRow, j) * kMultiplier;

        return true;
    }

    /** Append column "k" to column "j"
    * @param m matrix under which we perform modifications
    * @param jColumn number of column "i"
    * @param kColumn number of column "k"
    * @param kMultiplier multiplier for column "k"
    * @return true if all is ok
    */
    template<class Mat, class TValue>
    inline bool matAppendKColToJCol(Mat& m, size_t jColumn, size_t kColumn, TValue kMultiplier)
    {
        if (jColumn >= m.columns() || kColumn >= m.columns())
        {
            return false;
        }

        size_t mRows = m.rows();
        for (size_t r = 0; r < mRows; ++r)
            m.getRaw(r, jColumn) += m.getRaw(r, kColumn) * kMultiplier;
        return true;
    }

    template<class Mat, class TValue>
    inline size_t findRankViaStepTransform(const Mat& a, TValue epsTolerance)
    {
        size_t numOfRowSwaps = 0;
        Mat m = a;

        size_t curRow = 0;
        size_t curColumn = 0;

        for (; curRow < m.rows() && curColumn < m.columns();)
        {
            if (dopt::abs(m.get(curRow, curColumn)) > epsTolerance)
            {
                for (size_t i = curRow + 1; i < m.rows(); ++i)
                {
                    auto mulitplier = -m.get(i, curColumn) / m.get(curRow, curColumn);
                    matAppendKRowToIRow(m, i, curRow, mulitplier);
                    m.set(i, curColumn, TValue());
                }
                ++curRow;
                ++curColumn;
            }
            else
            {
                bool findRowWithNonZeroLead = false;
                size_t foundRow = 0;

                for (size_t i = curRow + 1; i < m.rows(); ++i)
                {
                    if (dopt::abs(m.get(i, curColumn)) > epsTolerance)
                    {
                        foundRow = i;
                        findRowWithNonZeroLead = true;
                        break;
                    }
                }

                if (findRowWithNonZeroLead)
                {
                    matSwapTwoRows(m, curRow, foundRow);
                    numOfRowSwaps++;
                }
                else
                {
                    curColumn += 1;
                }
            }
        }

        size_t notNullRows = 0;

        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = 0; j < m.columns(); ++j)
            {
                if (dopt::abs(m.get(i, j)) > epsTolerance)
                {
                    notNullRows += 1;
                    break;
                }
            }
        }

        return notNullRows;
    }
}
