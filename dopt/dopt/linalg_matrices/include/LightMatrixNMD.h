#pragma once

#include <initializer_list>
#include <sstream>
#include <iostream>

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    /** Sub matrix located in [i_start, i_end) X [j_start, j_end)
    */
    template <typename TMat>
    class LightMatrixNMD
    {
    public:
        typedef TMat TMatrix;                                  ///< Typedef for matrix row if accessed
        typedef typename TMatrix::MatrixRow MatrixRow;         ///< Typedef for matrix row if accessed
        typedef typename TMatrix::MatrixColumn MatrixColumn;   ///< Typedef for matrix column if accessed
        typedef typename TMatrix::TElementType TElementType;   ///< Typedef for element type

        const TMatrix& refMatrix;
        size_t i_start_; ///< Start row (inclusive)
        size_t i_end_;   ///< End row (exclusive)
        size_t j_start_; ///< Start column (inclusive)
        size_t j_end_;   ///< End column (exclusive)
        size_t rows_;    ///< Total number of rows (Derived quantity. But stored for not recompute.)
        size_t columns_; ///< Total number of columns (Derived quantity. But stored for not recompute.)

        LightMatrixNMD(const TMatrix& parentMatrix,
                       size_t i_start, size_t i_end,
                       size_t j_start, size_t j_end)
        : refMatrix(parentMatrix)
        , i_start_(i_start)
        , i_end_(i_end)
        , j_start_(j_start)
        , j_end_(j_end)
        , rows_(i_end - i_start)
        , columns_(j_end - j_start)
        {
        }

        /** Copy constructor
        */
        LightMatrixNMD(const LightMatrixNMD& rhs)
        : refMatrix(rhs.refMatrix)
        , i_start_(rhs.i_start_)
        , i_end_(rhs.i_end_)
        , j_start_(rhs.j_start_)
        , j_end_(rhs.j_end_)
        , rows_(rhs.rows_)
        , columns_(rhs.columns_)
        {}

        /** Debug print of the matrix content into standard output stream
        * @param variableName name of the variable used in debug outputting
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out,
                                   const char* variableName  = "x=",
                                   const char* itemDelimiter = ",", 
                                   const char* rowDelimiter  = ";\n") const
        {
            out << variableName << "=[";
            for (size_t i = 0; i < rows(); ++i)
            {
                if (i != 0)
                    out << rowDelimiter;
                for (size_t j = 0; j < columns(); ++j)
                {
                    if (j != 0)
                        out << itemDelimiter;
                    out << get(i, j);
                }
            }
            out << "]\n";
        }

        /** Dump all items of the matrix row by row into memory pointed by out
        * @param out pointer to memory into which all items of the matrix will be dumped
        */
        void dumpByRows(TElementType* out) const
        {
            return dumpByRows(out, 0, rows() - 1, 0, columns() - 1);
        }

        /** Dump items of the matrix row by row into memory pointed by out
        * @param out pointer to memory into which all items of the matrix will be dumped
        * @param startRow start row which will be considered for dumping
        * @param endRow end row which will be considered for dumping
        * @param startColumn start column which will be considered for dumping
        * @param endColumn end column row which will be considered for dumping
        */
        void dumpByRows(TElementType* out, size_t startRow, size_t endRow, size_t startColumn, size_t endColumn) const
        {
            for (size_t i = startRow; i <= endRow; ++i)
                for (size_t j = startColumn; j <= endColumn; ++j)
                    *(out++) = get(i, j);
        }

        /** Dump all items of the matrix column by colulmn into memory pointed by out
        * @param out pointer to memory into which all items of the matrix will be dumped
        */
        void dumpByCols(TElementType* out) const
        {
            return dumpByCols(out, 0, rows() - 1, 0, columns() - 1);
        }

        /** Dump items of the matrix column by colulmn into memory pointed by out
        * @param out pointer to memory into which all items of the matrix will be dumped
        * @param startRow start row which will be considered for dumping
        * @param endRow end row which will be considered for dumping
        * @param startColumn start column which will be considered for dumping
        * @param endColumn end column row which will be considered for dumping
        */
        void dumpByCols(TElementType* out, size_t startRow, size_t endRow, size_t startColumn, size_t endColumn) const
        {
            for (size_t j = startColumn; j <= endColumn; ++j)
                for (size_t i = startRow; i <= endRow; ++i)
                    *(out++) = get(i, j);
        }

        /** Get number of rows in the matrix
        * @return number of rows in the matrix
        */
        size_t rows() const {
            return rows_;
        }

        /** Get number of columns in the matrix
        * @return number of rows in the matrix
        */
        size_t columns() const {
            return columns_;
        }

        /** Get total number of items in the matrix
        * @return number of rows in the matrix
        */
        size_t size() const {
            return rows_ * columns_;
        }

        /** Sum of diagonal elements of the matrix
        * @return trace of the matrix which is of all diagonal elements
        * @remark Sum of diagonal elements it is equal to the Sum of all eigenvalues of the matrix
        */
        TElementType trace() const
        {
            if (!isSquareMatrix())
            {
                assert(!"TRACE IS ONLY DEFINED FOR SQUARE MATRIX");
                return TElementType();
            }

            TElementType res = TElementType();
            for (size_t i = 0; i < rows(); ++i) {
                res += get(i,i);
            }

            return res;
        }

        /** Return flag that matrix is row matrix
        * @return true if condition is true
        */
        bool isRowMatrix() const {
            return rows() == 1;
        }

        /** Return flag that matrix is column matrix
        * @return true if condition is true
        */
        bool isColumnMatrix() const {
            return columns() == 1;
        }

        /** Return flag that matrix is square matrix
        * @return true if condition is true
        */
        bool isSquareMatrix() const {
            return rows() == columns();
        }

        /** Return flag that matrix is diagonal
        * @return true if condition is true
        */
        bool isDiagonal() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < columns(); ++j)
                {
                    if (i == j)
                        continue;
                    else if (get(i,j) == nullItem)
                        continue;
                    else
                        return false;
                }
            }

            return true;
        }

        /** Return flag that matrix is three diagonal
        * @return true if condition is true
        */
        bool isThreeDiagonal() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < columns(); ++j) {
                    if (i == j)
                        continue;
                    else if (i + 1 == j)
                        continue;
                    else if (i == j + 1)
                        continue;
                    else if (get(i,j) == nullItem)
                        continue;
                    else
                        return false;
                }
            }
            return true;
        }

        /** Return flag that matrix is upper triangular
        * @return true if condition is true
        */
        bool isUpperTriangular() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if ( get(i,j) != nullItem )
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /** Return flag that matrix is lower triangular
        * @return true if condition is true
        */
        bool isLowerTriangular() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i + 1; j < columns(); ++j)
                {
                    if ( get(i,j) != nullItem )
                        return false;
                }
            }

            return true;
        }

        /** Return flag that matrix is zero matrix
        * @return true if condition is true
        */
        bool isZeroMatrix() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < columns(); ++j)
                {
                    if ( get(i,j) != nullItem )
                        return false;
                }
            }

            return true;
        }

        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        size_t nnz() const
        {
            size_t result = 0;
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < columns(); ++j)
                {
                    if (get(i,j) != nullItem)
                    {
                        result++;
                    }
                }
            }

            return result;
        }

        /** Return flag that matrix is symmetric
        * @return true if condition is true
        */
        bool isSymmetric() const
        {
            if (!isSquareMatrix())
                return false;

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i + 1; j < columns(); ++j) {
                    if (get(i,j) != get(j,i)) {
                        return false;
                    }
                }
            }

            return true;
        }

        /** Return flag that matrix is skew symmetric
        * @return true if condition is true
        */
        bool isSkewSymmetric() const
        {
            TElementType nullItem = TElementType();

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = i; j < columns(); ++j)
                {
                    if (get(i,j) == -get(j,i))
                        continue;
                    else
                        return false;
                }
            }

            return true;
        }

        /** Get specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j]
        */
        TElementType get(size_t i, size_t j) const {
            return refMatrix.get(i_start_ + i, j_start_ + j);
        }

        /** Get specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j]
        */
        TElementType& getRaw(size_t i, size_t j) {
            return refMatrix.getRaw(i_start_ + i, j_start_ + j);
        }
        /** Set specific item in the matrix locating in (i,j) to specific value
        * @param i number of row
        * @param j number of column
        * @param value setuped value
        * @return reference to this
        */
        LightMatrixNMD& set(size_t i, size_t j, TElementType value)
        {
            refMatrix.set(i + i_start_, j + j_start_, value);
            return *this;
        }

        /** Is item locating in (i,j) position in zero
        * @param i number of row
        * @param j number of column
        * @return true if it so
        */
        bool isNull(size_t i, size_t j) const {
            return refMatrix.isNull(i + i_start_, j + j_start_);
        }

        /** Check that *this matrix is not equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix not equal to rhs
        */
        bool operator != (const LightMatrixNMD& rhs) const {
            return !(*this == rhs);
        }

        /** Check that *this matrix is equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix is equal to rhs
        */
        bool operator == (const LightMatrixNMD& rhs) const
        {
            size_t sz = size();

            if (rhs.size() != sz)
                return false;

            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < columns(); ++j)
                {
                    const TElementType& itemA = get(i, j);
                    const TElementType& itemB = rhs.get(i, j);
                    if (itemA != itemB)
                        return false;
                }
            }
            return true;
        }

        size_t compress() {
            return refMatrix.compress();
        }

        LightMatrixNMD& operator += (const LightMatrixNMD& other)
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < columns(); ++j)
                {
                    TElementType& myItem = getRaw(i,j);
                    const TElementType& otherItem = other.get(i, j);
                    myItem += otherItem;
                }
            }

            return *this;
        }

        LightMatrixNMD& operator -= (const LightMatrixNMD& other)
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < columns(); ++j)
                {
                    TElementType& myItem = getRaw(i,j);
                    const TElementType& otherItem = other.get(i, j);
                    myItem -= otherItem;
                }
            }

            return *this;
        }

        template<typename TFactorType>
        LightMatrixNMD& operator *= (TFactorType factor)
        {
            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < columns(); ++j)
                {
                    getRaw(i, j) *= factor;
                }
            }
            return *this;
        }

        template<typename TFactorType>
        LightMatrixNMD& operator /= (TFactorType factor)
        {
            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < columns(); ++j)
                {
                    getRaw(i, j) /= factor;
                }
            }
            return *this;
        }
    };
}
