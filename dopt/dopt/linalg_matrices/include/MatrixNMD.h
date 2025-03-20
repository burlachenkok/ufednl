/** @file
* Template for dense matrix implementation.
*/

#pragma once

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_vectors/include/LightVectorND.h"
#include "dopt/linalg_matrices/include/LightMatrixNMD.h"

#include "dopt/system/include/FloatUtils.h"

#include <initializer_list>
#include <sstream>
#include <iostream>

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    /** Dense matrix stored by columns
    * @tparam VectorType underlying type used to instantiate dense vector which will viewed column-wise matrix.
    * @note Underlying vector stores columns. However, there is no guarantee that there is no any padding.
    * @note Need stride is obtaining from computeLda()
    * @tparam VectorType underlying type for alignment
    */
    template <class VectorType>
    class MatrixNMD
    {
    public:
        /** Is the memory alighned or not for nearby columns by extra padding
        */
        static constexpr bool isIncludePaddingForAlignment()
        {
            return true;
        }

    private:
        /** Compute leading dimension for matrix. The number of rows in matrix A, B which include any memory padding for access efficiency.
        @param theRows number of rows matrix
        @return offset between nearby columns
        */
        constexpr static size_t computeLda(size_t theRows)
        {
            constexpr size_t kIncludePaddingForAlignment = isIncludePaddingForAlignment();

            if constexpr (kIncludePaddingForAlignment == false)
            {
                // Economic style in terms of allocated bytes for dense matrix
                // Benefits:
                //  -- Simplify computation of Frobenious norm of the matrix
                //  -- Simplify prepare memory buffers for transfer over the network
                return theRows;
            }
            else /*if constexpr (kIncludePaddingForAlignment == true)*/
            {
                
#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD
                // More refined value for alignment
                typedef typename dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
                constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
                constexpr size_t kCacheLizeSizeInBytes = kVecBatchSize * sizeof(TElementType);
#else
                // More cache friendly style with ability to use aligned store and load SIMD instructions
                #if __cpp_lib_hardware_interference_size >= 201703L
                    constexpr size_t kCacheLizeSizeInBytes = std::hardware_destructive_interference_size;
                #else
                    constexpr size_t kCacheLizeSizeInBytes = 64;
                #endif
                constexpr size_t kVecBatchSize = kCacheLizeSizeInBytes / sizeof(TElementType);
#endif
                static_assert(kCacheLizeSizeInBytes % sizeof(TElementType) == 0);
                static_assert(kCacheLizeSizeInBytes >= sizeof(TElementType));

                size_t LDAInItems = dopt::roundToNearestMultipleUp<kVecBatchSize>(theRows);
                assert(LDAInItems >= theRows);
                
                return LDAInItems;
            }
        }

    public:
        typedef typename VectorType::TElementType TElementType;                              ///< Typedef for element type

        typedef VectorType MatrixRow;          ///< Typedef for matrix row if accessed
        typedef VectorType MatrixColumn;       ///< Typedef for matrix column if accessed

        size_t rows_;                          ///< Number of rows in matrix [EQUAL TO LDA]
        size_t columns_;                       ///< Number of colums in matrix
        size_t LDA;                            ///< Leading dimension for matrix. The number of rows in matrix A, B which include any memory padding for access efficiency. [For easy of transfering and developement rows_ == LDA]
        VectorType matrixByCols;               ///< Components of the matrix stored by columns

        /** Construct empty dense matrix
        */
        MatrixNMD() noexcept
        : matrixByCols()
        , rows_(0)
        , columns_(0)
        , LDA(0)
        {
        }

        /** Construct dense matrix with specified size
        * @param theRows number of rows
        * @param theColumns number of columns
        * @remark all component are initialization with default ctor or zero.
        */
        MatrixNMD(size_t theRows, size_t theColumns) noexcept
        : rows_(theRows)
        , columns_(theColumns)
        , LDA(computeLda(theRows))
        , matrixByCols(LDA * theColumns)
        {
        }
        
        /** Copy constructor
        * @param rhs array from which copy is occurring
        */
        MatrixNMD(const MatrixNMD& rhs) noexcept
        : matrixByCols(rhs.matrixByCols)
        , rows_(rhs.rows_)
        , columns_(rhs.columns_)
        , LDA(rhs.LDA)
        {}

        /** Assignment operator
        * @param rhs expression from which we perform copy
        */
        MatrixNMD& operator = (const MatrixNMD& rhs) noexcept
        {
            matrixByCols = rhs.matrixByCols;
            rows_ = rhs.rows_;
            columns_ = rhs.columns_;
            LDA = rhs.LDA;
            return *this;
        }

        /** Assignment move operator
        * @param rhs xvalue expression from which we perform move
        */
        MatrixNMD& operator = (MatrixNMD&& rhs) noexcept
        {
            matrixByCols = std::move(rhs.matrixByCols);
            rows_ = rhs.rows_;
            columns_ = rhs.columns_;
            LDA = rhs.LDA;
            return *this;
        }

        /** Copy move operator
        */
        MatrixNMD(MatrixNMD&& rhs) noexcept
        : matrixByCols(std::move(rhs.matrixByCols))
        , rows_(rhs.rows_)
        , columns_(rhs.columns_)
        , LDA(rhs.LDA)
        {}

        /** Destructor
        */
        ~MatrixNMD() = default;

        /** Size in bytes for all matrix elements excluding padding
        * @return number of bytes to store all elements excluding padding
        */
        size_t sizeInBytesNoPadding() const
        {
            return rows_ * columns_ * sizeof(TElementType);
        }

        /** Debug print shape info
        * @param out text stream into which printing will have place to be
        * @param variableName name of the variable
        * @return string represented shape of the matrix
        */
        template<class text_out_steam>
        std::string dbgPrintShapeInfo(text_out_steam& out,
                                      const char* variableName = "xCpu") const
        {
            out << variableName << " has shape [ rows = " << rows() << ", " << " columns = " << columns() << "]";
            return out.str();
        }
        
        /** Debug print of the matrix content into standard output stream
        * @param out text stream into which printing will have place to be
        * @param variableName name of the variable used in debug outputting
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out,
                                   const char* variableName  = "xCpu=",
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

        /** Get total number of items in the matrix including any padding
        * @return number of items including padding
        */
        size_t size() const {
            return matrixByCols.size();
        }

        /** Sum of diagonal elements of the matrix
        * @return trace of the matrix which is of all diagonal elements
        * @remark Sum of diagonal elements it is equal to the Sum of all eigenvalues of the matrix
        */
        TElementType trace() const
        {
            if (!isSquareMatrix()) [[unlikely]]
            {
                assert(!"TRACE IS ONLY DEFINED FOR SQUARE MATRIX");
                return TElementType();
            }

            TElementType res = TElementType();
            
            size_t r = rows();

            for (size_t i = 0; i < r; ++i) {
                res += get(i, i);
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
            size_t r = rows();
            size_t c = columns();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j)
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
            
            size_t r = rows();
            size_t c = columns();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
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

        /** Append extra columns to the matrix
        * @param extraColumns number of extra columns to append
        * @return reference to this
        * @remark this call may invalidate previous pointer to internal storage
        */
        MatrixNMD& appendColumns(size_t extraColumns)
        {
            size_t oldSize = size();

            size_t newSize = oldSize + extraColumns * LDA;
            matrixByCols.resize(newSize);            
            columns_ += extraColumns;

            return *this;
        }

        /** Remove the last `extraColumns` fromthe matrix
        * @param extraColumns number of extra columns to append
        * @return reference to this
        */
        MatrixNMD& removeColumns(size_t extraColumns)
        {
            size_t sizeInItems = size();
            size_t toRemoveItems = extraColumns * LDA;

            if (sizeInItems < toRemoveItems) [[unlikely]]
            {
                assert(!"IT IS NOT ALLOWABLE TO MAKE MATRIX COMPLETELY WITH ZERO DIMNENSIONS");
                return *this;
            }
            matrixByCols.resize(sizeInItems - toRemoveItems);
            columns_ -= extraColumns;

            return *this;
        }

        /** Return flag that matrix is upper triangular
        * @return true if condition is true
        */
        bool isUpperTriangular(TElementType eps = TElementType()) const
        {
            TElementType nullItem = TElementType();

            size_t r = rows();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if ( dopt::abs(get(i,j)) > eps )
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
        bool isLowerTriangular(TElementType eps = TElementType()) const
        {
            TElementType nullItem = TElementType();
            
            size_t r = rows();
            size_t c = columns();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = i + 1; j < c; ++j)
                {
                    if (dopt::abs(get(i, j)) > eps)
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
            return matrixByCols.isNull();
        }
        
        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        size_t nnz() const
        {
            return matrixByCols.nnz();
        }

        /**  Frobenius norm of the matrix
        * @return Frobenius norm.
        * @remark It's equal to sqrt(trace(m'm))
        * @remark Some people use L2 name for it and it may confuse for people familiar with Applied Math.
        * @remark Norm can be not so useful in some application because it's similar for matrices with permuted elements.
        */
        TElementType frobeniusNorm() const {
            return matrixByCols.vectorL2Norm();
        }

        /** Evaluate Frobenius norm for symmetric matrix
        * @return Frobenius norm
        * @remark During evaluation of the Frobenius norm only the upper triangular part is touched
        */
        TElementType frobeniusNormForSymmetricMatrixFromUpPart() const {
            return matrixByCols.vectorL2Norm();
        }

        /** Evaluate Frobenius norm square for symmetric matrix
        * @return Frobenius norm square
        * @remark During evaluation of the Frobenius norm only the upper triangular part is touched
        */
        TElementType frobeniusNormSquareForSymmetricMatrixFromUpPart() const {
            return matrixByCols.vectorL2NormSquare();
        }

        /** Column of vectors are orthonormal, i.e. M^T * M = E.
        * Such linear mapping save "norm" property, and such transform sometimes is called Isometric.
        * Also it's not necessary that matrix is square
        * @param eps threshold for various numerical computations inside
        * @return true if matrix is orthogonal (columns are orthogonal) and
        */
        bool isOrthogonal(TElementType eps = TElementType()) const
        {
            // By definition
            // return getTranspose() * (*this) == getIdentitySquareMatrix(columns());

            TElementType nullItem = TElementType();

            for (size_t j1 = 0; j1 < columns(); ++j1)
            {
                auto j1Column = getColumn(j1);

                // Check that length of the column is equal to one
                if ( (j1Column & j1Column) < TElementType(1) - eps || (j1Column & j1Column) > TElementType(1) + eps)
                {
                    return false;
                }

                // Check orthogonality of other columns
                for (size_t j2 = j1 + 1; j2 < columns(); ++j2)
                {
                    auto j2Column = getColumn(j2);
                    if (dopt::abs(j1Column & j2Column) > eps)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /** Return flag that matrix is symmetric
        * @return true if condition is true
        */
        bool isSymmetric() const
        {
            if (!isSquareMatrix()) [[unlikely]]
                return false;

            TElementType nullItem = TElementType();
            
            size_t r = rows();
            size_t c = columns();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = i + 1; j < c; ++j) {
                    if (get(i,j) != get(j,i)) 
                    {
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

            size_t r = rows();
            size_t c = columns();

            for (size_t i = 0; i < r; ++i) {
                for (size_t j = i; j < c; ++j)
                {
                    if (get(i,j) == -get(j,i))
                        continue;
                    else
                        return false;
                }
            }

            return true;
        }

        /** Create or instantiate matrix contains only zeros with specified dimensions flag that matrix is skew symmetric
        * @param theRows number of rows in matrix
        * @param theColumns number of columns in matrix
        * @return result matrix
        */
        static MatrixNMD getZeroMatrix(size_t theRows, size_t theColumns)
        {
            MatrixNMD res(theRows, theColumns);
            return res;
        }

        /** Create or instantiate matrix contains non-zeros only in diagonal
        * @param theRows number of rows in matrix
        * @param theColumns number of columns in matrix
        * @param diagElement diagonal element which will be used for initialize
        * @return result matrix
        */
        static MatrixNMD getDiagonalMatrix(size_t theRowsAndColumns, TElementType diagElement = TElementType(1))
        {
            MatrixNMD res(theRowsAndColumns, theRowsAndColumns);
            for (size_t i = 0; i < theRowsAndColumns; ++i)
                res.set(i, i, diagElement);
            return res;
        }

        /** Create or instantiate  square matrix contains only zeros
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        static MatrixNMD getZeroSquareMatrix(size_t dim)
        {
            return MatrixNMD(dim, dim);
        }

        /** Create or instantiate  identity square matrix
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        static MatrixNMD getIdentitySquareMatrix(size_t dim)
        {
            return getDiagonalMatrix(dim, TElementType(1));
        }

        /** Get transpose of the give matrix. Naive implementation.
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        MatrixNMD getTransposeNaive() const
        {
            size_t r = rows();
            size_t c = columns();
            MatrixNMD res(c, r);

            for (size_t i = 0; i < r; ++i)
            {
                for (size_t j = 0; j < c; ++j)
                {
                    res.set(j, i, get(i, j));
                }
            }

            return res;
        }

        /** Naive straightforward tranposition of matrix block
         * @param B output matrix columnwise raw buffer
         * @param ldb stride between two column in B in elements
         * @param A input matrix columnwise raw buffer
         * @param lda stride between two columns in A in elements
         * @param rowInA number of rows in A, and number of columns in A
         * @param colsInA number of columns in A, and number of rows in B
         */
        static void internal_execute_transpose_blockwise(TElementType* restrict_ext B,
                                                         const size_t ldb,
                                                         const TElementType* restrict_ext A,
                                                         const size_t lda,
                                                         const size_t rowInA, const size_t colsInA)
        {
            for (size_t j = 0; j < colsInA; j++)
            {
                for (size_t i = 0; i < rowInA; i++)
                {
                    B[j + i * ldb] = A[i + j * lda];
                }
            }
        }
        
        /** Take in Cache Oblivious Style transposition of matrix [x, x + delx) x [y, y + dely).
        *   @remark X is horizontal axis (Columns of input)
        *   @remark Y is verical axis (Rows of input)
        * 
        *   @param x start of the block in X(column) axis 
        *   @param delx length of the block in X(column) axis
        *   @param y start of the block in Y(rows) axis
        *   @param dely length of the block in Y(rows) axis
        *   @param in_elements input matrix stored columnwise
        *   @param in_lda input leading dimension
        *   @param out_elements output matrix stored columnwise
        *   @param out_lda input leading dimension
        */
        static void internal_co_transposition(size_t x, size_t delx,size_t y, size_t dely,
                                              const TElementType* restrict_ext in_elements, size_t in_lda,
                                              TElementType* restrict_ext out_elements, size_t out_lda)
        {
            for (;;)
            {
                if (delx + dely < 64)
                {
                    internal_execute_transpose_blockwise(out_elements + out_lda * y + x, out_lda,
                                                         in_elements + in_lda * x + y, in_lda,
                                                         dely, delx);

                    return;
                }
                else
                {
                    // At least max(delx, dely) is 2. So we can divide and we garantee that there are no zero residuals.
                    if (delx >= dely)
                    {
                        // Recursive Case -- divide and conquer in OX. It has bigger extent.

                        size_t xmid = (delx >> 1);
                        internal_co_transposition(x, xmid, y, dely, in_elements, in_lda, out_elements, out_lda);
                        
                        // Tail Recursion Elimination: equivalent to
                        // internal_co_transposition(x + xmid, delx - xmid, y, dely, in_elements, in_lda, out_elements, out_lda);
                        x = x + xmid;
                        delx = delx - xmid;
                    }
                    else
                    {
                        // Recursive Case -- divide and conquer in OX. It has bigger extent.
                        size_t ymid = (dely >> 1);
                        internal_co_transposition(x, delx, y, ymid, in_elements, in_lda, out_elements, out_lda);
                        
                        // Tail Recursion Elimination
                        // internal_co_transposition(x, delx, y + ymid, dely - ymid, in_elements, in_lda, out_elements, out_lda);
                        y = y + ymid;
                        dely = dely - ymid;
                    }
                }
            }
        }       

        /** Symmetrize matrix in Cache Oblivious Style
        *   @remark X is horizontal axis (Columns of input)
        *   @remark Y is verical axis (Rows of input)
        *   @param x start of the block in X(column) axis
        *   @param delx length of the block in X(column) axis
        *   @param y start of the block in Y(rows) axis
        *   @param dely length of the block in Y(rows) axis
        *   @param in_out_elements input/output matrix stored columnwise
        *   @param in_out_lda input/output leading dimension
        */
        static void internal_co_symmetrize(size_t x, size_t delx, size_t y, size_t dely, TElementType* restrict_ext in_out_elements, size_t in_out_lda)
        {
            // -----X-------------------------------------------
            // | [x,y]             [x + delx - 1, y]
            // Y
            // | [x, y + dely -1]  [x + delx - 1, y + dely - 1]
            
            // [x,y] indices example with [4,4] matrix:
            //  00 10 20 30
            //  01 11 21 31
            //  02 12 22 32
            //  03 13 23 33

            for (;;)
            {
                size_t xRight = x + delx - 1;
                size_t yTop = y;

                if (yTop >= xRight)
                {
                    // Do nothing - the block below diagonal
                    // Prune
                    return;
                }
                
                if (delx + dely < 64)
                {
                    size_t yBottom = y + dely - 1;

                    if (x >= yBottom)
                    {
                        // Whole block above the diagonal - we can place it more effectively
                        //  x - column in input matrix
                        //  y - row in input matrix
                        internal_execute_transpose_blockwise(in_out_elements + in_out_lda * y + x, in_out_lda,
                                                             in_out_elements + in_out_lda * x + y, in_out_lda,
                                                             dely, delx);
                        return;
                    }           
#if 0
                    // Manual Transposition
                    else if (x >= yBottom)
                    {
                        // Whole block above the diagonal - we can place it more effectively
                        TElementType* in_elements_for_block = in_out_elements + in_out_lda * x + y;
                        TElementType* out_elements_for_block = in_out_elements + in_out_lda * y + x;
                        size_t c = delx;
                        size_t r = dely;
                        
                        for (size_t j = 0; j < c; j++)
                        {
                            for (size_t i = 0; i < r; i++)
                            {
                                out_elements_for_block[j + i * in_out_lda] = in_elements_for_block[i + j * in_out_lda];
                            }
                        }
                    }
#endif
                    else
                    {
                        // Whole block below the diagonal
                        
                        TElementType* in_elements_for_block = in_out_elements + in_out_lda * x + y;
                        TElementType* out_elements_for_block = in_out_elements + in_out_lda * y + x;
                        
                        size_t c = delx;
                        size_t r = dely;
                        size_t xGlobal = x;

                        for (size_t j = 0; j < c; ++j, ++xGlobal)
                        {
                            size_t yGlobal = y;

                            for (size_t i = 0; i < r && yGlobal < xGlobal; ++i, ++yGlobal)
                            {
                                out_elements_for_block[j + i * in_out_lda] = in_elements_for_block[i + j * in_out_lda];
                            }
                        }
                    }
                    return;
                }
                else
                {
                    // At least max(delx, dely) is 2. So we can divide and we garantee that there are no zero residuals.
                    // Recursive Case -- divide and conquer in OX. It has bigger extent.

                    if (delx >= dely)
                    {
                        size_t xmid = (delx >> 1);
                        internal_co_symmetrize(x, xmid, y, dely, in_out_elements, in_out_lda);

                        // Tail Recursion Elimination
                        // internal_co_transposition(x + xmid, delx - xmid, y, dely, in_elements, in_lda, out_elements, out_lda);
                        x = x + xmid;
                        delx = delx - xmid;
                    }
                    else
                    {
                        // Recursive Case -- divide and conquer in OX. It has bigger extent.
                        size_t ymid = (dely >> 1);
                        internal_co_symmetrize(x, delx, y, ymid, in_out_elements, in_out_lda);

                        // Tail Recursion Elimination
                        // internal_co_transposition(x, delx, y + ymid, dely - ymid, in_elements, in_lda, out_elements, out_lda);
                        y = y + ymid;
                        dely = dely - ymid;
                    }
                }
            }
        }
        /** Get transpose of the matrix. Cache Oblivious implementation.
        * @return result matrix
        * @remark This cache oblivious implementation of matrix transpose (http://users.cecs.anu.edu.au/~Alistair.Rendell/papers/coa.pdf).
        * @remark Algorithm has asymptotically optimal cache performance without knowledge about cache.
        */
        MatrixNMD getTransposeCO() const
        {
            size_t r = rows();
            size_t c = columns();
            MatrixNMD res(c, r);
            internal_co_transposition(0, c, 0, r, matrixByCols.dataConst(), LDA, res.matrixByCols.data(), res.LDA);

            return res;
        }

        /** Symmetrize matrix with using upper triangular part in place.
        *   Copy elements from upper triangular part excluding diagonal to lower triangular part.
        */
        MatrixNMD& symmetrizeLowerTriangInPlace()
        {
            size_t r = rows();
            size_t c = columns();
            internal_co_symmetrize(0, c, 0, r, matrixByCols.data(), LDA);
            return *this;
        }

        /** Get transpose of the give matrix
        * @return result matrix
        */
        MatrixNMD getTranspose() const
        {
            size_t myLDA = LDA;
            size_t r = rows();
            size_t c = columns();

            MatrixNMD res(c, r);
            size_t resLDA = res.LDA;

            size_t write_pos = 0;
            size_t read_pos = 0;
            size_t read_pos_start = 0;

            for (size_t j = 0; j < c; ++j, read_pos_start += myLDA)
            {
                read_pos = read_pos_start;
                write_pos = res.getFlattenIndexFromPosition(j, 0);

                for (size_t i = 0; i < r; ++i, ++read_pos, write_pos += resLDA)
                {
                    res.matrixByCols[write_pos] = matrixByCols[read_pos];
                }
            }

            return res;
        }

        /** Get column of the matrix with number j (first column has index 0)
        * @param[in] j The number of column to obtain
        * @return Deep copy of column
        * @see getRow()
        */
        MatrixColumn getColumn(size_t j) const
        {            
            size_t readPos = getFlattenIndexFromColumn</*i*/0>(j);

            size_t r = rows();

            MatrixColumn column = MatrixColumn::getUninitializedVector(r);

            dopt::CopyHelpers::copy(column.data(), &matrixByCols.getRaw(readPos), r);

            return column;
        }

        /** Get row of the matrix with number i (first row has index 0)
        * @param i number of row to obtain
        * @return result row
        * @remark access pattern is pretty bad
        */
        MatrixRow getRow(size_t i) const
        {
            size_t c = columns();

            MatrixColumn res = MatrixColumn::getUninitializedVector(c);
            
            for (size_t j = 0; j < c; ++j) {
                res.set(j, get(i, j));
            }

            return res;
        }

        /** Set column of the matrix with number j (first column has index 0) equal to specific column
        * @param j number of column to obtain
        * @param column values for which column should be setuped
        * @return result column
        */
        void setColumn(size_t j, const MatrixColumn& column)
        {
            size_t r = rows();
            size_t writePos = getFlattenIndexFromColumn</*i*/0>(j);
            
            dopt::CopyHelpers::copy(&matrixByCols[writePos], column.dataConst(), r);
        }

        /** Get specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j]
        */
        TElementType get(size_t i, size_t j) const
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            TElementType item = matrixByCols.get(globalIndex);
            return item;
        }

        /** Get const pointer to a specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return pointer to item at position [i,j]
        */
        const TElementType* dataConst(size_t i, size_t j) const
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            const TElementType* item = matrixByCols.dataConst() + globalIndex;
            return item;
        }

        /** Get non-const pointer to a specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return pointer to item at position [i,j]
        */
        TElementType* data(size_t i, size_t j)
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            TElementType* item = matrixByCols.data() + globalIndex;
            return item;
        }

        /** Get a pointer to a specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j]
        */
        TElementType& getRaw(size_t i, size_t j)
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            TElementType* item = matrixByCols.data() + globalIndex;
            return *item;
        }

        /** Get index in underlying flatten column wise representation of matrix
        * @param i number of row in which item is
        * @param j number of column in which item is
        * @return global index
        */
        size_t getFlattenIndexFromPosition(size_t i, size_t j) const {
            size_t plainIndex = j * LDA + i;
            return plainIndex;
        }

        /** Get index in underlying flatten column wise representation of matrix
        * @tparam i number of row in which item is
        * @param j number of column in which item is
        * @return global index
        */
        template <size_t i>
        size_t getFlattenIndexFromColumn(size_t j) const {
            size_t plainIndex = j * LDA + i;
            return plainIndex;
        }

        /** Get index in underlying flatten column wise representation of matrix
        * @tparam j number of column in which item is
        * @param i number of row in which item is
        * @return global index
        */
        template <size_t j>
        size_t getFlattenIndexFromRow(size_t i) const {
            size_t plainIndex = j * LDA + i;
            return plainIndex;
        }
        
        /** Get row and column in which flatten element is lying
        * @param[out] iRow number of row in which item is lying
        * @param[out] jCol number of column in which item is lying
        * @param[in] flatternedIndexByColumns flatten index obtained from getIndexDuringFlatteringByColumns()
        * @sa getIndexDuringFlatteringByColumns
        */
        void getPositionFromFlatternIndex(size_t& iRow, size_t& jCol, size_t flatternedIndexByColumns) const
        {
            // Replace two divisions (that are costly) into two division and multiplication
            // jCol = flatternedIndexByColumns / LDA;
            // iRow = flatternedIndexByColumns % LDA;
            jCol = flatternedIndexByColumns / LDA;
            iRow = flatternedIndexByColumns - jCol * LDA;
        }

        /** Is index flatternedIndexByColumns from upper triangular part
        * @param flatternedIndexByColumns flatterned index of item in the matrix
        * @return true if index from upper triangular part
        */
        bool isIndexFromUpperTriangularPart(size_t flatternedIndexByColumns) const
        {
            size_t jCol = flatternedIndexByColumns / LDA;
            size_t iRow = flatternedIndexByColumns - jCol * LDA;
            return jCol >= iRow;
        }

        /** Get traposed index position in underlying flatten column wise representation of matrix
        * @param[in] flatternedIndexByColumns flatten index
        * @return flatterned index which corresponds to transpose position
        */
        size_t getTranspoedIndexFromFlatternIndex(size_t flatternedIndexByColumns) const
        {
            size_t myLDA = this->LDA;

            size_t jCol = flatternedIndexByColumns / LDA;
            size_t iRow = flatternedIndexByColumns - jCol * LDA;
            size_t indexTranspose = iRow * LDA + jCol;

            return indexTranspose;
        }

        /** Set specific item in the matrix locating in (i,j) to specific value
        * @param i number of row
        * @param j number of column
        * @param value setuped value
        * @return reference to this
        */
        MatrixNMD& set(size_t i, size_t j, TElementType value)
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            matrixByCols.set(globalIndex, value);
            return *this;
        }

        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        MatrixNMD& zeroOutItems(TElementType eps)
        {
            matrixByCols.zeroOutItems(eps);
            return *this;
        }

        /** Set item in the matrix locating in (i,*) to specific value
        * @param i number of row
        * @param items items for which value will be setuped
        * @return reference to this
        */
        MatrixNMD& setRow(size_t i, std::initializer_list<TElementType> rowValues)
        {
            size_t rowValuesSize = rowValues.size();
            assert(rowValuesSize == columns_);

            for (size_t j = 0; j < rowValuesSize; ++j)
            {
                const TElementType& value = *(rowValues.begin() + j);
                set(i, j, value);
            }

            return *this;
        }

        /** Set item in the matrix locating in (i,*) to specific value
        * @param i number of row
        * @param items items for which value will be setuped
        * @return reference to this
        */
        MatrixNMD& setRow(size_t i, const MatrixRow& rowValues)
        {
            size_t rowValuesSize = rowValues.size();
            assert(rowValuesSize == columns_);
            
            for (size_t j = 0; j < rowValuesSize; ++j)
            {
                const TElementType& value = rowValues.get(j);
                set(i, j, value);
            }

            return *this;
        }

        /** Set item in the matrix locating in (*,j) to specific value
        * @param j number of column
        * @param items items for which value will be setuped
        * @return reference to this
        */
        MatrixNMD& setColumn(size_t j, std::initializer_list<TElementType> columnValues)
        {
            assert(rows() == columnValues.size());
            size_t columnValuesSize = columnValues.size();

            for (size_t i = 0; i < columnValuesSize; ++i)
            {
                const TElementType& value = *(columnValues.begin() + i);
                set(i, j, value);
            }

            return *this;
        }

        /** Is item locating in (i,j) position in zero
        * @param i number of row
        * @param j number of column
        * @return true if it so
        */
        bool isNull(size_t i, size_t j) const
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            return matrixByCols.isNull(globalIndex);
        }
        
        /** Is item locating in (i,j) position in zero
        * @param i number of row
        * @param j number of column
        * @return true if it so
        */
        bool isSpecificRowNull(size_t i) const
        {
            size_t cols = columns();
            for (size_t j = 0; j < cols; ++j)
            {
                if (!isNull(i,j))
                    return false;
            }

            return true;
        }

        /** Apply unary minus of matrix affected in element wise way to all items in the matrix and return fresh copy
        * @return copy of the matrix
        */
        MatrixNMD operator - () const
        {
            MatrixNMD<VectorType> res(rows(), columns());

            size_t sz = size();

            for (size_t i = 0; i < sz; ++i)
            {
                const TElementType& item = matrixByCols.get(i);
                res.matrixByCols.set(i, -item);
            }

            return res;
        }

        /** Check that *this matrix is not equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix not equal to rhs
        */
        bool operator != (const MatrixNMD& rhs) const {
            return !(*this == rhs);
        }

        /** Check that *this matrix is equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix is equal to rhs
        */
        bool operator == (const MatrixNMD& rhs) const
        {
            size_t sz = size();

            if (rhs.size() != sz) [[unlikely]]
                return false;

            for (size_t i = 0; i < sz; ++i)
            {
                const TElementType& itemA = matrixByCols.get(i);
                const TElementType& itemB = rhs.matrixByCols.get(i);

                if (itemA != itemB)
                    return false;
            }
            return true;
        }

        /** Special case of matrix multiplication column "u(n,1)" to by row vector "v(1,m)"
        * @param u column vector
        * @param v row vector
        * @result result matrix
        */
        template<class TVec>
        static MatrixNMD outerProduct(const TVec& u, const TVec& v)
        {
            size_t uSize = u.size();
            size_t vSize = v.size();

            MatrixNMD res(uSize, vSize);
            size_t write_pos = 0;
            size_t write_pos_start = 0;

            for (size_t j = 0; j < vSize; ++j, write_pos_start += res.LDA)
            {
                auto vj = v.get(j);

                write_pos = write_pos_start;
                for (size_t i = 0; i < uSize; ++i, ++write_pos)
                {
                    res.matrixByCols.set(write_pos, u.get(i) * vj);
                }
            }

            return res;
        }

        /** Compute transpose(mTranspose) * x, where '*' is usual matrix-vector multiplication
        * @param mTranspose input matrix
        * @param x input vector
        * @result result from matrix vector multiplication with firstly (cocneptually) transpose mTranspose
        */
        static MatrixColumn matrixVectorMultiplyWithPreTranspose(const MatrixNMD& mTranspose, 
                                                                 const MatrixColumn& x)
        {
            assert(mTranspose.rows() == x.size());

            size_t r = mTranspose.columns();
            size_t c = mTranspose.rows();
            
            MatrixColumn res = MatrixColumn::getUninitializedVector(r);

            LightVectorND<VectorType> x_light_vector(const_cast<TElementType*>(&x[0]), c);

            typename MatrixColumn::TElementType* restrict_ext res_out = res.data();

            size_t LDA_For_M_Tr = mTranspose.LDA;
            size_t hint_MTranspose_index = 0;

            for (size_t i = 0; i < r; ++i, hint_MTranspose_index += LDA_For_M_Tr)
            {
                LightVectorND<VectorType> x_col_k(const_cast<TElementType*>(&mTranspose.matrixByCols[hint_MTranspose_index]), c);
                res_out[i] = x_light_vector & x_col_k;
            }

            return res;
        }

        /** Compute transpose(mTranspose) * x, where '*' is usual matrix-vector multiplication
        * @param mTranspose input matrix
        * @param x input vector
        * @result result from matrix vector multiplication with firstly (cocneptually) transpose mTranspose
        * @remark Default (a bit sub-optimal) implementation
        */
        static MatrixColumn matrixVectorMultiplyWithPreTranspose(const MatrixNMD& mTranspose, const MatrixColumn& x, typename MatrixColumn::TElementType beta, const MatrixColumn& v)
        {
            MatrixColumn res = matrixVectorMultiplyWithPreTranspose(mTranspose, x);
            res += beta * v;
            return res;
        }
        
        /** Compute matrix-vector multplication [m*x].
        * @param m input matrix
        * @param x input vector x
        * @return computed matrix-vector product [m*x]
        */
        static MatrixColumn matrixVectorMultiply(const MatrixNMD& m, const MatrixColumn& x)
        {
            assert(m.columns() == x.size());

            size_t r = m.rows();
            size_t c = m.columns();

            MatrixColumn res(r);

            typename MatrixColumn::TElementType* restrict_ext res_out = res.data();

            // Iterate through columns
            size_t jColumnFlatIndex = 0;

            for (size_t j = 0; j < c; ++j, jColumnFlatIndex += m.LDA)
            {
                const typename MatrixColumn::TElementType  xj = x.get(j);
                const typename MatrixColumn::TElementType* restrict_ext xij = &(m.matrixByCols[jColumnFlatIndex]);

                // Use prior assumption that xij point to elements inside one column. Iterate through rows.
                for (size_t i = 0; i < r; ++i)
                {
                    res_out[i] += (xij[i]) * (xj);
                }
            }

            return res;        
        }

        /** Compute [m*x + beta*v].
        * @param m input matrix
        * @param x input vector x
        * @param beta scalar beta
        * @param v vector to multiply vector v
        * @return computed matrix-vector product [m*x] and perform summing of it with vector [beta*v]
        */
        static MatrixColumn matrixVectorMultiply(const MatrixNMD& m, const MatrixColumn& x,
                                                 typename MatrixColumn::TElementType beta, const MatrixColumn& v)
        {
            assert(m.columns() == x.size());

            size_t r = m.rows();
            size_t c = m.columns();

            MatrixColumn res = v * beta;

            typename MatrixColumn::TElementType* restrict_ext res_out = res.data();

            // Iterate through columns
            size_t jColumnFlatIndex = 0;

            for (size_t j = 0; j < c; ++j, jColumnFlatIndex += m.LDA)
            {
                const typename MatrixColumn::TElementType  xj_scaled = x.get(j);
                const typename MatrixColumn::TElementType* restrict_ext xij = &(m.matrixByCols[jColumnFlatIndex]);

                // Use prior assumption that xij point to elements inside one column. Iterate through rows.
                for (size_t i = 0; i < r; ++i)
                {
                    res_out[i] += (xij[i]) * (xj_scaled);
                }
            }

            return res;
        }

        /** Usual matrix-vector multiplication (*this) * (x)
        * @param x the column vector which used for matrix multiplication
        * @return result matrix
        */
        MatrixColumn operator * (const MatrixColumn& x) const 
        {
            return matrixVectorMultiply( (*this), x );
        }

        /** Perform matrix-matrix multiplication "mat(diag) x rhs"
         * @param diag vector which represents elements of the diagonal with dimension [r]
         * @param rhs matrix with shape [r,c]
         * @return result matrix with the shape [r,c]
         */
        static MatrixNMD multiplyDiagonalByDense(const VectorType& diag, const MatrixNMD& rhs)
        {
            MatrixNMD res(rhs);

            size_t r = res.rows();
            size_t c = res.columns();

            size_t write_pos = 0;
            size_t write_pos_start = 0;

            for (size_t j = 0; j < c; ++j, write_pos_start += res.LDA)
            {
                write_pos = write_pos_start;

                for (size_t i = 0; i < r; ++i, ++write_pos)
                {
                    // res[i,j]
                    res.matrixByCols[write_pos] *= diag.get(i);
                }
            }

            return res;
        }

        /** Perform matrix-matrix multiplication "rhs x mat(diag)"
        * @param lhs matrix with shape [r,c]
        * @param diag vector which represents elements of the diagonal with dimension [c]
        * @return result matrix with the shape [r,c]
        */
        static MatrixNMD multiplyDenseByDiagonal(const MatrixNMD& lhs, const VectorType& diag)
        {
            MatrixNMD res(lhs);

            size_t r = res.rows();
            size_t c = res.columns();

            size_t write_pos = 0;
            size_t write_pos_start = 0;

            for (size_t j = 0; j < c; ++j, write_pos_start += res.LDA)
            {
                write_pos = write_pos_start;
                auto aj = diag.get(j);

                for (size_t i = 0; i < r; ++i, ++write_pos)
                {
                    // res[i,j]
                    res.matrixByCols[write_pos] *= aj;
                }
            }

            return res;
        }

        size_t compress() {
            return 0;
        }

        MatrixNMD& operator += (const MatrixNMD& other)
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());
            matrixByCols += other.matrixByCols;

            return *this;
        }

        /** Add to current matrix [this] the muplipler of another matrix [other] in a way that:
        *  [this] := this + (multiple) * other
        * @param multiple muplitple of matrix to add
        * @param other another matrix to add with specific multiplicative factor
        * @tparam updateUpperTriangularPartOnly update only upper triangular part
        */
        template<bool updateUpperTriangularPartOnly>
        void addInPlaceMatrixWithMultiple(TElementType multiple, const MatrixNMD& other)
        {
            assert(LDA == other.LDA);
            assert(columns_ == other.columns_);
            assert(rows_ == other.rows_);
            
            if constexpr (updateUpperTriangularPartOnly)
            {
                size_t cols = columns_;
                size_t myLDA = this->LDA;
                
                dopt::LightVectorND<VectorType> source((const_cast<MatrixNMD&>(other)).matrixByCols.data(), 1);
                dopt::LightVectorND<VectorType> target(matrixByCols.data(), 1);
                
                for (size_t j = 1; j <= cols; ++j, source.components += myLDA, target.components += myLDA)
                {
                    source.componentsCount = j;
                    target.componentsCount = j;                    
                    target.addInPlaceVectorWithMultiple(multiple, source);
                }
            }
            else
            {
                matrixByCols.addInPlaceVectorWithMultiple(multiple, other.matrixByCols);
            }
        }

        MatrixNMD operator + (const MatrixNMD& other) const
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            MatrixNMD<VectorType> res(*this);
            res += other;

            return res;
        }

        MatrixNMD& operator -= (const MatrixNMD& other)
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());
            matrixByCols -= other.matrixByCols;

            return *this;
        }

        MatrixNMD operator - (const MatrixNMD& other) const
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            MatrixNMD<VectorType> res(*this);
            res -= other;
            return res;
        }

        static MatrixNMD computeDifferenceWithUpperTriangularPart(const MatrixNMD& a, const MatrixNMD& b)
        {
            size_t r = a.rows();
            size_t c = a.columns();

            MatrixNMD res(r, c);

            TElementType* aCols = const_cast<TElementType*>(a.matrixByCols.dataConst());
            TElementType* bCols = const_cast<TElementType*>(b.matrixByCols.dataConst());
            TElementType* resCols = res.matrixByCols.data();
            
            assert(a.LDA == b.LDA);
            assert(b.LDA == res.LDA);
            
            size_t usedLDA = a.LDA;

            for (size_t j = 0; j < c; ++j, aCols += usedLDA, bCols += usedLDA, resCols += usedLDA)
            {
                size_t lenForOperation = j + 1;

                LightVectorND<VectorType> a_light_vector(aCols, lenForOperation);
                LightVectorND<VectorType> b_light_vector(bCols, lenForOperation);
                LightVectorND<VectorType> res_light_vector(resCols, lenForOperation);
                
                res_light_vector.assignWithVectorDifferenceAligned(a_light_vector, b_light_vector);
            }

            return res;
        }

        template <typename TFactorType>
        MatrixNMD operator * (TFactorType factor) const
        {
            MatrixNMD<VectorType> res(*this);
            res *= factor;
            return res;
        }

        template<typename TFactorType>
        MatrixNMD& operator *= (TFactorType factor)
        {
            matrixByCols *= factor;
            return *this;
        }

        static MatrixNMD computeDifferenceAndEvalL2Norm(const MatrixNMD& a, const MatrixNMD& b, TElementType& restrict_ext l2NormOfDifference)
        {                
            MatrixNMD<VectorType> res(a);
            res.matrixByCols.computeDiffAndComputeL2Norm(b.matrixByCols, l2NormOfDifference);
            return res;          
        }

        template<typename Type>
        MatrixNMD& addToAllDiagonalEntries(Type addToDiagonal)
        {
            size_t c = columns();

            size_t globalPos = 0;

            size_t offset = LDA + 1;

            for (size_t j = 0; j < c; ++j, globalPos += offset)
            {
                matrixByCols[globalPos] += addToDiagonal;
            }

            return *this;
        }

        template <typename TFactorType>
        MatrixNMD operator / (TFactorType factor) const
        {
            MatrixNMD<VectorType> res(*this);
            res /= factor;
            return res;
        }

        template<typename TFactorType>
        MatrixNMD& operator /= (TFactorType factor)
        {
            matrixByCols /= factor;
            return *this;
        }        

        /** C = C + AB
        * @param A matrix A with (conceputal) shape [m x n]
        * @param B matrix A with (conceputal) shape [n x p] 
        * @param C matrix with (conceputal) shape   [m x p]
        * @param m number of rows of A, and rows in C
        * @param n number of columns of A and rows of B
        * @param p number of columns of B and columns C
        * @remark The physical size of A, B, and C: are m x a_lda, n x b_lda, and m x c_lda,
        */
        static void internal_add_matmul_rec_co(const TElementType* A, const TElementType* B, TElementType* C,
                                               size_t m, size_t n, size_t p,
                                               size_t a_lda, size_t b_lda, size_t c_lda)
        {
            if (m + n + p <= 48)
            { 
                for (size_t k = 0; k < p; ++k)
                    for (size_t j = 0; j < n; ++j)
                        for (size_t i = 0; i < m; ++i)
                            C[i + k * c_lda] += A[i + j * a_lda] * B[j + k * b_lda];
            }
            else
            {
                constexpr size_t a11 = 0;
                size_t a12 = (n / 2) * a_lda;
                size_t a21 = (m / 2);
                size_t a22 = (n / 2) * a_lda + m/2;

                constexpr size_t b11 = 0;
                size_t b12 = (p / 2) * b_lda;
                size_t b21 = (n / 2);
                size_t b22 = (p / 2) * b_lda + n/2;

                constexpr size_t c11 = 0;
                size_t c12 = (p / 2) * c_lda;
                size_t c21 = (m / 2);
                size_t c22 = (p / 2) * c_lda + m/2;

                // A matrix shape [m x n]
                // B matrix shape [n x p]
                // C matrix shape [m x p]
                
                // If blocking A, B, C
                // C11 C12 = A11 A12   *   B11 B12
                // C21 C22 = A21 A22       B21 B22
#if 1
                internal_add_matmul_rec_co(A + a11, B + b11, C + c11, m/2,     n/2,     p/2,     a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a12, B + b21, C + c11, m/2,     n - n/2, p/2,     a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a11, B + b12, C + c12, m/2,     n/2,     p - p/2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a12, B + b22, C + c12, m/2,     n - n/2, p - p/2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a21, B + b11, C + c21, m - m/2, n/2,     p/2,     a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a22, B + b21, C + c21, m - m/2, n - n/2, p/2,     a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a21, B + b12, C + c22, m - m/2, n/2,     p - p/2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a22, B + b22, C + c22, m - m/2, n - n/2, p - p/2, a_lda, b_lda, c_lda);
#else
                // Can be launched in parallel (all write to different parts of C)
                internal_add_matmul_rec_co(A + a11, B + b11, C + c11, m / 2, n / 2, p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a22, B + b21, C + c21, m - m / 2, n - n / 2, p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a11, B + b12, C + c12, m / 2, n / 2, p - p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a22, B + b22, C + c22, m - m / 2, n - n / 2, p - p / 2, a_lda, b_lda, c_lda);
                
                // Sync is needed

                // Can be launched in parallel (all write to different parts of C)
                internal_add_matmul_rec_co(A + a12, B + b21, C + c11, m / 2, n - n / 2, p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a21, B + b11, C + c21, m - m / 2, n / 2, p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a12, B + b22, C + c12, m / 2, n - n / 2, p - p / 2, a_lda, b_lda, c_lda);
                internal_add_matmul_rec_co(A + a21, B + b12, C + c22, m - m / 2, n / 2, p - p / 2, a_lda, b_lda, c_lda);

                // Sync is needed                
#endif
            }
        }
        
        /** Compute matrix-matrix multiplication c = ab
        * @param a first matrix
        * @param b second matrix
        * @tparam kAlgoNumber4MatrixMatrixMult 0-Naive, 1, 2, 3, 4 - Cache Oblivious
        */
        template <int kAlgoNumber4MatrixMatrixMult = 3>
        static MatrixNMD matrixMatrixMultiply(const MatrixNMD& a, const MatrixNMD& b)
        {
            assert(a.columns() == b.rows());
            MatrixNMD res(a.rows(), b.columns());

            size_t m_myrows = res.rows();
            size_t n_mycolumns = res.columns();
            size_t k_xrows = b.rows();


            if (kAlgoNumber4MatrixMatrixMult == 0)
            {
                for (size_t j = 0; j < n_mycolumns; ++j)
                {
                    for (size_t i = 0; i < m_myrows; ++i)
                    {
                        for (size_t k = 0; k < k_xrows; ++k)
                        {
                            res.getRaw(i, j) += a.get(i, k) * b.get(k, j);
                        }
                    }
                }
            }
            if (kAlgoNumber4MatrixMatrixMult == 1)
            {
                for (size_t k = 0; k < k_xrows; ++k)
                {
                    for (size_t j = 0; j < n_mycolumns; ++j)
                    {
                        for (size_t i = 0; i < m_myrows; ++i)
                        {
                            res.getRaw(i, j) += a.get(i, k) * b.get(k, j);
                        }
                    }
                }
            }
            else if (kAlgoNumber4MatrixMatrixMult == 2)
            {
                constexpr size_t i_tile_s_level = 32;
                constexpr size_t j_tile_s_level = 32;
                constexpr size_t k_tile_s_level = 32;

                constexpr size_t i_tile_t_level = 4;
                constexpr size_t j_tile_t_level = 4;
                constexpr size_t k_tile_t_level = 4;

                static_assert(i_tile_s_level % i_tile_t_level == 0);
                static_assert(j_tile_s_level % j_tile_t_level == 0);
                static_assert(k_tile_s_level % k_tile_t_level == 0);

                const size_t r = res.rows();
                const size_t c = res.columns();
                const size_t k_xrows = b.rows();

                const size_t a_lda = a.LDA;
                const size_t b_lda = b.LDA;
                const size_t res_lda = res.LDA;

                for (size_t is = 0; is < r; is += i_tile_s_level)
                {
                    size_t it_bound = minimum(is + i_tile_s_level, r);

                    for (size_t js = 0; js < c; js += j_tile_s_level)
                    {
                        size_t jt_bound = minimum(js + j_tile_s_level, c);

                        for (size_t ks = 0; ks < k_xrows; ks += k_tile_s_level)
                        {
                            size_t kt_bound = minimum(ks + k_tile_s_level, k_xrows);

                            for (size_t it = is; it < it_bound; it += i_tile_t_level)
                            {
                                const size_t& ist = it;

                                size_t ik_bound = minimum(ist + i_tile_t_level, r);

                                for (size_t jt = js; jt < jt_bound; jt += j_tile_t_level)
                                {
                                    const size_t& jst = jt;

                                    size_t jk_bound = minimum(jst + j_tile_t_level, c);

                                    for (size_t kt = ks; kt < kt_bound; kt += k_tile_t_level)
                                    {
                                        const size_t& kst = kt;
                                        size_t kk_bound = minimum(kst + k_tile_t_level, k_xrows);

                                        for (size_t ik = ist; ik < ik_bound; ++ik)
                                        {
                                            const size_t& i = ik;

                                            for (size_t jk = jst; jk < jk_bound; ++jk)
                                            {
                                                const size_t& j = jk;

                                                // res[i, j]
                                                TElementType* result_pointer = &(res.matrixByCols.getRaw(i + j * res_lda));

                                                for (size_t kk = kst; kk < kk_bound; ++kk)
                                                {
                                                    const size_t& k = kk;

                                                    //res.getRaw(i, j) += get(i, k) * b.get(k, j);
                                                    *result_pointer += a.matrixByCols.get(i + k * a_lda) *
                                                                       b.matrixByCols.get(k + j * b_lda);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (kAlgoNumber4MatrixMatrixMult == 3)
            {
                // If select s,t [HK81] shows that there are specific tiling sizes which are in some sense are optimal
                
                constexpr size_t i_tile_s_level = 32; /// Tiles size for optimality should be 1/3 * sqrt(L2/8)
                constexpr size_t j_tile_s_level = 32;
                constexpr size_t k_tile_s_level = 32;

                constexpr size_t i_tile_t_level = 4;  /// Tiles size for optimality should be 1/3 * sqrt(L1/8)
                constexpr size_t j_tile_t_level = 4;
                constexpr size_t k_tile_t_level = 4;

                static_assert(i_tile_s_level % i_tile_t_level == 0);
                static_assert(j_tile_s_level % j_tile_t_level == 0);
                static_assert(k_tile_s_level % k_tile_t_level == 0);

                const size_t r = res.rows();
                const size_t c = res.columns();
                const size_t k_brows = b.rows();

                const size_t a_lda = a.LDA;
                const size_t b_lda = b.LDA;
                const size_t res_lda = res.LDA;

                for (size_t is = 0; is < r; is += i_tile_s_level)
                {
                    for (size_t js = 0; js < c; js += j_tile_s_level)
                    {
                        for (size_t ks = 0; ks < k_xrows; ks += k_tile_s_level)
                        {
                            for (size_t it = 0; it < i_tile_s_level; it += i_tile_t_level)
                            {
                                const size_t ist = is + it;
                                if (ist >= r)
                                    break;

                                for (size_t jt = 0; jt < j_tile_s_level; jt += j_tile_t_level)
                                {
                                    const size_t jst = js + jt;
                                    if (jst >= c)
                                        break;

                                    for (size_t kt = 0; kt < k_tile_s_level; kt += k_tile_t_level)
                                    {
                                        const size_t kst = ks + kt;
                                        if (kst >= k_xrows)
                                            break;

                                        for (size_t kk = 0; kk < k_tile_t_level; ++kk)
                                        {
                                            const size_t k = kst + kk;
                                            if (k >= k_xrows)
                                                break;

                                            for (size_t jk = 0; jk < j_tile_t_level; ++jk)
                                            {
                                                const size_t j = jst + jk;
                                                if (j >= c)
                                                    break;

                                                for (size_t ik = 0; ik < i_tile_t_level; ++ik)
                                                {
                                                    const size_t i = ist + ik;
                                                    if (i >= r)
                                                        break;

                                                    // res[i, j]
                                                    TElementType* result_pointer = &(res.matrixByCols.getRaw(i + j * res_lda));
                                                    *result_pointer += a.matrixByCols.get(i + k * a_lda) *
                                                                       b.matrixByCols.get(k + j * b_lda);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (kAlgoNumber4MatrixMatrixMult == 4)
            {
                // Recursive implementation of Matrix Multiplication with optimal cache misses for large matrices in full-associative cache: Cache Misses (n^3/ (B sqrt{M}) ) which is optimal
                // But this algorithms is Cache-Oblivious
                
                internal_add_matmul_rec_co(a.matrixByCols.dataConst(), b.matrixByCols.dataConst(), res.matrixByCols.data(),
                                           a.rows(), a.columns(), b.columns(),
                                           a.LDA, b.LDA, res.LDA);
            }

            return res;
        }
        
        MatrixNMD operator * (const MatrixNMD & x) const {
            return matrixMatrixMultiply(*this, x);
        }

        MatrixNMD& operator *= (const MatrixNMD& x)
        {
            MatrixNMD<VectorType> mat_copy(*this);
            *this = mat_copy * x;
            return *this;
        }

        MatrixNMD operator ^ (int power) const
        {
            if (!isSquareMatrix())
            {
                assert(!"FOR POWER OPERATION PLEASE USE SQUARE MATRIX");
                return *this;
            }

            if (power < 0)
            {
                assert(!"FOR POWER OPERATION INVERSTION IS NOT SUPPORTED");
                return *this;
            }
            else if (power == 0)
            {
                return MatrixNMD::getIdentitySquareMatrix(rows());
            }
            else if (power % 2 == 0)
            {
                MatrixNMD tmp = *this ^ (power / 2);
                return tmp * tmp;
            }
            else
            {
                MatrixNMD tmp = (*this) ^ ((power - 1) / 2);
                return tmp * tmp * (*this);
            }
        }
        
        /** Does this class contain SIMD support
        * @return true if class specialization support CPU acceleration with SIMD, false otherwise.
        */
        static bool hasSIMDSupport() {
            return false;
        }

        /** Does this class contain CUDA support
        * @return true if class specialization support GPU acceleration with CUDA, false otherwise.
        */
        static bool hasCUDASupport() {
            return false;
        }

        /** Set all elements (excluding padding) randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        template<class Generator>
        void setAllRandomly(Generator& generator)
        {
            size_t myLDA = LDA;
            size_t index_columnd_start = 0;
            
            for (size_t j = 0; j < columns_; ++j, index_columnd_start += myLDA)
            {
                for (size_t i = 0; i < rows_; ++i)
                {
                    matrixByCols.set(i + index_columnd_start, generator.generateReal());
                }
            }
        }

        /** Set all elements (excluding padding) for specific value
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        void setAll(TElementType value)
        {
            size_t myLDA = LDA;
            size_t index_columnd_start = 0;
            
            for (size_t j = 0; j < columns_; ++j, index_columnd_start += myLDA)
            {
                LightVectorND<VectorType> col(matrixByCols.data() + index_columnd_start, 
                                              rows_);
                col.setAll(value);
            }
        }

        /** Set all items of matrix to default value
        * @return reference to itself
        */
        MatrixNMD& setAllToDefault() {
            matrixByCols.setAllToDefault();
            return *this;
        }

        template<class RandomGen>
        void applyNaturalCompressor(RandomGen& rndGen)
        {
            size_t myLDA = LDA;
            size_t index_columnd_start = 0;

            for (size_t j = 0; j < columns_; ++j, index_columnd_start += myLDA)
            {
                TElementType* item = matrixByCols.data() + index_columnd_start;
                
                for (size_t i = 0; i < rows_; ++i, ++item)
                {
                    auto pack = getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item[i]);
                    double m = getMantissaPartNoLeadingOne<DOPT_ARCH_LITTLE_ENDIAN> (pack);
                    
                    double choice = rndGen.generateRealInUnitInterval();

                    if (choice < m) 
                    {
                        // w.p. m
                        pack.components.exponent += 1;
                    }
                    
                    pack.components.mantissa = 0;

                    item[i] = pack.real_value_repr;
                }
            }
        }

        void applyNaturalCompressor() {
            return applyNaturalCompressorNaive();
        }

        void applyNaturalCompressorNaive()
        {
            size_t myLDA = LDA;
            size_t index_columnd_start = 0;

            for (size_t j = 0; j < columns_; ++j, index_columnd_start += myLDA)
            {
                TElementType* item = matrixByCols.data() + index_columnd_start;

                for (size_t i = 0; i < rows_; ++i)
                {
                    auto pack = getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item[i]);
                    pack.components.mantissa = 0;
                    item[i] = pack.real_value_repr;
                }
            }
        }
    };

    template<class VectorType>
    MatrixNMD<VectorType> operator * (typename VectorType::TElementType factor, const MatrixNMD<VectorType>& x)
    {
        return x * factor;
    }
}

#if DOPT_INCLUDE_VECTORIZED_CPU_IMP_MATS
    #include "dopt/linalg_matrices/include/include_internal/MatrixNMD_SIMD_double.h"
    #include "dopt/linalg_matrices/include/include_internal/MatrixNMD_SIMD_float.h"
#endif
