#pragma once

#include "dopt/copylocal/include/Copier.h"

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/system/include/FloatUtils.h"

#include "dopt/gpu_compute_support/include/linalg_vectors/LightVectorND_CUDA.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_matrices/MatrixNMD_CUDA_Raw_kernels.h"
#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"

#include <initializer_list>
#include <sstream>
#include <iostream>

#include <assert.h>
#include <stddef.h>
#include <vector>

namespace dopt
{
    /** Dense matrix stored by columns. Under development.
    * @note Possible problems LDA != rows => we use it for memory padding (at least for now). If it is the case we should somehow preserve the padding as zeros (to work vecL2norm properly).
    * @note For now we just make LDA == rows
    */
    template <class VectorType>
    class MatrixNMD_CUDA
    {
    private:        
        /** Compute leading dimension for matrix. The number of rows in matrix A, B which include any memory padding for access efficiency.
        * @param theRows number of rows matrix
        * @return offset between near by columns
        */
        constexpr static inline size_t computeLda(size_t theRows)
        {
            constexpr size_t kAlgorithm = 1; // See note

            if constexpr (kAlgorithm == 1)
            {
                // Economic style in terms of allocated bytes for dense matrix
                // Benefits:
                //  -- Simplify computation of Frobenious norm of the matrix
                //  -- Simplify prepare memory buffers for transfer over the network
                return theRows;
            }
            else if constexpr (kAlgorithm == 2)
            {                
                constexpr size_t kCacheLizeSizeInBytes = 64;
                constexpr size_t kVecBatchSize = kCacheLizeSizeInBytes / sizeof(TElementType);
                static_assert(kCacheLizeSizeInBytes % sizeof(TElementType) == 0);
                static_assert(kCacheLizeSizeInBytes >= sizeof(TElementType));
                size_t LDAInItems = dopt::roundToNearestMultipleUp<kVecBatchSize>(theRows);
                assert(LDAInItems >= theRows);
                return LDAInItems;
            }
            else
            {
                assert(! "PLEASE SPECIFY ALGORITHM");
            }
        }

    public:
        typedef typename VectorType::TElementType TElementType;  ///< Typedef for element type
        typedef VectorType MatrixRow;                            ///< Typedef for matrix row if accessed
        typedef VectorType MatrixColumn;                         ///< Typedef for matrix column if accessed

        size_t rows_;                                            ///< Number of rows in matrix [EQUAL TO LDA]
        size_t columns_;                                         ///< Number of colums in matrix
        size_t LDA;                                              ///< Leading dimension for matrix. The number of rows in matrix A, B which include any memory padding for access efficiency. [For easy of transfering and developement rows_ == LDA]
        VectorType matrixByCols;                                 ///< Components of the matrix stored by columns

        /** Construct empty dense matrix
        */
        MatrixNMD_CUDA() noexcept
        : matrixByCols()
        , rows_(0)
        , columns_(0)
        , LDA(0)
        {
        }

        enum InitPolicyForStorage {
            eNotAllocate = 0,        ///< Warning: Not allocate underlying storage. Use it if you understand what you are doing.
            eAllocNotInit = 1,       ///< Warning: Allocate underlying storage, but not initialized. Use it if you understand what you are doing.
            eAllocAndSetToZero = 2   ///< Allocate underlying storage and initialize to zero
        };

        /** Construct dense matrix with specified size
        * @param theRows number of rows
        * @param theColumns number of columns
        * @param initPolicy policy for initialization of storage
        * @remark all component are initialization with default ctor or zero.
        */
        MatrixNMD_CUDA(size_t theRows, size_t theColumns, InitPolicyForStorage initPolicy) noexcept
        : rows_(theRows)
        , columns_(theColumns)
        , LDA(computeLda(theRows))
        , matrixByCols( (typename VectorType::InitPolicyForStorage) initPolicy, LDA * theColumns, dopt::GpuManagement(GpuManagement::defaultGPUDevice()) )
        {
        }


        /** Construct dense matrix with specified size
        * @param theRows number of rows
        * @param theColumns number of columns
        * @remark all component are initialization with default ctor or zero.
        */
        MatrixNMD_CUDA(size_t theRows, size_t theColumns) noexcept
        : MatrixNMD_CUDA(theRows, theColumns, eAllocAndSetToZero)
        {
        }

        /** Construct dense matrix with specified size
        * @param theRows number of rows
        * @param theColumns number of columns
        * @param targetDevice device where matrix will be allocated
        * @param initPolicy policy for initialization of storage
        * @remark all component are initialization with default ctor or zero.
        */
        MatrixNMD_CUDA(size_t theRows, size_t theColumns, GpuManagement targetDevice, InitPolicyForStorage initPolicy) noexcept
        : rows_(theRows)
        , columns_(theColumns)
        , LDA(computeLda(theRows))
        , matrixByCols((typename VectorType::InitPolicyForStorage) initPolicy, LDA* theColumns, targetDevice)
        {
        }
        
        /** Copy constructor
        * @param rhs array from which copy is occurring
        */
        MatrixNMD_CUDA(const MatrixNMD_CUDA& rhs) noexcept
        : matrixByCols(rhs.matrixByCols)
        , rows_(rhs.rows_)
        , columns_(rhs.columns_)
        , LDA(rhs.LDA)
        {}

        /** Assignment operator
        * @param rhs expression from which we perform copy
        */
        MatrixNMD_CUDA& operator = (const MatrixNMD_CUDA& rhs) noexcept
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
        MatrixNMD_CUDA& operator = (MatrixNMD_CUDA&& rhs) noexcept
        {
            matrixByCols = std::move(rhs.matrixByCols);
            rows_ = rhs.rows_;
            columns_ = rhs.columns_;
            LDA = rhs.LDA;
            return *this;
        }

        /** Copy move operator
        */
        MatrixNMD_CUDA(MatrixNMD_CUDA&& rhs) noexcept
        : matrixByCols(std::move(rhs.matrixByCols))
        , rows_(rhs.rows_)
        , columns_(rhs.columns_)
        , LDA(rhs.LDA)
        {}

        /** Destructor
        */
        ~MatrixNMD_CUDA() = default;

        /** Size in bytes for all matrix elements excluding padding
        * @return number of bytes to store all elements
        */
        size_t sizeInBytesNoPadding() const {
            return rows_ * columns_ * sizeof(TElementType);
        }

        /** Debug print shape info
        * @param out text stream into which printing will have place to be
        * @param variableName name of the variable
        * @return string represented shape of the matrix
        */
        template<class text_out_steam>
        std::string dbgPrintShapeInfo(text_out_steam& out, 
                                      const char* variableName = "xGpu") const
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
                                   const char* variableName  = "xGpu=",
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
        * @remark the return number includes padding for LDA
        */
        size_t size() const {
            return matrixByCols.size();
        }

        /** Sum of diagonal elements of the matrix [NOT FULLY OPTIMIZED]
        * @return trace of the matrix which is of all diagonal elements
        * @remark sum of diagonal elements it is equal to the sum of all eigenvalues of the matrix
        * @todo add better GPU implementation in case of having trace clients
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
        bool isDiagonal(TElementType eps = TElementType()) const
        {
            GpuManagement dev(device());

            bool hostFailFlag = false;
            bool* devFailFlag = (bool*) dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));            
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsDiagonal, dev, rawDevData(), LDA, rows(), columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            return !hostFailFlag;
        }

        /** Return flag that matrix is three diagonal
        * @return true if condition is true
        */
        bool isThreeDiagonal(TElementType eps = TElementType()) const
        {
            GpuManagement dev(device());

            bool hostFailFlag = false;
            bool* devFailFlag = (bool*)dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsThreeDiagonal, dev, rawDevData(), LDA, rows(), columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            return !hostFailFlag;
        }

        /** Append extra columns to the matrix
        * @param extraColumns number of extra columns to append
        * @return reference to this
        * @remark this call may invalidate previous pointer to internal storage
        */
        MatrixNMD_CUDA& appendColumns(size_t extraColumns)
        {
            size_t oldSize = size();
            size_t newSize = oldSize + extraColumns * LDA;
            matrixByCols.resize(newSize);            
            columns_ += extraColumns;

            return *this;
        }

        /** Append extra columns to the matrix
        * @param extraColumns number of extra columns to append
        * @return reference to this
        */
        MatrixNMD_CUDA& removeColumns(size_t extraColumns)
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
            GpuManagement dev(device());

            bool hostFailFlag = false;
            bool* devFailFlag = (bool*)dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsUpperTriangular, dev, rawDevData(), LDA, rows(), columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            return !hostFailFlag;
        }

        /** Return flag that matrix is lower triangular
        * @return true if condition is true
        */
        bool isLowerTriangular(TElementType eps = TElementType()) const
        {
            GpuManagement dev(device());

            bool hostFailFlag = false;
            bool* devFailFlag = (bool*)dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsLowerTriangular, dev, rawDevData(), LDA, rows(), columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            return !hostFailFlag;
        }

        /** Return flag that matrix is zero matrix
        * @return true if condition is true
        */
        bool isZeroMatrix(TElementType eps = TElementType()) const
        {
            GpuManagement dev(device());

            bool hostFailFlag = false;
            bool* devFailFlag = (bool*)dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsZero, dev, rawDevData(), LDA, rows(), columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            return !hostFailFlag;
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
        * @todo add better GPU implementation in case of having isOrthogonal() clients
        */
        bool isOrthogonal(TElementType eps = TElementType()) const
        {
            MatrixNMD_CUDA tr = getTranspose();
            MatrixNMD_CUDA res = matrixMatrixMultiply(tr, *this);


            GpuManagement dev(res.device());
            bool hostFailFlag = false;
            bool* devFailFlag = (bool*)dev.allocateBytesInDevice(sizeof(bool));

            dev.setDeviceMemoryToZero(devFailFlag, sizeof(bool));
            applyMatrixCheck(devFailFlag, MatrixTestApi::eIsIdentity, dev, res.rawDevData(), res.LDA, res.rows(), res.columns(), eps);
            dev.copyDevice2HostSync(&hostFailFlag, devFailFlag, 1);
            dev.freeMemoryInDevice(devFailFlag);

            if (hostFailFlag)
                return false;
            else 
                return true; 
        }

        /** Return flag that matrix is symmetric [NOT FULLY OPTIMIZED]
        * @return true if condition is true
        * @todo add better GPU implementation in case of having isSymmetric() clients
        */
        bool isSymmetric() const
        {
            if (!isSquareMatrix()) [[unlikely]]
                return false;

            MatrixNMD_CUDA tmp = getTranspose();
            return tmp == *this;
        }

        /** Return flag that matrix is skew symmetric [NOT FULLY OPTIMIZED]
        * @return true if condition is true
        * @todo add better GPU implementation in case of having isSkewSymmetric() clients
        */
        bool isSkewSymmetric() const
        {
            MatrixNMD_CUDA tmp = getTranspose();
            tmp *= (-TElementType(1));
            return *this == tmp;            
        }

        /** Create or instantiate matrix contains only zeros with specified dimensions flag that matrix is skew symmetric
        * @param theRows number of rows in matrix
        * @param theColumns number of columns in matrix
        * @return result matrix
        */
        static MatrixNMD_CUDA getZeroMatrix(size_t theRows, size_t theColumns)
        {
            MatrixNMD_CUDA res(theRows, theColumns);
            return res;
        }

        /** Create or instantiate matrix contains non-zeros only in diagonal
        * @param theRows number of rows in matrix
        * @param theColumns number of columns in matrix
        * @param diagElement diagonal element which will be used for initialize
        * @return result matrix
        */
        static MatrixNMD_CUDA getDiagonalMatrix(size_t theRowsAndColumns, TElementType diagElement = TElementType(1))
        {
            MatrixNMD_CUDA res(theRowsAndColumns, theRowsAndColumns);
            applyKernelToSetItemToDiagonal(res.rawDevData(),
                                           res.matrixByCols.device(),
                                           res.LDA,
                                           theRowsAndColumns, 
                                           diagElement);
            return res;
        }

        /** Create or instantiate  square matrix contains only zeros
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        static MatrixNMD_CUDA getZeroSquareMatrix(size_t dim)
        {
            return MatrixNMD_CUDA(dim, dim);
        }

        /** Create or instantiate  identity square matrix
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        static MatrixNMD_CUDA getIdentitySquareMatrix(size_t dim)
        {
            return getDiagonalMatrix(dim, TElementType(1));
        }

        /** Get transpose of the give matrix. Naive implementation. [NOT OPTIMIZED]
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        * @remark This function is not optimized for performance. Use it only for testing purposes. It is extremely slow.
        */
        MatrixNMD_CUDA getTransposeNaive() const
        {
            size_t r = rows();
            size_t c = columns();

            TElementType* hostThis = new TElementType[size()];

            MatrixNMD_CUDA res(c, r);
            TElementType* hostRes = new TElementType[res.size()];

            dopt::GpuManagement dev(device());
            dev.copyDevice2HostSync(hostThis, rawDevData(), size());
            
            for (size_t i = 0; i < r; ++i)
            {
                for (size_t j = 0; j < c; ++j)
                {
                    TElementType valIn = hostThis[getFlattenIndexFromPosition(i, j)];
                    TElementType& valOut = hostRes[res.getFlattenIndexFromPosition(j, i)];
                    valOut = valIn;
                }
            }
            dev.copyHost2DeviceSync(res.rawDevData(), hostRes, res.size());
            
            delete []hostRes;
            delete []hostThis;
            
            return res;
        }

        /** Get transpose of the matrix. Cache Oblivious implementation.
        * @return result matrix
        * @remark This cache oblivious implementation of matrix transpose (http://users.cecs.anu.edu.au/~Alistair.Rendell/papers/coa.pdf).
        * @remark Algorithm has asymptotically optimal cache performance without knowledge about cache.
        */
        MatrixNMD_CUDA getTransposeCO() const {
            return getTranspose();
        }

        /** Symmetrize matrix with using upper triangular part in place.
        *   Copy elements from upper triangular part excluding diagonal to lower triangular part.
        */
        MatrixNMD_CUDA& symmetrizeLowerTriangInPlace()
        {
            assert(isSquareMatrix());
            applyKernelToSymmetrizeLowerTriangInPlace(device(), rawDevData(), rows(), LDA);
            return *this;
        }

        /** Get transpose of the give matrix
        * @param dim number of rows and columns in the matrix
        * @return result matrix
        */
        MatrixNMD_CUDA getTranspose() const
        {
            MatrixNMD_CUDA res(columns(), rows());

            applyKernelToCreateTranspose(res.rawDevData(),
                                         res.LDA,
                                         res.matrixByCols.device(),
                                         rawDevData(),
                                         LDA,
                                         rows(),
                                         columns());
            return res;
        }

        /** Get column of the matrix with number j (first column has index 0)
        * @param j number of column to obtain
        * @return result column
        */
        MatrixColumn getColumn(size_t j) const
        {
            size_t r = rows();
            MatrixColumn res(r);
            size_t readPos = getFlattenIndexFromColumn</*i*/0> (j);
            
            res.device().copyHost2DeviceSync(res.rawDevData(),
                                             rawDevData() + readPos,
                                             r);
            return res;
        }

        /** Get row of the matrix with number i (first row has index 0)
        * @param i number of row to obtain
        * @return result row
        * @remark access pattern is pretty bad
        * @todo add better GPU implementation in case of having getRow() clients
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

            matrixByCols.device().copyHost2DeviceSync(rawDevData() + writePos,
                                                      column.rawDevData(),
                                                      r);
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

        /* Obtain raw pointer to underlying data
        * @return raw pointer
        */
        TElementType* rawDevData() {
            return matrixByCols.rawDevData();
        }

        /* Obtain raw const pointer to underlying data
        * @return raw pointer
        */
        const TElementType* rawDevData() const {
            return matrixByCols.rawDevData();
        }

        /** Get a pointer to a specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j] in GPU memory.
        * @remark This a bit dangerous method. Use it only if you know what you are doing. It is a pointer to GPU memory.
        */
        TElementType& getRaw(size_t i, size_t j)
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            return matrixByCols.getRaw(globalIndex);
        }

        /** Get a pointer to a specific item in the matrix locating in (i,j)
        * @param i number of row
        * @param j number of column
        * @return reference to item at position [i,j] in GPU memory.
        * @remark This a bit dangerous method. Use it only if you know what you are doing. It is a pointer to GPU memory.
        */
        const TElementType& getRaw(size_t i, size_t j) const
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            return matrixByCols.getRaw(globalIndex);
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
        MatrixNMD_CUDA& set(size_t i, size_t j, TElementType value)
        {
            size_t globalIndex = getFlattenIndexFromPosition(i, j);
            matrixByCols.set(globalIndex, value);
            return *this;
        }

        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        MatrixNMD_CUDA& zeroOutItems(TElementType eps)
        {
            matrixByCols.zeroOutItems(eps);
            return *this;
        }

        /** Set item in the matrix locating in (i,*) to specific value
        * @param i number of row
        * @param items items for which value will be setuped
        * @return reference to this
        */
        MatrixNMD_CUDA& setRow(size_t i, std::initializer_list<TElementType> rowValues)
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
        MatrixNMD_CUDA& setRow(size_t i, const MatrixRow& rowValues)
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
        MatrixNMD_CUDA& setColumn(size_t j, std::initializer_list<TElementType> columnValues)
        {
            assert(rows() == columnValues.size());
            
            size_t r = rows();
            size_t writePos = getFlattenIndexFromColumn</*i*/0>(j);

            std::vector<TElementType> columnValuesTmp(columnValues.begin(), columnValues.end());

            matrixByCols.device().copyHost2DeviceSync(rawDevData() + writePos,
                                                      columnValuesTmp.data(),
                                                      r);
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
        * @todo add better GPU implementation in case of having trace clients
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
        MatrixNMD_CUDA operator - () const
        {
            MatrixNMD_CUDA<VectorType> res(rows(), columns(), InitPolicyForStorage::eNotAllocate);
            res.matrixByCols = -matrixByCols;
            return res;
        }

        /** Check that *this matrix is not equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix not equal to rhs
        */
        bool operator != (const MatrixNMD_CUDA& rhs) const {
            return !(*this == rhs);
        }

        /** Check that *this matrix is equal to rhs
        * @param rhs other matrix with which we perform compare
        * @return if current matrix is equal to rhs
        */
        bool operator == (const MatrixNMD_CUDA& rhs) const
        {
            size_t sz = size();
            if (rhs.size() != sz) [[unlikely]]
                return false;
                
            return matrixByCols == rhs.matrixByCols;
        }
        
        /** Special case of matrix multiplication column "u(n,1)" to by row vector "v(1,m)"
        * @param u column vector
        * @param v row vector
        * @result result matrix
        */
        template<class TVec>
        static MatrixNMD_CUDA outerProduct(const TVec& u, const TVec& v)
        {
            size_t uSize = u.size();
            size_t vSize = v.size();
            MatrixNMD_CUDA res(uSize, vSize, InitPolicyForStorage::eAllocAndSetToZero);

            applyKernelToMatrixOuterProduct(res.matrixByCols.rawDevData(), 
                                            res.LDA, 
                                            res.matrixByCols.device(),
                                            u.rawDevData(), 
                                            v.rawDevData(),
                                            uSize, 
                                            vSize);
            return res;
        }

        /** Compute transpose(mTranspose) * x, where '*' is usual matrix-vector multiplication
        * @param mTranspose input matrix
        * @param x input vector
        * @result result from matrix vector multiplication with firstly (cocneptually) transpose mTranspose
        * @warning [NOT OPTIMIZED]
        */
        static MatrixColumn matrixVectorMultiplyWithPreTranspose(const MatrixNMD_CUDA& mTranspose, 
                                                                 const MatrixColumn& x)
        {
            auto m = mTranspose.getTranspose();
            return matrixVectorMultiply(m, x);            
        }

        /** Compute transpose(mTranspose) * x, where '*' is usual matrix-vector multiplication
        * @param mTranspose input matrix
        * @param x input vector
        * @result result from matrix vector multiplication with firstly (cocneptually) transpose mTranspose
        * @remark Default (a bit sub-optimal) implementation
        * @warning [NOT OPTIMIZED]
        */
        static MatrixColumn matrixVectorMultiplyWithPreTranspose(const MatrixNMD_CUDA& mTranspose, const MatrixColumn& x, typename MatrixColumn::TElementType beta, const MatrixColumn& v)
        {
            auto m = mTranspose.getTranspose();
            return matrixVectorMultiply(m, x, beta, v);
        }
        
        static MatrixColumn matrixVectorMultiply(const MatrixNMD_CUDA& m, const MatrixColumn& x)
        {
            assert(m.columns() == x.size());

            size_t r = m.rows();
            size_t c = m.columns();

            MatrixColumn res(r);

            applyKernelToMatrixTimesVector( res.device(),
                                            res.rawDevData(),
                                            m.matrixByCols.rawDevData(),
                                            r, c, m.LDA,
                                            x.rawDevData() );
            return res;
        }

        /** Compute [alpha(m*x) + beta*v]
        */
        static MatrixColumn matrixVectorMultiply(const MatrixNMD_CUDA& m, const MatrixColumn& x, typename MatrixColumn::TElementType beta, const MatrixColumn& v)
        {
            assert(m.columns() == x.size());

            size_t r = m.rows();
            size_t c = m.columns();

            MatrixColumn res(r);
            assert(res.size() == v.size());

            applyKernelToExtMatrixTimesVector(res.device(),
                                              res.rawDevData(),
                                              m.matrixByCols.rawDevData(),
                                              r, c, m.LDA,
                                              x.rawDevData(),
                                              beta, v.rawDevData());
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

        static MatrixNMD_CUDA multiplyDiagonalByDense(const VectorType& diag, const MatrixNMD_CUDA& rhs)
        {
            size_t r = rhs.rows();
            size_t c = rhs.columns();            
            MatrixNMD_CUDA res(r, c, InitPolicyForStorage::eAllocAndSetToZero);

            applyKernelToDiagTimesMatrix(res.matrixByCols.rawDevData(),
                                         res.LDA,
                                         res.matrixByCols.device(),
                                         diag.rawDevData(),
                                         rhs.matrixByCols.rawDevData(),
                                         rhs.LDA,
                                         r, c);

            return res;
        }

        static MatrixNMD_CUDA multiplyDenseByDiagonal(const MatrixNMD_CUDA& lhs, const VectorType& diag)
        {
            size_t r = lhs.rows();
            size_t c = lhs.columns();
            MatrixNMD_CUDA res(r, c, InitPolicyForStorage::eAllocAndSetToZero);

            applyKernelToMatrixTimesDiag(res.matrixByCols.rawDevData(),
                                         res.LDA,
                                         res.matrixByCols.device(),
                                         lhs.matrixByCols.rawDevData(),
                                         lhs.LDA,
                                         diag.rawDevData(), 
                                         r, c);

            return res;
        }
        
        size_t compress() {
            return 0;
        }

        MatrixNMD_CUDA& operator += (const MatrixNMD_CUDA& other)
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
        */
        template<bool updateUpperTriangularPartOnly>
        void addInPlaceMatrixWithMultiple(TElementType multiple, const MatrixNMD_CUDA& other)
        {
            assert(LDA == other.LDA);
            assert(columns_ == other.columns_);
            assert(rows_ == other.rows_);
            
            if constexpr (updateUpperTriangularPartOnly)
            {
                size_t cols = columns_;
                size_t myLDA = this->LDA;
                
                dopt::LightVectorND_CUDA<VectorType> source((const_cast<MatrixNMD_CUDA&>(other)).matrixByCols.rawDevData(), 1);
                dopt::LightVectorND_CUDA<VectorType> target(matrixByCols.rawDevData(), 1);
                
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

        MatrixNMD_CUDA operator + (const MatrixNMD_CUDA& other) const
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            MatrixNMD_CUDA<VectorType> res(rows(), columns(), InitPolicyForStorage::eAllocAndSetToZero);
            res.matrixByCols = matrixByCols + other.matrixByCols;

            return res;
        }

        MatrixNMD_CUDA& operator -= (const MatrixNMD_CUDA& other)
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());
            matrixByCols -= other.matrixByCols;

            return *this;
        }

        MatrixNMD_CUDA operator - (const MatrixNMD_CUDA& other) const
        {
            assert(rows() == other.rows());
            assert(columns() == other.columns());

            MatrixNMD_CUDA<VectorType> res(rows(), columns(), InitPolicyForStorage::eAllocAndSetToZero);
            res.matrixByCols = matrixByCols - other.matrixByCols;

            return res;
        }

        static MatrixNMD_CUDA computeDifferenceWithUpperTriangularPart(const MatrixNMD_CUDA& a, const MatrixNMD_CUDA& b)
        {
            size_t r = a.rows();
            size_t c = a.columns();

            MatrixNMD_CUDA res(r, c);

            TElementType* aCols = const_cast<TElementType*>(a.matrixByCols.rawDevData());
            TElementType* bCols = const_cast<TElementType*>(b.matrixByCols.rawDevData());
            TElementType* resCols = res.matrixByCols.rawDevData();
            
            assert(a.LDA == b.LDA);
            assert(b.LDA == res.LDA);
            
            size_t usedLDA = a.LDA;

            for (size_t j = 0; j < c; ++j, aCols += usedLDA, bCols += usedLDA, resCols += usedLDA)
            {
                size_t lenForOperation = j + 1;

                LightVectorND_CUDA<VectorType> a_light_vector(aCols, lenForOperation, a.matrixByCols.device());
                LightVectorND_CUDA<VectorType> b_light_vector(bCols, lenForOperation, b.matrixByCols.device());
                LightVectorND_CUDA<VectorType> res_light_vector(resCols, lenForOperation, res.matrixByCols.device());
                
                res_light_vector.assignWithVectorDifferenceAligned(a_light_vector, b_light_vector);
            }

            return res;
        }

        template <typename TFactorType>
        MatrixNMD_CUDA operator * (TFactorType factor) const
        {
            MatrixNMD_CUDA<VectorType> res(rows(), columns(), InitPolicyForStorage::eAllocAndSetToZero);
            res.matrixByCols = matrixByCols * factor;
            return res;
        }

        template<typename TFactorType>
        MatrixNMD_CUDA& operator *= (TFactorType factor)
        {
            matrixByCols *= TElementType(factor);
            return *this;
        }

        static MatrixNMD_CUDA computeDifferenceAndEvalL2Norm(const MatrixNMD_CUDA& a, const MatrixNMD_CUDA& b, TElementType& restrict_ext l2NormOfDifference)
        {                
            MatrixNMD_CUDA<VectorType> res(a);
            res.matrixByCols.computeDiffAndComputeL2Norm(b.matrixByCols, l2NormOfDifference);
            return res;          
        }

        template<typename Type>
        MatrixNMD_CUDA& addToAllDiagonalEntries(Type addToDiagonal)
        {
            assert(isSquareMatrix());            
            applyKernelToAddItemToDiagonal(matrixByCols.rawDevData(), matrixByCols.device(), LDA, columns(), addToDiagonal);
        }

        template<typename Type>
        MatrixNMD_CUDA& setAllDiagonalEntries(TElementType value)
        {
            assert(isSquareMatrix());            
            applyKernelToSetItemToDiagonal(matrixByCols.rawDevData(), matrixByCols.device(), LDA, columns(), value);
        }
        
        template <typename TFactorType>
        MatrixNMD_CUDA operator / (TFactorType factor) const
        {
            MatrixNMD_CUDA<VectorType> res(*this);
            res /= factor;
            return res;
        }

        template<typename TFactorType>
        MatrixNMD_CUDA& operator /= (TFactorType factor)
        {
            matrixByCols /= factor;
            return *this;
        }
        
        /** Compute matrix-matrix multiplication c = ab
        * @param a first matrix
        * @param b second matrix
        * @tparam kAlgoNumber4MatrixMatrixMult 0-Naive, 1, 2, 3, 4 - Cache Oblivious
        */
        static MatrixNMD_CUDA matrixMatrixMultiply(const MatrixNMD_CUDA& a, const MatrixNMD_CUDA& b)
        {
            assert(a.columns() == b.rows());

            MatrixNMD_CUDA res(a.rows(), b.columns(), InitPolicyForStorage::eAllocAndSetToZero);
            applyKernelToMatrixTimesMatrix(res.matrixByCols.device(),
                                           res.matrixByCols.rawDevData(),
                                           a.matrixByCols.rawDevData(),
                                           b.matrixByCols.rawDevData(),                                           
                                           a.rows(), a.columns(), a.LDA, b.columns(), b.LDA, res.LDA);
            return res;
        }
        
        MatrixNMD_CUDA operator * (const MatrixNMD_CUDA & x) const {
            return matrixMatrixMultiply(*this, x);
        }

        MatrixNMD_CUDA& operator *= (const MatrixNMD_CUDA& x)
        {
            MatrixNMD_CUDA<VectorType> mat_copy(*this);
            *this = mat_copy * x;
            return *this;
        }

        MatrixNMD_CUDA operator ^ (int power) const
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
                return MatrixNMD_CUDA::getIdentitySquareMatrix(rows());
            }
            else if (power % 2 == 0)
            {
                MatrixNMD_CUDA tmp = *this ^ (power / 2);
                return tmp * tmp;
            }
            else
            {
                MatrixNMD_CUDA tmp = (*this) ^ ((power - 1) / 2);
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
            return true;
        }

        /** Set all elements (excluding padding) randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        * @todo improve performance if there are any clients
        */
        template<class Generator>
        void setAllRandomly(Generator& generator)
        {
            matrixByCols.setAllRandomly(generator);
        }

        /** Set all elements (excluding padding) for specific value
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        void setAll(TElementType value)
        {
            matrixByCols.setAll(value);
        }

        /** Set all items of matrix to default value
        * @return reference to itself
        */
        MatrixNMD_CUDA& setAllToDefault() {
            matrixByCols.setAllToDefault();
            return *this;
        }

        /** Store(or dump) values from current vector into CPU/Virtual memory
        * @param valuesCopyTo storage in which values from vector will be copied into
        * @return this vector
        */
        template<class HostMatrix>
        MatrixNMD_CUDA& store(HostMatrix& valuesCopyTo)
        {
            if (valuesCopyTo.rows() != rows() || valuesCopyTo.columns() != columns())
                valuesCopyTo = HostMatrix(rows(), columns());

            assert(valuesCopyTo.rows() == rows());
            assert(valuesCopyTo.columns() == columns());

            GpuManagement dev(device());
            size_t myRows = rows();

            for (size_t j = 0, jDevOffset = 0; j < columns(); ++j, jDevOffset += LDA)
            {
                dev.copyDevice2HostSync(valuesCopyTo.data(0, j),
                                        matrixByCols.rawDevData() + jDevOffset,
                                        myRows);
            }
            
            return *this;
        }

        /** Load values into GPU vector
        * @param valuesCopy2vec values to copy into vector stored in GPU memory
        * @return reference vector
        */
        template<class HostMatrix>
        MatrixNMD_CUDA& load(const HostMatrix& valuesCopyFrom)
        {
            assert(valuesCopyFrom.rows() == rows());
            assert(valuesCopyFrom.columns() == columns());

            GpuManagement dev(device());
            size_t myRows = rows();
            
            for (size_t j = 0, jDevOffset = 0; j < columns(); ++j, jDevOffset += LDA)
            {
                dev.copyHost2DeviceSync(matrixByCols.rawDevData() + jDevOffset,
                                        valuesCopyFrom.data(0, j),
                                        myRows);
            }

            return *this;

        }
        template<class RandomGen>
        void applyNaturalCompressor(RandomGen& rndGen)
        {
            applyNaturalCompressor();
        }

        void applyNaturalCompressor() {
            applyKernelToMakeNaturalCompressor(matrixByCols.rawDevData(), matrixByCols.rawDevData(), size(), device());
        }

        void applyNaturalCompressorNaive()
        {
            size_t sz = size();
            VectorNDRaw<TElementType> hostVec(sz);
            
            matrixByCols.store(hostVec);
            
            for (size_t i = 0; i < sz; ++i)
            {
                auto pack = getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(hostVec[i]);
                pack.components.mantissa = 0;
                hostVec[i] = pack.real_value_repr;
            }
            
            matrixByCols.load(hostVec);
        }
        
    public:
        GpuManagement& device() {
            return matrixByCols.device();
        }
        
        const GpuManagement& device() const {
            return matrixByCols.device();
        }
    };

    template<class VectorType>
    MatrixNMD_CUDA<VectorType> operator * (typename VectorType::TElementType factor, const MatrixNMD_CUDA<VectorType>& x)
    {
        return x * factor;
    }
}
