#pragma once

#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/LinalgComputePreprocessing.h"

#include <stdint.h>
#include <assert.h>

//================================================================================================================//
// Get gobal offset for element matrix at specific (row,col) positinon in matrix stored 
// by columns with Leading Dimension LDA, where LDA := is number of rows in matrix A including any memory padding
#define compFlattenIndexFromPosition(row, col, Lda) ((row) + (col)*(Lda))

//================================================================================================================//

/** Apply function to diagonal of matrix stored by columns with Leading Dimension LDA
 *  @param matrixByCols - pointer to matrix stored by columns
 *  @param LDA - leading dimension of matrix
 *  @param numberOfRowsAndCols - number of rows and columns in matrix
 *  @param arg - argument for function
 *  @tparam functionSelector - function selector to apply for diagonal items
 *  @tparam BLOCKS_SIZE - number of threads in block
 *  @tparam K_UNROLL_FACTOR - unroll factor for loop
 *  @tapram T - type of matrix elements
 */
template<dopt::SingleArgUnaryOperation functionSelector, 
         size_t BLOCKS_SIZE, 
         size_t K_UNROLL_FACTOR, 
         class T>
KR_KERNEL_ENTRY_FN void krApplyKernelToDiagonalSingleArgTemplate_1_VectorInput_and_1_scalar(T* __restrict matrixByCols,
                                                                                            size_t LDA, 
                                                                                            size_t numberOfRowsAndCols, 
                                                                                            T arg)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();
    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;
    
    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        int idx_unroll = idx + i * BLOCKS_SIZE;
        if (idx_unroll < numberOfRowsAndCols)
        {
            unsigned int globalOffset = compFlattenIndexFromPosition(idx_unroll, idx_unroll, LDA);

            switch (functionSelector)
            {
                case dopt::SingleArgUnaryOperation::eMultByValue:
                {
                    matrixByCols[globalOffset] *= arg;
                    break;
                }
                case dopt::SingleArgUnaryOperation::eDivByValue:
                {
                    matrixByCols[globalOffset] /= arg;
                    break;
                }
                case dopt::SingleArgUnaryOperation::eSetToValue:
                {
                    matrixByCols[globalOffset] = arg;
                    break;
                }
                case dopt::SingleArgUnaryOperation::eZeroOutItems:
                {
                    matrixByCols[globalOffset] = T();
                    break;
                }
                case dopt::SingleArgUnaryOperation::eAddValue:
                {
                    matrixByCols[globalOffset] += arg;
                    break;
                }
                default:
                {
                    assert(!"UNKNOWN FUNCTION TO APPLY");
                    break;
                }
            }
        }
    }
}
//================================================================================================================//


/* Symmetrize in-place matrix with taking elements from upper triangular part and put them into lower triangular part [NOT FULLY OPTIMIZED]
* @param matA pointer to input/output matrix
* @param rowsAndCols number of rows and columns in input matrix
* @param LDA leading dimension of input/ouput matrix
* @tparam BLOCKS_SIZE_1D number of threads in first block dimenion (columns)
* @tparam BLOCKS_SIZE_2D number of threads in second block dimenion (rows)
* @tparam K_UNROLL_FACTOR_1D number of elements to process in one iteration in first block dimenion (column direction)
* @tparam K_UNROLL_FACTOR_2D number of elements to process in one iteration in second block dimenion (rows direction)
* @tparam T element type
* @todo The mapping is easy todo, but in fact half of thread blocks just do nothing. Possible optimizations: add early test to discard some updates, more hard - find suitable matrix indexing.
*/
template<size_t BLOCKS_SIZE_1D,
         size_t BLOCKS_SIZE_2D,
         size_t K_UNROLL_FACTOR_1D,
         size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void applyKernelToSymmetrizeLowerTriangInPlaceNoSmem(T* __restrict matA, size_t rowsAndCols, size_t LDA)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4ReadA = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, LDA);
            unsigned int globalOffset4WriteOut = compFlattenIndexFromPosition(idx_unroll_1_col, idx_unroll_2_row, LDA);

            if (idx_unroll_2_row < rowsAndCols && idx_unroll_1_col < rowsAndCols)
            {
                if (idx_unroll_2_row < idx_unroll_1_col)
                {
                    matA[globalOffset4WriteOut] = matA[globalOffset4ReadA];
                }
            }
        }
    }
}
//================================================================================================================//



/* Compute matrix "a" transpose without using shared memory and store results in matrix "out"
* @param out pointer to output matrix
* @param outLDA leading dimension of output matrix
* @param aMatInput pointer to input matrix
* @param aLDA leading dimension of input matrix
* @param aRows number of rows in input matrix
* @param aColumns number of columns in input matrix
* @tparam BLOCKS_SIZE_1D number of threads in first block dimenion (columns)
* @tparam BLOCKS_SIZE_2D number of threads in second block dimenion (rows)
* @tparam K_UNROLL_FACTOR_1D number of elements to process in one iteration in first block dimenion (column direction)
* @tparam K_UNROLL_FACTOR_2D number of elements to process in one iteration in second block dimenion (rows direction)
* @tparam T element type
*/
template<size_t BLOCKS_SIZE_1D, 
         size_t BLOCKS_SIZE_2D, 
         size_t K_UNROLL_FACTOR_1D, 
         size_t K_UNROLL_FACTOR_2D, 
         class T>
KR_KERNEL_ENTRY_FN void krApplyTransposeNoSmem(T* __restrict out, size_t outLDA,
                                               const T* __restrict aMatInput, size_t aLDA,
                                               size_t aRows,
                                               size_t aColumns)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();
    
    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4ReadA    = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, aLDA);
            unsigned int globalOffset4WriteOut = compFlattenIndexFromPosition(idx_unroll_1_col, idx_unroll_2_row, outLDA);

            if (idx_unroll_2_row < aRows && idx_unroll_1_col < aColumns)
            {
                out[globalOffset4WriteOut] = aMatInput[globalOffset4ReadA];
            }
        }
    }
}
//================================================================================================================//

/** Matrix outer product kernel.
* @param out pointer to output matrix
* @param outLDA leading dimension of output matrix
* @param u pointer to the column vector u in outer product u x v
* @param v pointer to the row vector v in outer product u x v
* @param outRows number of rows in output matrix
* @param outColumns number of columns in output matrix
* @tparam BLOCKS_SIZE_1D number of threads in first block dimenion (columns)
* @tparam BLOCKS_SIZE_2D number of threads in second block dimenion (rows)
* @tparam K_UNROLL_FACTOR_1D number of elements to process in one iteration in first block dimenion (column direction)
* @tparam K_UNROLL_FACTOR_2D number of elements to process in one iteration in second block dimenion (rows direction)
* @tparam T element type
*/
template<size_t BLOCKS_SIZE_1D,
         size_t BLOCKS_SIZE_2D,
         size_t K_UNROLL_FACTOR_1D,
         size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void krMatrixOuterProduct(T* __restrict out, size_t outLDA, 
                                             const T* __restrict u, const T* __restrict v, 
                                             size_t outRows, size_t outColumns)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4Write = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, outLDA);

            if (idx_unroll_2_row < outRows && idx_unroll_1_col < outColumns)
            {
                out[globalOffset4Write] = u[idx_unroll_2_row] * v[idx_unroll_1_col];
            }
        }
    }
}
//================================================================================================================//

/** Apply special form of matrix multiplication - [diagonal] x [aMatInput].
* @param out pointer to output matrix
* @param outLDA leading dimension of output matrix
* @param diag pointer to the diagonal matrix (only diagonal items stored in a dense way). Number of items equal to the number of rows in the aMatInput
* @param aMatInput pointer to the input matrix
* @param aLDA leading dimension of input matrix
* @param aRows number of rows in input matrix
* @param aColumns number of columns in input matrix
*/
template<size_t BLOCKS_SIZE_1D,
         size_t BLOCKS_SIZE_2D,
         size_t K_UNROLL_FACTOR_1D,
         size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void krApplyDiagTimesMatrix(T* __restrict out, size_t outLDA,
                                               const T* __restrict diag,
                                               const T* __restrict aMatInput, size_t aLDA,
                                               size_t aRows,
                                               size_t aColumns)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4ReadA = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, aLDA);
            unsigned int globalOffset4WriteOut = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, outLDA);

            if (idx_unroll_2_row < aRows && idx_unroll_1_col < aColumns)
            {
                out[globalOffset4WriteOut] = aMatInput[globalOffset4ReadA] * diag[idx_unroll_2_row];
            }
        }
    }
}
//================================================================================================================//

/** Apply special form of matrix multiplication - [aMatInput] x [diagonal]
* @param out pointer to output matrix
* @param outLDA leading dimension of output matrix
* @param aMatInput pointer to the input matrix
* @param aLDA leading dimension of input matrix
* @param diag pointer to the diagonal matrix (only diagonal items stored in a dense way). Number of items equal to the number of columns in the aMatInput
* @param aRows number of rows in input matrix
* @param aColumns number of columns in input matrix
*/
template<size_t BLOCKS_SIZE_1D,
         size_t BLOCKS_SIZE_2D,
         size_t K_UNROLL_FACTOR_1D,
         size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void krApplyMatrixTimesDiag(T* __restrict out, size_t outLDA,
                                               const T* __restrict aMatInput, size_t aLDA,
                                               const T* __restrict diag,
                                               size_t aRows,
                                               size_t aColumns)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4ReadA = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, aLDA);
            unsigned int globalOffset4WriteOut = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, outLDA);

            if (idx_unroll_2_row < aRows && idx_unroll_1_col < aColumns)
            {
                out[globalOffset4WriteOut] = aMatInput[globalOffset4ReadA] * diag[idx_unroll_1_col];
            }
        }
    }
}
//================================================================================================================//

/** Matrix-matrix multiply kernel.
* @tparam BLOCKS_SIZE_1D number of threads in first block dimenion (columns in a)
* @tparam BLOCKS_SIZE_2D number of threads in second block dimenion (rows in a)
* @tparam K_UNROLL_FACTOR_1D number of elements to process in one iteration in first block dimenion (column direction)
* @tparam K_UNROLL_FACTOR_2D number of elements to process in one iteration in second block dimenion (rows direction)
* @tparam T element type
* @todo maybe add unrolling
*/
template<size_t BLOCKS_SIZE_1D_COLS, 
         size_t BLOCKS_SIZE_2D_ROWS,
         //size_t K_UNROLL_FACTOR_1D,
         //size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void krMatrixMultiply(T* __restrict ab, 
                                         const T* __restrict a, const T* __restrict b,
                                         size_t aRows, size_t aColumns, size_t aLDA,
                                         size_t bColumns, size_t bLDA,
                                         size_t abLDA)
{
    // aaaa    bb
    // aaaa x  bb
    // aaaa    bb
    //         bb
    const size_t bRows = aColumns;

    // create shorthand names
    const int tx = KR_LOCAL_THREAD_ID_1D();
    const int bx = KR_LOCAL_BLOCK_ID_1D();

    const int ty = KR_LOCAL_THREAD_ID_2D();
    const int by = KR_LOCAL_BLOCK_ID_2D();
    
    const size_t TILE_WIDTH_COLS = BLOCKS_SIZE_1D_COLS;
    const size_t TILE_WIDTH_ROWS = BLOCKS_SIZE_2D_ROWS;

    // allocate 2D tiles in __shared__ memory
    KR_SHARED_MEM_PREFIX T s_a[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];
    KR_SHARED_MEM_PREFIX T s_b[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];

    // calculate the row & column index of the element
    int col = bx * BLOCKS_SIZE_1D_COLS + tx;
    int row = by * BLOCKS_SIZE_2D_ROWS + ty;

    T result = T();

    // loop over the tiles of the input in phases
    int tilesSteps = (aColumns + TILE_WIDTH_COLS - 1) / TILE_WIDTH_COLS;
    
    for (int p = 0; p < tilesSteps; ++p)
    {
        // collaboratively load tiles into __shared__

        int localColumnA = p * TILE_WIDTH_COLS + tx;
        if (localColumnA < aColumns)
        {
            s_a[ty][tx] = a[compFlattenIndexFromPosition(row, localColumnA, aLDA)];
        }
        else
        {
            s_a[ty][tx] = T(0);
        }

        int localRowB = p * TILE_WIDTH_ROWS + ty;
        if (localRowB < bRows)
        {
            s_b[ty][tx] = b[compFlattenIndexFromPosition(localRowB, col, bLDA)];
        }
        else
        {
            s_b[ty][tx] = T(0);
        }

        // wait until all data is loaded before allowing
        // any thread in this block to continue
        KR_SYNC_THREAD_BLOCK();

        // do dot product between row of s_a and column of s_b
        for (int k = 0; k < TILE_WIDTH_COLS; ++k)
        {
            result += s_a[ty][k] * s_b[k][tx];
        }

        // wait until all threads are finished with the data
        // before allowing any thread in this block to continue
        KR_SYNC_THREAD_BLOCK();
    }

    // write out this thread's result
    if (row < aRows && col < bColumns)
    {
        ab[compFlattenIndexFromPosition(row, col, abLDA)] = result;
    }
}

//================================================================================================================//

/** Matrix-vector multiply kernel
*/
template<size_t BLOCKS_SIZE_1D_COLS,
        size_t BLOCKS_SIZE_2D_ROWS,
        //size_t K_UNROLL_FACTOR_1D,
        //size_t K_UNROLL_FACTOR_2D,
        class T>
KR_KERNEL_ENTRY_FN void krMatrixVectorMultiply(T* vecAb,
                                               const T* matA, size_t aRows, size_t aColumns, size_t aLDA,
                                               const T* vecB)
{
    static_assert(BLOCKS_SIZE_1D_COLS == 1024 || BLOCKS_SIZE_1D_COLS == 512 || BLOCKS_SIZE_1D_COLS == 256 ||
        BLOCKS_SIZE_1D_COLS == 128 || BLOCKS_SIZE_1D_COLS == 64 || BLOCKS_SIZE_1D_COLS == 32 ||
        BLOCKS_SIZE_1D_COLS == 16 || BLOCKS_SIZE_1D_COLS == 8,
        "BLOCKS_SIZE_1D_COLS must be 1024, 512, 256, 128, 64, 32, 16, 8");

    // aaaa    b
    // aaaa x  b
    // aaaa    b
    //         b

    const size_t bRows = aColumns;

    // create shorthand names
    const int tx = KR_LOCAL_THREAD_ID_1D();
    const int bx = KR_LOCAL_BLOCK_ID_1D();

    const int ty = KR_LOCAL_THREAD_ID_2D();
    const int by = KR_LOCAL_BLOCK_ID_2D();

    const size_t TILE_WIDTH_COLS = BLOCKS_SIZE_1D_COLS;
    const size_t TILE_WIDTH_ROWS = BLOCKS_SIZE_2D_ROWS;

    // allocate 2D tiles in __shared__ memory
    KR_SHARED_MEM_PREFIX T s_a[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];
    KR_SHARED_MEM_PREFIX T s_b[TILE_WIDTH_COLS];
    KR_SHARED_MEM_PREFIX T s_ab[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];
 
    // calculate the row & column index of the element
    int col = bx * BLOCKS_SIZE_1D_COLS + tx;
    int row = by * BLOCKS_SIZE_2D_ROWS + ty;

    T result = T();

    // loop over the tiles of the input in phases
    int tilesSteps = (aColumns + TILE_WIDTH_COLS - 1) / TILE_WIDTH_COLS;

    for (int p = 0; p < tilesSteps; ++p)
    {
        // collaboratively load tile for matrix A
        int localColA = p * TILE_WIDTH_COLS + tx;
        if (localColA < aColumns)
        {
            s_a[ty][tx] = matA[compFlattenIndexFromPosition(row, localColA, aLDA)];
        }
        else
        {
            s_a[ty][tx] = T(0);
        }
        
        // load b vector
        if (ty == 0)
        {
            int localRowB = p * TILE_WIDTH_ROWS + tx;
            if (localRowB < bRows)
            {
                s_b[tx] = vecB[localRowB];
            }
            else
            {
                s_b[tx] = T(0);
            }
        }

        // wait until all data is loaded before allowing any thread in this block to continue
        KR_SYNC_THREAD_BLOCK();
        s_ab[ty][tx] = s_a[ty][tx] * s_b[tx];

        KR_SYNC_THREAD_BLOCK();
        // Do need multiplication for dot product in form of reduction (in x dimension -- columns)
        // In-place reduction in shared memory
        if (TILE_WIDTH_COLS >= 1024 && tx < 512) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 512];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 512 && tx < 256) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 256];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 256 && tx < 128) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 128];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 128 && tx < 64) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 64];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 64 && tx < 32) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 32];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 32 && tx < 16) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 16];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 16 && tx < 8) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 8];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 8 && tx < 4) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 4];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 4 && tx < 2) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 2];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 2 && tx < 1) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 1];
        KR_SYNC_THREAD_BLOCK();

        result += s_ab[ty][tx];
    }

    // write out this thread's result
    if (tx == 0 && row < aRows)
    {
        vecAb[row] = result;
    }
}



//================================================================================================================//

/** Matrix-vector multiply kernel
*/
template<size_t BLOCKS_SIZE_1D_COLS,
    size_t BLOCKS_SIZE_2D_ROWS,
    //size_t K_UNROLL_FACTOR_1D,
    //size_t K_UNROLL_FACTOR_2D,
    class T>
KR_KERNEL_ENTRY_FN void krExtMatrixVectorMultiply(T* vecAb,
                                                  const T* matA, size_t aRows, size_t aColumns, size_t aLDA, const T* vecB,
                                                  T beta, const T* v)
{
    static_assert(BLOCKS_SIZE_1D_COLS == 1024 || BLOCKS_SIZE_1D_COLS == 512 || BLOCKS_SIZE_1D_COLS == 256 ||
                  BLOCKS_SIZE_1D_COLS == 128 || BLOCKS_SIZE_1D_COLS == 64 || BLOCKS_SIZE_1D_COLS == 32 ||
                  BLOCKS_SIZE_1D_COLS == 16 || BLOCKS_SIZE_1D_COLS == 8,
                  "BLOCKS_SIZE_1D_COLS must be 1024, 512, 256, 128, 64, 32, 16, 8");

    // aaaa    b
    // aaaa x  b
    // aaaa    b
    //         b

    const size_t bRows = aColumns;

    // create shorthand names
    const int tx = KR_LOCAL_THREAD_ID_1D();
    const int bx = KR_LOCAL_BLOCK_ID_1D();

    const int ty = KR_LOCAL_THREAD_ID_2D();
    const int by = KR_LOCAL_BLOCK_ID_2D();

    const size_t TILE_WIDTH_COLS = BLOCKS_SIZE_1D_COLS;
    const size_t TILE_WIDTH_ROWS = BLOCKS_SIZE_2D_ROWS;

    // allocate 2D tiles in __shared__ memory
    KR_SHARED_MEM_PREFIX T s_a[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];
    KR_SHARED_MEM_PREFIX T s_b[TILE_WIDTH_COLS];
    KR_SHARED_MEM_PREFIX T s_ab[TILE_WIDTH_ROWS][TILE_WIDTH_COLS];

    // calculate the row & column index of the element
    int col = bx * BLOCKS_SIZE_1D_COLS + tx;
    int row = by * BLOCKS_SIZE_2D_ROWS + ty;

    T result = beta * v[row];

    // loop over the tiles of the input in phases
    int tilesSteps = (aColumns + TILE_WIDTH_COLS - 1) / TILE_WIDTH_COLS;

    for (int p = 0; p < tilesSteps; ++p)
    {
        // collaboratively load tile for matrix A
        int localColA = p * TILE_WIDTH_COLS + tx;
        if (localColA < aColumns)
        {
            s_a[ty][tx] = matA[compFlattenIndexFromPosition(row, localColA, aLDA)];
        }
        else
        {
            s_a[ty][tx] = T(0);
        }

        // load b vector
        if (ty == 0)
        {
            int localRowB = p * TILE_WIDTH_ROWS + tx;
            if (localRowB < bRows)
            {
                s_b[tx] = vecB[localRowB];
            }
            else
            {
                s_b[tx] = T(0);
            }
        }

        // wait until all data is loaded before allowing any thread in this block to continue
        KR_SYNC_THREAD_BLOCK();

        s_ab[ty][tx] = s_a[ty][tx] * s_b[tx];

        // Do need multiplication for dot product in form of reduction (in x dimension -- columns)

        // In-place reduction in shared memory
        if (TILE_WIDTH_COLS >= 1024 && tx < 512) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 512];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 512 && tx < 256) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 256];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 256 && tx < 128) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 128];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 128 && tx < 64) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 64];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 64 && tx < 32) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 32];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 32 && tx < 16) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 16];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 16 && tx < 8) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 8];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 8 && tx < 4) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 4];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 4 && tx < 2) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 2];
        KR_SYNC_THREAD_BLOCK();

        if (TILE_WIDTH_COLS >= 2 && tx < 1) s_ab[ty][tx] = s_ab[ty][tx] + s_ab[ty][tx + 1];
        KR_SYNC_THREAD_BLOCK();

        result += s_ab[ty][tx];
    }

    // write out this thread's result
    if (tx == 0 && row < aRows)
    {
        vecAb[row] = result;
    }
}
//================================================================================================================//
template<class T>
KR_DEV_FN inline bool isApproxZero(T a, T eps)
{
    return a >= -eps && a <= eps;
}

template<class T>
KR_DEV_FN inline bool isApproxNotZero(T a, T eps)
{
    return a < -eps || a > eps;
}

KR_DEV_FN inline void unsafeWrite(bool *flag, bool value2set) {
    *flag = value2set;
}

enum KrMatrixTest
{
    eKrIsDiagonal,
    eKrIsThreeDiagonal,
    eKrIsLowerTriangular,
    eKrIsUpperTriangular,
    eKrIsZero,
    eKrIsIdentity
};

template<KrMatrixTest TEST,
         size_t BLOCKS_SIZE_1D,
         size_t BLOCKS_SIZE_2D,
         size_t K_UNROLL_FACTOR_1D,
         size_t K_UNROLL_FACTOR_2D,
         class T>
KR_KERNEL_ENTRY_FN void krApplyMatrixCheck(bool* __restrict notSatisfyPredicate,
                                           const T* __restrict aMatInput, size_t aLDA, size_t aRows, size_t aColumns, T eps)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int idx_1 = bid_1 * (BLOCKS_SIZE_1D * K_UNROLL_FACTOR_1D) + tid_1;
    int idx_2 = bid_2 * (BLOCKS_SIZE_2D * K_UNROLL_FACTOR_2D) + tid_2;

    #pragma unroll
    for (size_t i1 = 0; i1 < K_UNROLL_FACTOR_1D; ++i1)
    {
        int idx_unroll_1_col = idx_1 + i1 * BLOCKS_SIZE_1D;

        #pragma unroll
        for (size_t i2 = 0; i2 < K_UNROLL_FACTOR_2D; ++i2)
        {
            int idx_unroll_2_row = idx_2 + i2 * BLOCKS_SIZE_2D;

            unsigned int globalOffset4ReadA = compFlattenIndexFromPosition(idx_unroll_2_row, idx_unroll_1_col, aLDA);

            if (idx_unroll_2_row < aRows && idx_unroll_1_col < aColumns)
            {
                switch (TEST)
                {
                    case eKrIsDiagonal:
                    {
                        if (idx_unroll_2_row != idx_unroll_1_col)
                        {
                            if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }
                        break;
                    }

                    case eKrIsThreeDiagonal:
                    {   
                        if (idx_unroll_2_row == idx_unroll_1_col || idx_unroll_2_row + 1 == idx_unroll_1_col || idx_unroll_2_row == idx_unroll_1_col + 1)
                        {
                            // nothing to do
                        }
                        else
                        {
                            if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }
                        break;
                    }

                    case eKrIsLowerTriangular:
                    {
                        if (idx_unroll_2_row < idx_unroll_1_col)
                        {
                            if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }
                        break;
                    }

                    case eKrIsUpperTriangular:
                    {
                        if (idx_unroll_2_row > idx_unroll_1_col)
                        {
                            if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }
                        break;
                    }

                    case eKrIsZero:
                    {
                        if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                        {
                            unsafeWrite(notSatisfyPredicate, true);
                            return;
                        }
                        break;
                    }

                    case eKrIsIdentity:
                    {
                        if (idx_unroll_2_row == idx_unroll_1_col) // [1 - item] should be close to zero
                        {
                            if (isApproxNotZero(T(1) - aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }
                        else // (idx_unroll_2_row != idx_unroll_1_col) [should be close to zero]
                        {
                            if (isApproxNotZero(aMatInput[globalOffset4ReadA], eps))
                            {
                                unsafeWrite(notSatisfyPredicate, true);
                                return;
                            }
                        }

                        break;
                    }
                }            
            }
        }
    }
}
//================================================================================================================//

template<size_t BLOCKS_SIZE_1D, size_t BLOCKS_SIZE_2D, class TIndex>
KR_KERNEL_ENTRY_FN void krApplyKernelToFillInIndiciesForUpperTriangPartTemplate(TIndex* __restrict outIndicies, size_t rowsAndCols)
{
    int tid_1 = KR_LOCAL_THREAD_ID_1D();
    int bid_1 = KR_LOCAL_BLOCK_ID_1D();

    int tid_2 = KR_LOCAL_THREAD_ID_2D();
    int bid_2 = KR_LOCAL_BLOCK_ID_2D();

    int col = bid_1 * (BLOCKS_SIZE_1D) + tid_1;
    int row = bid_2 * (BLOCKS_SIZE_2D) + tid_2;
    
    if (col < rowsAndCols && row < rowsAndCols && col >= row)
    {
        size_t flatternPos = compFlattenIndexFromPosition(row, col, rowsAndCols);
        size_t writePos = (col * (col + 3)) / 2; // offset to diag it [col,col] from zero
        writePos -= (col-row);                   // decrease offset to diagonal because we are not yet
        
        /** Formula for writePos:
        *  for 1-based indexing: 1, 2, 3,... items are in 1,2,3,... columns
        *  therefore number of items up to colums, and row [c,c] including itself is c*(c+1)/2 [arithmetic series]
        *  in zero-based indexing for elements of flattened series c*(c+1)/2  correpons to c*(c+1)/2  - 2/2
        *  in zero-based indexing for columns (c+1)*(c+1+1)/2  - 2/2
        *   finally: [(c+1)*(c+1+1) - 2] /2 = [(c+1)^2 + c - 1]/2=(c*c+3c)/2=c*(c+3)/2 
        */

        outIndicies[writePos] = flatternPos;
    }
}
