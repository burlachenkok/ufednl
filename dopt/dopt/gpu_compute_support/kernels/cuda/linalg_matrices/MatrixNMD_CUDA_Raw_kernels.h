#pragma once

#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#include <stdint.h>

namespace dopt
{
    void applyKernelToAddItemToDiagonal(double* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, double diagItem);
    void applyKernelToAddItemToDiagonal(float* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, float diagItem);
    void applyKernelToAddItemToDiagonal(uint32_t* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, uint32_t diagItem);
    void applyKernelToAddItemToDiagonal(int32_t* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, int32_t diagItem);
    //===============================================================================================================================================================================================================//

    void applyKernelToSetItemToDiagonal(double* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, double diagItem);
    void applyKernelToSetItemToDiagonal(float* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, float diagItem);
    void applyKernelToSetItemToDiagonal(uint32_t* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, uint32_t diagItem);
    void applyKernelToSetItemToDiagonal(int32_t* matrixByCols, GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, int32_t diagItem);
    //===============================================================================================================================================================================================================//

    void applyKernelToCreateTranspose(double* outMatrix, size_t outLDA, GpuManagement& outMatrixGPU, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToCreateTranspose(float* outMatrix, size_t outLDA, GpuManagement& outMatrixGPU, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToCreateTranspose(uint32_t* outMatrix, size_t outLDA, GpuManagement& outMatrixGPU, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToCreateTranspose(int32_t* outMatrix, size_t outLDA, GpuManagement& outMatrixGPU, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    //===============================================================================================================================================================================================================//

    void applyKernelToMatrixOuterProduct(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* u, const double* v, size_t outRows, size_t outColumns);
    void applyKernelToMatrixOuterProduct(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* u, const float* v, size_t outRows, size_t outColumns);
    void applyKernelToMatrixOuterProduct(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* u, const uint32_t* v, size_t outRows, size_t outColumns);
    void applyKernelToMatrixOuterProduct(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* u, const int32_t* v, size_t outRows, size_t outColumns);
    //===============================================================================================================================================================================================================//

    void applyKernelToDiagTimesMatrix(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* diag, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToDiagTimesMatrix(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* diag, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToDiagTimesMatrix(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* diag, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    void applyKernelToDiagTimesMatrix(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* diag, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns);
    //===============================================================================================================================================================================================================//

    void applyKernelToMatrixTimesDiag(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* aMatInput, size_t aLDA, const double* diag, size_t aRows, size_t aColumns);
    void applyKernelToMatrixTimesDiag(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* aMatInput, size_t aLDA, const float* diag, size_t aRows, size_t aColumns);
    void applyKernelToMatrixTimesDiag(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* aMatInput, size_t aLDA, const uint32_t* diag, size_t aRows, size_t aColumns);
    void applyKernelToMatrixTimesDiag(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* aMatInput, size_t aLDA, const int32_t* diag, size_t aRows, size_t aColumns);
    //===============================================================================================================================================================================================================//

    void applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU,  double* ab, const double* a, const double* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA);
    void applyKernelToMatrixTimesMatrix(dopt::GpuManagement & abMatrixGPU, float* ab, const float* a, const float* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA);
    void applyKernelToMatrixTimesMatrix(dopt::GpuManagement & abMatrixGPU, uint32_t* ab, const uint32_t * a, const uint32_t * b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA);
    void applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU, int32_t* ab, const int32_t* a, const int32_t* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA);
    //===============================================================================================================================================================================================================//

    void applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, double* vecAb, const double* matA, size_t aRows, size_t aColumns, size_t aLDA, const double* vecB);
    void applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, float* vecAb, const float* matA, size_t aRows, size_t aColumns, size_t aLDA, const float* vecB);
    void applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, uint32_t* vecAb, const uint32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const uint32_t* vecB);
    void applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, int32_t* vecAb, const int32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const int32_t* vecB);
    //===============================================================================================================================================================================================================//

    void applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, double* vecAb, const double* matA, size_t aRows, size_t aColumns, size_t aLDA, const double* vecB, double beta, const double* v);
    void applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, float* vecAb, const float* matA, size_t aRows, size_t aColumns, size_t aLDA, const float* vecB, float beta, const float* v);
    void applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, uint32_t* vecAb, const uint32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const uint32_t* vecB, uint32_t beta, const uint32_t* v);
    void applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, int32_t* vecAb, const int32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const int32_t* vecB, int32_t beta, const int32_t* v);
    //===============================================================================================================================================================================================================//

    void applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, double* matA, size_t rowsAndCols, size_t LDA);
    void applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, float* matA, size_t rowsAndCols, size_t LDA);
    void applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, uint32_t* matA, size_t rowsAndCols, size_t LDA);
    void applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, int32_t* matA, size_t rowsAndCols, size_t LDA);
    //===============================================================================================================================================================================================================//

    enum class MatrixTestApi
    {
        eIsDiagonal,        ///< Check that input matrix is diagonal
        eIsThreeDiagonal,   ///< Check that input matrix is three - diagonal
        eIsLowerTriangular, ///< Check that input matrix is lower triangular
        eIsUpperTriangular, ///< Check that input matrix is upper triangular
        eIsZero,            ///< Check that input matrix is zero
        eIsIdentity         ///< Check that input matrix is identity, i.e. matrix which has zero everywhere, except diagonal which has "1"
    };
    
    void applyMatrixCheck(bool* devNotSatisfy, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, double eps);
    void applyMatrixCheck(bool* devNotSatisfy, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, float eps);
    void applyMatrixCheck(bool* devNotSatisfy, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, uint32_t eps);
    void applyMatrixCheck(bool* devNotSatisfy, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, int32_t eps);
    //===============================================================================================================================================================================================================//
    //===============================================================================================================================================================================================================//
    void applyKernelToFillInIndiciesForUpperTriangPart(uint16_t* devIndicies, size_t rowsAndCols, dopt::GpuManagement& dev);
    void applyKernelToFillInIndiciesForUpperTriangPart(uint32_t* devIndicies, size_t rowsAndCols, dopt::GpuManagement& dev);
    //===============================================================================================================================================================================================================//
}
