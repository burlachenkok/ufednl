#include "dopt/gpu_compute_support/kernels/cuda/CUDASystemHelpers.h"

/*
* #include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/LinalgComputePreprocessing.h"
* #include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/VectorND_CUDA_Raw_kernels.h"
*/

#include "dopt/gpu_compute_support/kernels/cuda/linalg_matrices/MatrixNMD_CUDA_Raw_kernels.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_matrices/MatrixNMD_CUDA_Raw_kernels_templates.cuh"

#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stddef.h>

//==============================================================================================================================//
inline constexpr size_t defaultBlockSize() {
    return 256;
}
inline constexpr size_t defaultBlockSize1D() {
    return 32;
}
inline constexpr size_t defaultBlockSize2D() {
    return 32;
}
//==============================================================================================================================//

template <dopt::SingleArgUnaryOperation functionSelector, class T>
void applyKernelToDiagonalSingleArgTemplate(T* matrixByCols, dopt::GpuManagement& gpu, size_t LDA, size_t numRowsAndCols, T diagItem)
{
    if (numRowsAndCols == 0)
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((numRowsAndCols + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total/(kUnrollFactor*kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krApplyKernelToDiagonalSingleArgTemplate_1_VectorInput_and_1_scalar");
        krApplyKernelToDiagonalSingleArgTemplate_1_VectorInput_and_1_scalar<functionSelector, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (matrixByCols, LDA, numRowsAndCols, diagItem);
        gpu.notificationKernelWasLaunched("krApplyKernelToDiagonalSingleArgTemplate_1_VectorInput_and_1_scalar");
    }

    gpu.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToAddItemToDiagonal(double* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, double diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eAddValue>(matrixByCols, gpuForMatrix, LDA, numRowsAndCols, diagItem);
}
void dopt::applyKernelToAddItemToDiagonal(float* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, float diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eAddValue>(matrixByCols, gpuForMatrix, LDA, numRowsAndCols, diagItem);
}
void dopt::applyKernelToAddItemToDiagonal(uint32_t* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, uint32_t diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eAddValue>(matrixByCols, gpuForMatrix, LDA, numRowsAndCols, diagItem);
}
void dopt::applyKernelToAddItemToDiagonal(int32_t* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, int32_t diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eAddValue>(matrixByCols, gpuForMatrix, LDA, numRowsAndCols, diagItem);
}
//==============================================================================================================================//
void dopt::applyKernelToSetItemToDiagonal(double* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numRowsAndCols, double diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(matrixByCols, gpuForMatrix, LDA, numRowsAndCols, diagItem);
}
void dopt::applyKernelToSetItemToDiagonal(float* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numberOfRowsAndCols, float diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(matrixByCols, gpuForMatrix, LDA, numberOfRowsAndCols, diagItem);
}
void dopt::applyKernelToSetItemToDiagonal(uint32_t* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numberOfRowsAndCols, uint32_t diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(matrixByCols, gpuForMatrix, LDA, numberOfRowsAndCols, diagItem);
}
void dopt::applyKernelToSetItemToDiagonal(int32_t* matrixByCols, dopt::GpuManagement& gpuForMatrix, size_t LDA, size_t numberOfRowsAndCols, int32_t diagItem) {
    applyKernelToDiagonalSingleArgTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(matrixByCols, gpuForMatrix, LDA, numberOfRowsAndCols, diagItem);
}
//==============================================================================================================================//

template <class T>
void applyKernelToCreateTransposeTemplate(T* outMatrix, size_t outLDA,
                                          dopt::GpuManagement& outMatrixGPU, 
                                          const T* aMatInput, 
                                          size_t aLDA, 
                                          size_t aRows, 
                                          size_t aColumns)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(outMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;
    
    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows    + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid( cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY) );
    
    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = outMatrixGPU.currentDevice();

    {
        cudaStream_t s = outMatrixGPU.getCurrentStream();
        outMatrixGPU.selectThisGpu();
        outMatrixGPU.notificationKernelIsLaunching("krApplyTransposeNoSmem");
        krApplyTransposeNoSmem<kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s >>> (outMatrix, outLDA, aMatInput, aLDA, aRows, aColumns);
        outMatrixGPU.notificationKernelWasLaunched("krApplyTransposeNoSmem");
    }

    outMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToCreateTranspose(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToCreateTransposeTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToCreateTranspose(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToCreateTransposeTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToCreateTranspose(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToCreateTransposeTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToCreateTranspose(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToCreateTransposeTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, aRows, aColumns);
}
//==============================================================================================================================//


template <class T>
void applyKernelToMatrixOuterProductTemplate(T* outMatrix, size_t outLDA,
                                             dopt::GpuManagement& outMatrixGPU,
                                             const T* u, const T* v,
                                             size_t outRows, size_t outColumns)
{
    if (outRows == 0 || outColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(outMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((outColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((outRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = outMatrixGPU.currentDevice();

    {
        cudaStream_t s = outMatrixGPU.getCurrentStream();
        outMatrixGPU.selectThisGpu();
        outMatrixGPU.notificationKernelIsLaunching("krMatrixOuterProduct");
        krMatrixOuterProduct<kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s >>> (outMatrix, outLDA, u, v, outRows, outColumns);
        outMatrixGPU.notificationKernelWasLaunched("krMatrixOuterProduct");
    }

    outMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToMatrixOuterProduct(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* u, const double* v, size_t outRows, size_t outColumns) {
    applyKernelToMatrixOuterProductTemplate(outMatrix, outLDA, outMatrixGPU, u, v, outRows, outColumns);
}
void dopt::applyKernelToMatrixOuterProduct(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* u, const float* v, size_t outRows, size_t outColumns) {
    applyKernelToMatrixOuterProductTemplate(outMatrix, outLDA, outMatrixGPU, u, v, outRows, outColumns);
}
void dopt::applyKernelToMatrixOuterProduct(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* u, const uint32_t* v, size_t outRows, size_t outColumns) {
    applyKernelToMatrixOuterProductTemplate(outMatrix, outLDA, outMatrixGPU, u, v, outRows, outColumns);
}
void dopt::applyKernelToMatrixOuterProduct(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* u, const int32_t* v, size_t outRows, size_t outColumns) {
    applyKernelToMatrixOuterProductTemplate(outMatrix, outLDA, outMatrixGPU, u, v, outRows, outColumns);
}
//==============================================================================================================================//

template <class T>
void applyKernelToDiagTimesMatrixTemplate(T* outMatrix, size_t outLDA,
                                          dopt::GpuManagement& outMatrixGPU,
                                          const T* diag, const T* aMatInput,
                                          size_t aLDA, size_t aRows, size_t aColumns)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(outMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = outMatrixGPU.currentDevice();

    {
        cudaStream_t s = outMatrixGPU.getCurrentStream();
        outMatrixGPU.selectThisGpu();
        outMatrixGPU.notificationKernelIsLaunching("krApplyDiagTimesMatrix");
        krApplyDiagTimesMatrix<kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s >>> (outMatrix, outLDA, diag, aMatInput, aLDA, aRows, aColumns);
        outMatrixGPU.notificationKernelWasLaunched("krApplyDiagTimesMatrix");
    }

    outMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToDiagTimesMatrix(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* diag, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToDiagTimesMatrixTemplate(outMatrix, outLDA,  outMatrixGPU, diag, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToDiagTimesMatrix(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* diag, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToDiagTimesMatrixTemplate(outMatrix, outLDA, outMatrixGPU, diag, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToDiagTimesMatrix(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* diag, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToDiagTimesMatrixTemplate(outMatrix, outLDA, outMatrixGPU, diag, aMatInput, aLDA, aRows, aColumns);
}
void dopt::applyKernelToDiagTimesMatrix(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* diag, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns) {
    applyKernelToDiagTimesMatrixTemplate(outMatrix, outLDA, outMatrixGPU, diag, aMatInput, aLDA, aRows, aColumns);
}
//==============================================================================================================================//

template <class T>
void applyKernelToMatrixTimesDiagTemplate(T* outMatrix, size_t outLDA,
                                          dopt::GpuManagement& outMatrixGPU,
                                          const T* aMatInput,
                                          size_t aLDA,
                                          const T* diag,
                                          size_t aRows, size_t aColumns)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(outMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = outMatrixGPU.currentDevice();

    {
        cudaStream_t s = outMatrixGPU.getCurrentStream();
        outMatrixGPU.selectThisGpu();
        outMatrixGPU.notificationKernelIsLaunching("krApplyMatrixTimesDiag");
        krApplyMatrixTimesDiag<kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> << <grid, block, 0, s >> > (outMatrix, outLDA, aMatInput, aLDA, diag, aRows, aColumns);
        outMatrixGPU.notificationKernelWasLaunched("krApplyMatrixTimesDiag");
    }

    outMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToMatrixTimesDiag(double* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const double* aMatInput, size_t aLDA, const double* diag, size_t aRows, size_t aColumns) {
    applyKernelToMatrixTimesDiagTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, diag, aRows, aColumns);
}
void dopt::applyKernelToMatrixTimesDiag(float* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const float* aMatInput, size_t aLDA, const float* diag, size_t aRows, size_t aColumns) {
    applyKernelToMatrixTimesDiagTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, diag, aRows, aColumns);
}
void dopt::applyKernelToMatrixTimesDiag(uint32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const uint32_t* aMatInput, size_t aLDA, const uint32_t* diag, size_t aRows, size_t aColumns) {
    applyKernelToMatrixTimesDiagTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, diag, aRows, aColumns);
}
void dopt::applyKernelToMatrixTimesDiag(int32_t* outMatrix, size_t outLDA, dopt::GpuManagement& outMatrixGPU, const int32_t* aMatInput, size_t aLDA, const int32_t* diag, size_t aRows, size_t aColumns) {
    applyKernelToMatrixTimesDiagTemplate(outMatrix, outLDA, outMatrixGPU, aMatInput, aLDA, diag, aRows, aColumns);
}
//==============================================================================================================================//



template <class T>
void applyKernelToMatrixTimesMatrixTemplate(dopt::GpuManagement& abMatrixGPU, T* ab, const T* a, const T* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA)
{
    if (aRows == 0 || aColumns == 0 || bColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(abMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;
    constexpr size_t kRequiredSmem = 2 * kBlockSizeX * kBlockSizeY * sizeof(T);
    
    assert(abMatrixGPU.sharedMemoryPerBlock() >= kRequiredSmem);

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 1;
    constexpr size_t kUnrollFactorY = 1;

    size_t cols_to_process = ((bColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = abMatrixGPU.currentDevice();

    {
        cudaStream_t s = abMatrixGPU.getCurrentStream();
        abMatrixGPU.selectThisGpu();
        abMatrixGPU.notificationKernelIsLaunching("krMatrixMultiply");
        krMatrixMultiply<kBlockSizeX, kBlockSizeY> << <grid, block, 0, s >>> (ab, a, b, aRows, aColumns, aLDA, bColumns, bLDA, abLDA);
        abMatrixGPU.notificationKernelWasLaunched("krMatrixMultiply");
    }

    abMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU, double* ab, const double* a, const double* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA) {
    applyKernelToMatrixTimesMatrixTemplate(abMatrixGPU, ab, a, b, aRows, aColumns, aLDA, bColumns, bLDA, abLDA);
}
void dopt::applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU, float* ab, const float* a, const float* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA) {
    applyKernelToMatrixTimesMatrixTemplate(abMatrixGPU, ab, a, b, aRows, aColumns, aLDA, bColumns, bLDA, abLDA);
}
void dopt::applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU, uint32_t* ab, const uint32_t* a, const uint32_t* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA) {
    applyKernelToMatrixTimesMatrixTemplate(abMatrixGPU, ab, a, b, aRows, aColumns, aLDA, bColumns, bLDA, abLDA);
}
void dopt::applyKernelToMatrixTimesMatrix(dopt::GpuManagement& abMatrixGPU, int32_t* ab, const int32_t* a, const int32_t* b, size_t aRows, size_t aColumns, size_t aLDA, size_t bColumns, size_t bLDA, size_t abLDA) {
    applyKernelToMatrixTimesMatrixTemplate(abMatrixGPU, ab, a, b, aRows, aColumns, aLDA, bColumns, bLDA, abLDA);
}
//==============================================================================================================================//


template <class T>
void applyKernelToMatrixTimesVectorTemplate(dopt::GpuManagement& vecAbGPU,
                                            T* vecAb,
                                            const T* matA, size_t aRows, size_t aColumns, size_t aLDA,
                                            const T* vecB)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(vecAbGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 16;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 16;
    constexpr size_t kRequiredSmem = 2 * kBlockSizeX * kBlockSizeY * sizeof(T);

    assert(vecAbGPU.sharedMemoryPerBlock() >= kRequiredSmem);

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 1;
    constexpr size_t kUnrollFactorY = 1;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = vecAbGPU.currentDevice();

    {
        cudaStream_t s = vecAbGPU.getCurrentStream();
        vecAbGPU.selectThisGpu();
        vecAbGPU.notificationKernelIsLaunching("krMatrixVectorMultiply");
        krMatrixVectorMultiply<kBlockSizeX, kBlockSizeY> <<<grid, block, 0, s >>> (vecAb, matA, aRows, aColumns, aLDA, vecB);
        vecAbGPU.notificationKernelWasLaunched("krMatrixVectorMultiply");
    }

    vecAbGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, double* vecAb, const double* matA, size_t aRows, size_t aColumns, size_t aLDA, const double* vecB) {
    applyKernelToMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB);
}
void dopt::applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, float* vecAb, const float* matA, size_t aRows, size_t aColumns, size_t aLDA, const float* vecB) {
    applyKernelToMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB);
}
void dopt::applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, uint32_t* vecAb, const uint32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const uint32_t* vecB) {
    applyKernelToMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB);
}
void dopt::applyKernelToMatrixTimesVector(dopt::GpuManagement& vecAbGPU, int32_t* vecAb, const int32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const int32_t* vecB) {
    applyKernelToMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB);
}
//==============================================================================================================================//




template <class T>
void applyKernelToExtMatrixTimesVectorTemplate(dopt::GpuManagement& vecAbGPU,
                                               T* vecAb,
                                               const T* matA, size_t aRows, size_t aColumns, size_t aLDA,
                                               const T* vecB, T beta, const T* v)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(vecAbGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;
    constexpr size_t kRequiredSmem = 2 * kBlockSizeX * kBlockSizeY * sizeof(T);

    assert(vecAbGPU.sharedMemoryPerBlock() >= kRequiredSmem);

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 1;
    constexpr size_t kUnrollFactorY = 1;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = vecAbGPU.currentDevice();

    {
        cudaStream_t s = vecAbGPU.getCurrentStream();
        vecAbGPU.selectThisGpu();
        vecAbGPU.notificationKernelIsLaunching("krExtMatrixVectorMultiply");
        krExtMatrixVectorMultiply<kBlockSizeX, kBlockSizeY> <<<grid, block, 0, s >>> (vecAb, matA, aRows, aColumns, aLDA, vecB, beta, v);
        vecAbGPU.notificationKernelWasLaunched("krExtMatrixVectorMultiply");
    }

    vecAbGPU.selectGpu(prevDev);
}
//===============================================================================================================================================================================================================//

void dopt::applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, double* vecAb, const double* matA, size_t aRows, size_t aColumns, size_t aLDA, const double* vecB, double beta, const double* v) {
    applyKernelToExtMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB, beta, v);
}
void dopt::applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, float* vecAb, const float* matA, size_t aRows, size_t aColumns, size_t aLDA, const float* vecB, float beta, const float* v) {
    applyKernelToExtMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB, beta, v);
}
void dopt::applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, uint32_t* vecAb, const uint32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const uint32_t* vecB, uint32_t beta, const uint32_t* v) {
    applyKernelToExtMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB, beta, v);
}
void dopt::applyKernelToExtMatrixTimesVector(dopt::GpuManagement& vecAbGPU, int32_t* vecAb, const int32_t* matA, size_t aRows, size_t aColumns, size_t aLDA, const int32_t* vecB, int32_t beta, const int32_t* v) {
    applyKernelToExtMatrixTimesVectorTemplate(vecAbGPU, vecAb, matA, aRows, aColumns, aLDA, vecB, beta, v);
}
//===============================================================================================================================================================================================================//







template <class T>
void applyKernelToSymmetrizeLowerTriangInPlaceTemplate(dopt::GpuManagement& dev, T* matA, size_t rowsAndCols, size_t LDA)
{
    if (rowsAndCols == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(dev.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((rowsAndCols + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((rowsAndCols + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = dev.currentDevice();

    {
        cudaStream_t s = dev.getCurrentStream();
        dev.selectThisGpu();
        dev.notificationKernelIsLaunching("applyKernelToSymmetrizeLowerTriangInPlaceNoSmem");
        applyKernelToSymmetrizeLowerTriangInPlaceNoSmem<kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s >>> (matA, rowsAndCols, LDA);
        dev.notificationKernelWasLaunched("applyKernelToSymmetrizeLowerTriangInPlaceNoSmem");
    }

    dev.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, double* matA, size_t rowsAndCols, size_t LDA) {
    applyKernelToSymmetrizeLowerTriangInPlaceTemplate(dev, matA, rowsAndCols, LDA);
}
void dopt::applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, float* matA, size_t rowsAndCols, size_t LDA) {
    applyKernelToSymmetrizeLowerTriangInPlaceTemplate(dev, matA, rowsAndCols, LDA);
}
void dopt::applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, uint32_t* matA, size_t rowsAndCols, size_t LDA) {
    applyKernelToSymmetrizeLowerTriangInPlaceTemplate(dev, matA, rowsAndCols, LDA);
}
void dopt::applyKernelToSymmetrizeLowerTriangInPlace(dopt::GpuManagement& dev, int32_t* matA, size_t rowsAndCols, size_t LDA) {
    applyKernelToSymmetrizeLowerTriangInPlaceTemplate(dev, matA, rowsAndCols, LDA);
}


//==============================================================================================================================//
template <class T>
void applyMatrixCheckTemplate(bool* devNotSatisfyPredicate, dopt::MatrixTestApi test, dopt::GpuManagement& aMatrixGPU,
                              const T* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, T eps)
{
    if (aRows == 0 || aColumns == 0)
        return;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    // Pretty universal across all NVIDIA GPUs [Total blocksize irrespective of it's logical shape 1D/2D/2D]    
    constexpr size_t kMaxBlockSize = 1024;
    assert(aMatrixGPU.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;

    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 4;
    constexpr size_t kUnrollFactorY = 4;
    size_t cols_to_process = ((aColumns + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((aRows + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = aMatrixGPU.currentDevice();

    {
        cudaStream_t s = aMatrixGPU.getCurrentStream();
        aMatrixGPU.selectThisGpu();
        aMatrixGPU.notificationKernelIsLaunching("krApplyMatrixCheck");

        switch (test)
        {
        case dopt::MatrixTestApi::eIsDiagonal:
            krApplyMatrixCheck<KrMatrixTest::eKrIsDiagonal, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s>>> (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        case dopt::MatrixTestApi::eIsThreeDiagonal:
            krApplyMatrixCheck<KrMatrixTest::eKrIsThreeDiagonal, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s>>> (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        case dopt::MatrixTestApi::eIsLowerTriangular:
            krApplyMatrixCheck<KrMatrixTest::eKrIsLowerTriangular, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s>>> (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        case dopt::MatrixTestApi::eIsUpperTriangular:
            krApplyMatrixCheck<KrMatrixTest::eKrIsUpperTriangular, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s>>> (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        case dopt::MatrixTestApi::eIsZero:
            krApplyMatrixCheck<KrMatrixTest::eKrIsZero, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> <<<grid, block, 0, s>>> (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        case dopt::MatrixTestApi::eIsIdentity:
            krApplyMatrixCheck<KrMatrixTest::eKrIsIdentity, kBlockSizeX, kBlockSizeY, kUnrollFactorX, kUnrollFactorY> << <grid, block, 0, s >> > (devNotSatisfyPredicate, aMatInput, aLDA, aRows, aColumns, eps);
            break;
        default:
            assert(false);
            break;
        }
        
        aMatrixGPU.notificationKernelWasLaunched("krApplyMatrixCheck");
    }

    aMatrixGPU.selectGpu(prevDev);
}
//==============================================================================================================================//
void dopt::applyMatrixCheck(bool* devNotSatisfyPredicate, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const double* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, double eps) {
    applyMatrixCheckTemplate(devNotSatisfyPredicate, test, aMatrixGPU, aMatInput, aLDA, aRows, aColumns, eps);
}   
void dopt::applyMatrixCheck(bool* devNotSatisfyPredicate, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const float* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, float eps) {
    applyMatrixCheckTemplate(devNotSatisfyPredicate, test, aMatrixGPU, aMatInput, aLDA, aRows, aColumns, eps);
}
void dopt::applyMatrixCheck(bool* devNotSatisfyPredicate, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const uint32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, uint32_t eps) {
    applyMatrixCheckTemplate(devNotSatisfyPredicate, test, aMatrixGPU, aMatInput, aLDA, aRows, aColumns, eps);
}
void dopt::applyMatrixCheck(bool* devNotSatisfyPredicate, MatrixTestApi test, dopt::GpuManagement& aMatrixGPU, const int32_t* aMatInput, size_t aLDA, size_t aRows, size_t aColumns, int32_t eps) {
    applyMatrixCheckTemplate(devNotSatisfyPredicate, test, aMatrixGPU, aMatInput, aLDA, aRows, aColumns, eps);
}
//==============================================================================================================================//

template <class TIndex>
void applyKernelToFillInIndiciesForUpperTriangPartTemplate(TIndex* devIndicies, size_t rowsAndCols, dopt::GpuManagement& device)
{
    if (rowsAndCols == 0)
        return;

    constexpr size_t kMaxBlockSize = 1024;
    assert(device.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSizeX = kUseDefaultBlockSize ? defaultBlockSize1D() : 32;
    constexpr size_t kBlockSizeY = kUseDefaultBlockSize ? defaultBlockSize2D() : 32;
    static_assert(kBlockSizeX * kBlockSizeY <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactorX = 1;
    constexpr size_t kUnrollFactorY = 1;
    size_t cols_to_process = ((rowsAndCols + kUnrollFactorX * kBlockSizeX - 1) / (kUnrollFactorX * kBlockSizeX)) * (kUnrollFactorX * kBlockSizeX);
    size_t rows_to_process = ((rowsAndCols + kUnrollFactorY * kBlockSizeY - 1) / (kUnrollFactorY * kBlockSizeY)) * (kUnrollFactorY * kBlockSizeY);

    dim3 block(kBlockSizeX, kBlockSizeY);
    dim3 grid(cols_to_process / (kUnrollFactorX * kBlockSizeX), rows_to_process / (kUnrollFactorY * kBlockSizeY));

    assert(grid.x > 0);
    assert(grid.y > 0);

    dopt::GPUDevice prevDev = device.currentDevice();

    {
        cudaStream_t s = device.getCurrentStream();
        device.selectThisGpu();
        device.notificationKernelIsLaunching("krApplyKernelToFillInIndiciesForUpperTriangPartTemplate");
        krApplyKernelToFillInIndiciesForUpperTriangPartTemplate<kBlockSizeX, kBlockSizeY> <<<grid, block, 0, s >>> (devIndicies, rowsAndCols);
        device.notificationKernelWasLaunched("krApplyKernelToFillInIndiciesForUpperTriangPartTemplate");
    }

    device.selectGpu(prevDev);
}

void dopt::applyKernelToFillInIndiciesForUpperTriangPart(uint16_t* devIndicies, size_t rowsAndCols, dopt::GpuManagement& device) {
    applyKernelToFillInIndiciesForUpperTriangPartTemplate(devIndicies, rowsAndCols, device);
}

void dopt::applyKernelToFillInIndiciesForUpperTriangPart(uint32_t* devIndicies, size_t rowsAndCols, dopt::GpuManagement& device) {
    applyKernelToFillInIndiciesForUpperTriangPartTemplate(devIndicies, rowsAndCols, device);
}
