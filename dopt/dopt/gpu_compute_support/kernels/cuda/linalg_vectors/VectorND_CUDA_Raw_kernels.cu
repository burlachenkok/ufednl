#include "dopt/gpu_compute_support/kernels/cuda/CUDASystemHelpers.h"

#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/LinalgComputePreprocessing.h"

#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/VectorND_CUDA_Raw_kernels.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/VectorND_CUDA_Raw_kernels_templates.cuh"

#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stddef.h>
#include <assert.h>

//==========================================================================================//
inline constexpr size_t defaultBlockSize() {
    return 256;
}
//==========================================================================================//

template <dopt::BinaryReductionOperation reductionFunction, 
          dopt::PreprocessForBinaryReductionOperation itemPreprocessingFunction, 
          class T>
void applyKernelToEvaluateReductionTemplate(T* devOut, const T* devIn, size_t length, dopt::GpuManagement& gpu)
{
    if (length == 0)
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total/(kUnrollFactor*kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krReductionWith_1_VectorInput");
        krReductionWith_1_VectorInput<reductionFunction, itemPreprocessingFunction, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devOut, devIn, length);
        gpu.notificationKernelWasLaunched("krReductionWith_1_VectorInput");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToEvaluateMax(double* devOut, const double* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMax, dopt::PreprocessForBinaryReductionOperation::eIdentity> (devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMax(float* devOut, const float* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMax, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMax(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMax, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMax(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMax, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}


void dopt::applyKernelToEvaluateMin(double* devOut, const double* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMin, dopt::PreprocessForBinaryReductionOperation::eIdentity> (devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMin(float* devOut, const float* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMin, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMin(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMin, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateMin(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eMin, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}


void dopt::applyKernelToEvaluateSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eIdentity> (devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eIdentity>(devOut, devIn, items, gpu);
}


void dopt::applyKernelToEvaluateNnz(double* devResultAccum, double* devPtr, size_t length, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eCard>(devResultAccum, devPtr, length, gpu);
}
void dopt::applyKernelToEvaluateNnz(float* devResultAccum, float* devPtr, size_t length, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eCard>(devResultAccum, devPtr, length, gpu);
}
void dopt::applyKernelToEvaluateNnz(uint32_t* devResultAccum, uint32_t* devPtr, size_t length, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eCard>(devResultAccum, devPtr, length, gpu);
}
void dopt::applyKernelToEvaluateNnz(int32_t* devResultAccum, int32_t* devPtr, size_t length, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eCard>(devResultAccum, devPtr, length, gpu);
}

void dopt::applyKernelToEvaluateLogisticLossFromMarginSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromMargin>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromMargin>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromMargin>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromMargin>(devOut, devIn, items, gpu);
}


void dopt::applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromSigmoid>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromSigmoid>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromSigmoid>(devOut, devIn, items, gpu);
}
void dopt::applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu) {
    applyKernelToEvaluateReductionTemplate<dopt::BinaryReductionOperation::eSum, dopt::PreprocessForBinaryReductionOperation::eLogisticLossFromSigmoid>(devOut, devIn, items, gpu);
}


//=========================================================================================================//

template <dopt::UnaryOperation functionSelector, class T>
void applyKernelToUnaryFunctionTemplate(T* devOut, const T* devIn, size_t length, dopt::GpuManagement& gpu, dopt::OperationFlags flags)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krSimpleEwFunction_1_VectorInput");
        
        if (flags == dopt::OperationFlags::eOperationNone)
        {
            krSimpleEwFunction_1_VectorInput<functionSelector, kBlockSize, kUnrollFactor, 
                                             dopt::OperationFlags::eOperationNone> <<<grid, block, 0, s >>> (devOut, devIn, length);
        }
        else if (flags == dopt::OperationFlags::eMakeOperationAtomic)
        {
            krSimpleEwFunction_1_VectorInput<functionSelector, kBlockSize, kUnrollFactor, 
                                             dopt::OperationFlags::eMakeOperationAtomic> <<<grid, block, 0, s >>> (devOut, devIn, length);
        }
            
        gpu.notificationKernelWasLaunched("krSimpleEwFunction_1_VectorInput");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToEvaluateAbsItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAbsEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateAbsItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAbsEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateAbsItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAbsEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateAbsItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAbsEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateExpItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eExpEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateExpItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eExpEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateExpItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eExpEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateExpItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eExpEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateLogItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eLogEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateLogItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eLogEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateLogItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eLogEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateLogItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eLogEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateInvItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateSquareItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSquareItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSquareItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSquareItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateSqrtItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSqrtEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSqrtItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSqrtEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSqrtItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSqrtEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSqrtItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSqrtEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateInvSquareItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvSquareItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvSquareItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateInvSquareItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eInvSquareEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateSigmoidFn(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSigmoidEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSigmoidFn(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSigmoidEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSigmoidFn(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSigmoidEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSigmoidFn(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSigmoidEw>(devOut, devIn, length, gpuManagement, OperationFlags::eOperationNone);
}

 
void dopt::applyKernelToEvaluateNegItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eNegEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateNegItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eNegEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateNegItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eNegEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateNegItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eNegEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateAppendItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu, OperationFlags flags) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, gpu, flags);
}
void dopt::applyKernelToEvaluateAppendItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu, OperationFlags flags) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, gpu, flags);
}
void dopt::applyKernelToEvaluateAppendItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu, OperationFlags flags) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, gpu, flags);
}
void dopt::applyKernelToEvaluateAppendItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu, OperationFlags flags) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, gpu, flags);
}


void dopt::applyKernelToEvaluateSubItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSubEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSubItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSubEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSubItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSubEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateSubItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eSubEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateMultiplyItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eMulEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateMultiplyItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eMulEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateMultiplyItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eMulEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateMultiplyItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eMulEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}


void dopt::applyKernelToEvaluateDivItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eDivEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateDivItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eDivEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateDivItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eDivEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}
void dopt::applyKernelToEvaluateDivItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToUnaryFunctionTemplate<dopt::UnaryOperation::eDivEw>(devOut, devIn, length, gpu, OperationFlags::eOperationNone);
}


//=============================================================================================================//
template <class T>
void applyKernelToEvaluateSignOfItemsTemplate(T* devOut, const T* devIn, size_t length, 
                                              const T posSignValue, const T negSignValue, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024;// Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krSimpleEwFunction_1_VectorInput_and_2_scalars");
        krSimpleEwFunction_1_VectorInput_and_2_scalars<kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devOut, devIn, length, posSignValue, negSignValue);
        gpu.notificationKernelWasLaunched("krSimpleEwFunction_1_VectorInput_and_2_scalars");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToEvaluateSignOfItems(double* devOut, const double* devIn, size_t length, const double posSignValue, const double negSignValue, GpuManagement& gpu) {
    applyKernelToEvaluateSignOfItemsTemplate(devOut, devIn, length, posSignValue, negSignValue, gpu);
}
void dopt::applyKernelToEvaluateSignOfItems(float* devOut, const float* devIn, size_t length, const float posSignValue, const float negSignValue, GpuManagement& gpu) {
    applyKernelToEvaluateSignOfItemsTemplate(devOut, devIn, length, posSignValue, negSignValue, gpu);
}
void dopt::applyKernelToEvaluateSignOfItems(uint32_t* devOut, const uint32_t* devIn, size_t length, const uint32_t posSignValue, const uint32_t negSignValue, GpuManagement& gpu) {
    applyKernelToEvaluateSignOfItemsTemplate(devOut, devIn, length, posSignValue, negSignValue, gpu);
}
void dopt::applyKernelToEvaluateSignOfItems(int32_t* devOut, const int32_t* devIn, size_t length, const int32_t posSignValue, const int32_t negSignValue, GpuManagement& gpu) {
    applyKernelToEvaluateSignOfItemsTemplate(devOut, devIn, length, posSignValue, negSignValue, gpu);
}
//==================================================================================================================//

template <dopt::SingleArgUnaryOperation functionSelector, 
          class T>
void applyKernelToSingleArgFunctionTemplate(T* devInOut, size_t length, T arg, dopt::GpuManagement& gpu, dopt::OperationFlags mode)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024;// Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krSimpleEwFunction_1_VectorInput_and_1_scalar");

        if (mode == dopt::OperationFlags::eOperationNone)
        {
            krSimpleEwFunction_1_VectorInput_and_1_scalar<functionSelector, kBlockSize, kUnrollFactor, dopt::OperationFlags::eOperationNone> <<<grid, block, 0, s >>> (devInOut, arg, length);
        }
        else if (mode == dopt::OperationFlags::eMakeOperationAtomic)
        {
            krSimpleEwFunction_1_VectorInput_and_1_scalar<functionSelector, kBlockSize, kUnrollFactor, dopt::OperationFlags::eMakeOperationAtomic> <<<grid, block, 0, s >>> (devInOut, arg, length);
        }
        gpu.notificationKernelWasLaunched("krSimpleEwFunction_1_VectorInput_and_1_scalar");
    }

    gpu.selectGpu(prevDev);
}


void dopt::applyKernelToSetAllItemsToValue(double* devOut, size_t length, double value, GpuManagement& gpu) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eSetToValue> (devOut, length, value, gpu, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToSetAllItemsToValue(float* devOut, size_t length, float value, GpuManagement& gpu) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(devOut, length, value, gpu, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToSetAllItemsToValue(uint32_t* devOut, size_t length, uint32_t value, GpuManagement& gpu) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(devOut, length, value, gpu, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToSetAllItemsToValue(int32_t* devOut, size_t length, int32_t value, GpuManagement& gpu) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eSetToValue>(devOut, length, value, gpu, dopt::OperationFlags::eOperationNone);
}

void dopt::applyKernelToEvaluateMutiplyItemsByFactor(double* devInOut, size_t length, double factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eMultByValue> (devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateMutiplyItemsByFactor(float* devInOut, size_t length, float factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eMultByValue>(devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateMutiplyItemsByFactor(uint32_t* devInOut, size_t length, uint32_t factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eMultByValue>(devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateMutiplyItemsByFactor(int32_t* devInOut, size_t length, int32_t factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eMultByValue>(devInOut, length, factor, gpu, mode);
}


void dopt::applyKernelToEvaluateDivItemsByFactor(double* devInOut, size_t length, double factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eDivByValue>(devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateDivItemsByFactor(float* devInOut, size_t length, float factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eDivByValue>(devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateDivItemsByFactor(uint32_t* devInOut, size_t length, uint32_t factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eDivByValue>(devInOut, length, factor, gpu, mode);
}
void dopt::applyKernelToEvaluateDivItemsByFactor(int32_t* devInOut, size_t length, int32_t factor, dopt::GpuManagement& gpu, dopt::OperationFlags mode) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eDivByValue>(devInOut, length, factor, gpu, mode);
}

void dopt::applyKernelToZeroOutItems(double* devInOut, size_t length, double eps, GpuManagement& gpuManagement) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eZeroOutItems>(devInOut, length, eps, gpuManagement, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToZeroOutItems(float* devInOut, size_t length, float eps, GpuManagement& gpuManagement) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eZeroOutItems>(devInOut, length, eps, gpuManagement, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToZeroOutItems(uint32_t* devInOut, size_t length, uint32_t eps, GpuManagement& gpuManagement) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eZeroOutItems>(devInOut, length, eps, gpuManagement, dopt::OperationFlags::eOperationNone);
}
void dopt::applyKernelToZeroOutItems(int32_t* devInOut, size_t length, int32_t eps, GpuManagement& gpuManagement) {
    applyKernelToSingleArgFunctionTemplate<dopt::SingleArgUnaryOperation::eZeroOutItems>(devInOut, length, eps, gpuManagement, dopt::OperationFlags::eOperationNone);
}

//==========================================================================================================================

template <dopt::BinaryReductionOperation reductionFunction, 
          class T>
void applyKernelToReducedDotProductTemplate(T* devOut, T* devIn1, T* devIn2, size_t length, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024;// Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krReductionWith_2_VectorInput");
        krReductionWith_2_VectorInput<reductionFunction, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devOut, devIn1, devIn2, length);
        gpu.notificationKernelWasLaunched("krReductionWith_2_VectorInput");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToReducedDotProduct(double* devOut, double* devIn1, double* devIn2, size_t length, GpuManagement& gpu) {
    applyKernelToReducedDotProductTemplate<dopt::BinaryReductionOperation::eSum> (devOut, devIn1, devIn2, length, gpu);
}
void dopt::applyKernelToReducedDotProduct(float* devOut, float* devIn1, float* devIn2, size_t length, GpuManagement& gpu) {
    applyKernelToReducedDotProductTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn1, devIn2, length, gpu);
}
void dopt::applyKernelToReducedDotProduct(uint32_t* devOut, uint32_t* devIn1, uint32_t* devIn2, size_t length, GpuManagement& gpu) {
    applyKernelToReducedDotProductTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn1, devIn2, length, gpu);
}
void dopt::applyKernelToReducedDotProduct(int32_t* devOut, int32_t* devIn1, int32_t* devIn2, size_t length, GpuManagement& gpu) {
    applyKernelToReducedDotProductTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn1, devIn2, length, gpu);
}
//=========================================================================================================================//

template <dopt::BinaryReductionOperation reductionFunction, 
          class T>
void applyKernelToEvaluateLpNormHelperTemplate(T* devOut, T* devIn, size_t length, dopt::GpuManagement& gpu, uint32_t p)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krApplyKernelToEvaluateLpNormHelperTemplate");
        krApplyKernelToEvaluateLpNormHelperTemplate<reductionFunction, kBlockSize, kUnrollFactor> <<< grid, block, 0, s >>> (devOut, devIn, length, p);
        gpu.notificationKernelWasLaunched("krApplyKernelToEvaluateLpNormHelperTemplate");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToEvaluateLpNormToPowerP(double* devOut, double* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eSum> (devOut, devIn, length, gpuManagement, pnorm);
}
void dopt::applyKernelToEvaluateLpNormToPowerP(float* devOut, float* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn, length, gpuManagement, pnorm);
}
void dopt::applyKernelToEvaluateLpNormToPowerP(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn, length, gpuManagement, pnorm);
}
void dopt::applyKernelToEvaluateLpNormToPowerP(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eSum>(devOut, devIn, length, gpuManagement, pnorm);
}


void dopt::applyKernelToEvaluateLInfNorm(double* devOut, double* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eMax>(devOut, devIn, length, gpuManagement, 1);
}
void dopt::applyKernelToEvaluateLInfNorm(float* devOut, float* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eMax>(devOut, devIn, length, gpuManagement, 1);
}
void dopt::applyKernelToEvaluateLInfNorm(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eMax>(devOut, devIn, length, gpuManagement, 1);
}
void dopt::applyKernelToEvaluateLInfNorm(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpuManagement) {
    applyKernelToEvaluateLpNormHelperTemplate<dopt::BinaryReductionOperation::eMax>(devOut, devIn, length, gpuManagement, 1);
}
//=========================================================================================================================//

template <dopt::TwoArgUnaryOperation functionSelector, 
          class T>
void applyKernelToTwoArgFunctionTemplate(T* devInOut, size_t length, T arg1, T arg2, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krApplyKernelToTwoArgFunctionTemplate");
        krApplyKernelToTwoArgFunctionTemplate<functionSelector, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devInOut, arg1, arg2, length);
        gpu.notificationKernelWasLaunched("krApplyKernelToTwoArgFunctionTemplate");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToClampItems(double* devInOut, size_t length, double lower, double upper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, lower, upper, gpuManagement);
}
void dopt::applyKernelToClampItems(float* devInOut, size_t length, float lower, float upper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, lower, upper, gpuManagement);
}
void dopt::applyKernelToClampItems(uint32_t* devInOut, size_t length, uint32_t lower, uint32_t upper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, lower, upper, gpuManagement);
}
void dopt::applyKernelToClampItems(int32_t* devInOut, size_t length, int32_t lower, int32_t upper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, lower, upper, gpuManagement);
}


void dopt::applyKernelToScaledDifferenceWithEye(double* devInOut, size_t length, double a, double multiple, GpuManagement& gpu) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eScaledDiff>(devInOut, length, a, multiple, gpu);
}
void dopt::applyKernelToScaledDifferenceWithEye(float* devInOut, size_t length, float a, float multiple, GpuManagement& gpu) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eScaledDiff>(devInOut, length, a, multiple, gpu);
}
void dopt::applyKernelToScaledDifferenceWithEye(uint32_t* devInOut, size_t length, uint32_t a, uint32_t multiple, GpuManagement& gpu) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eScaledDiff>(devInOut, length, a, multiple, gpu);
}
void dopt::applyKernelToScaledDifferenceWithEye(int32_t* devInOut, size_t length, int32_t a, int32_t multiple, GpuManagement& gpu) {
    applyKernelToTwoArgFunctionTemplate<dopt::TwoArgUnaryOperation::eScaledDiff>(devInOut, length, a, multiple, gpu);
}
//=====================================================================================================================

template <dopt::TwoArgUnaryOperation functionSelector, 
          class T>
void applyKernelToTwoArgFunctionVectorizedTemplate(T* devInOut, size_t length, T* devArg1, T* devArg2, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krApplyKernelToTwoArgFunctionTemplate");
        krApplyKernelToTwoArgFunctionVectorizedTemplate<functionSelector, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devInOut, devArg1, devArg2, length);
        gpu.notificationKernelWasLaunched("krApplyKernelToTwoArgFunctionTemplate");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToClampItemsVectorized(double* devInOut, size_t length, double* devLower, double* devUpper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionVectorizedTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, devLower, devUpper, gpuManagement);
}
void dopt::applyKernelToClampItemsVectorized(float* devInOut, size_t length, float* devLower, float* devUpper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionVectorizedTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, devLower, devUpper, gpuManagement);
}
void dopt::applyKernelToClampItemsVectorized(uint32_t* devInOut, size_t length, uint32_t* devLower, uint32_t* devUpper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionVectorizedTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, devLower, devUpper, gpuManagement);
}
void dopt::applyKernelToClampItemsVectorized(int32_t* devInOut, size_t length, int32_t* devLower, int32_t* devUpper, GpuManagement& gpuManagement) {
    applyKernelToTwoArgFunctionVectorizedTemplate<dopt::TwoArgUnaryOperation::eClamp>(devInOut, length, devLower, devUpper, gpuManagement);
}
//========================================================================================================================

template <dopt::UnaryOperation functionSelector, 
          class T>
void applyKernelToUnaryFunctionWithExtraMultiplierTemplate(T* devOut, const T* devIn, size_t length, T extraMultiplier, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;
    
    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);
    
    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krSimpleEwFunctionWithMultiplier_1_VectorInput_1_scalar");
        krSimpleEwFunctionWithMultiplier_1_VectorInput_1_scalar<functionSelector, kBlockSize, kUnrollFactor> <<<grid, block, 0, s >>> (devOut, devIn, length, extraMultiplier);
        gpu.notificationKernelWasLaunched("krSimpleEwFunctionWithMultiplier_1_VectorInput_1_scalar");
    }

    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToEvaluateAppendItemsWithMultiplier(double* devOut, double* devIn, size_t length, double multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAppendEw> (devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAppendItemsWithMultiplier(float* devOut, float* devIn, size_t length, float multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAppendItemsWithMultiplier(uint32_t* devOut, uint32_t* devIn, size_t length, uint32_t multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAppendItemsWithMultiplier(int32_t* devOut, int32_t* devIn, size_t length, int32_t multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAppendEw>(devOut, devIn, length, multiple, gpu);
}

void dopt::applyKernelToEvaluateAssignItemsWithMultiplier(double* devOut, double* devIn, size_t length, double multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAssignEw>(devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAssignItemsWithMultiplier(float* devOut, float* devIn, size_t length, float multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAssignEw>(devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAssignItemsWithMultiplier(uint32_t* devOut, uint32_t* devIn, size_t length, uint32_t multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAssignEw>(devOut, devIn, length, multiple, gpu);
}
void dopt::applyKernelToEvaluateAssignItemsWithMultiplier(int32_t* devOut, int32_t* devIn, size_t length, int32_t multiple, GpuManagement& gpu) {
    applyKernelToUnaryFunctionWithExtraMultiplierTemplate<dopt::UnaryOperation::eAssignEw>(devOut, devIn, length, multiple, gpu);
}
//==============================================================================================================================//


template <class T>
void applyKernelToMakeNaturalCompressorTemplate(T* devOut, const T* devIn, size_t length, dopt::GpuManagement& gpu)
{
    if (length == 0) [[unlikely]]
        return;

    constexpr size_t kMaxBlockSize = 1024; // Pretty universal across all NVIDIA GPUs
    assert(gpu.maxThreadsPerBlock() == kMaxBlockSize);

    constexpr bool kUseDefaultBlockSize = true;
    constexpr size_t kBlockSize = kUseDefaultBlockSize ? defaultBlockSize() : 256;
    static_assert(kBlockSize <= kMaxBlockSize, "kBlockSize must be less than or equal to kMaxBlockSize");

    constexpr size_t kUnrollFactor = 4;
    size_t items_to_process_total = ((length + kUnrollFactor * kBlockSize - 1) / (kUnrollFactor * kBlockSize)) * (kUnrollFactor * kBlockSize);
    dim3 block(kBlockSize);
    dim3 grid(items_to_process_total / (kUnrollFactor * kBlockSize));
    assert(grid.x > 0);

    dopt::GPUDevice prevDev = gpu.currentDevice();

    {
        cudaStream_t s = gpu.getCurrentStream();
        gpu.selectThisGpu();
        gpu.notificationKernelIsLaunching("krNaturalCompressor");
        
        if (gpu.hasLittleEndianAdressing())
        {
            krNaturalCompressor<kBlockSize, kUnrollFactor, T, true> <<<grid, block, 0, s >>> (devOut, devIn, length);
        }
        else
        {
            krNaturalCompressor<kBlockSize, kUnrollFactor, T, false> <<<grid, block, 0, s >>> (devOut, devIn, length);
        }
        gpu.notificationKernelWasLaunched("krNaturalCompressor");
    }
    gpu.selectGpu(prevDev);
}

void dopt::applyKernelToMakeNaturalCompressor(double* devOut, double* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToMakeNaturalCompressorTemplate(devOut, devIn, length, gpu);
}
void dopt::applyKernelToMakeNaturalCompressor(float* devOut, float* devIn, size_t length, GpuManagement& gpu) {
    applyKernelToMakeNaturalCompressorTemplate(devOut, devIn, length, gpu);
}
void dopt::applyKernelToMakeNaturalCompressor(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpu) {
    assert(!"NOT IMPLEMENTED. NATURAL COMPRESSOR WITH INTEGER TYPE OF ELEMENTS MAKE NO SENSE");
}
void dopt::applyKernelToMakeNaturalCompressor(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpu) {
    assert(!"NOT IMPLEMENTED. NATURAL COMPRESSOR WITH INTEGER TYPE OF ELEMENTS MAKE NO SENSE");
}
//==============================================================================================================================//
