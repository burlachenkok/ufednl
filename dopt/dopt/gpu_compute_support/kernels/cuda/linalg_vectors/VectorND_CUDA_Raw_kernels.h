#pragma once

#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#include <stdint.h>

namespace dopt
{
    enum OperationFlags
    {
        eOperationNone = 0x0,
        eMakeOperationAtomic = 0x1,
    };
    
    void applyKernelToEvaluateMax(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMax(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMax(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMax(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);

    void applyKernelToEvaluateMin(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMin(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMin(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateMin(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);

    void applyKernelToEvaluateSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);

    void applyKernelToEvaluateNnz(double* devResultAccum, double* devPtr, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNnz(float* devResultAccum, float* devPtr, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNnz(uint32_t* devResultAccum, uint32_t* devPtr, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNnz(int32_t* devResultAccum, int32_t* devPtr, size_t length, GpuManagement& gpu);

    void applyKernelToEvaluateLogisticLossFromMarginSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);

    void applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateLogisticLossFromMarginSigmoidSum(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToEvaluateAbsItems(double* devOut, const double* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateAbsItems(float* devOut, const float* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateAbsItems(uint32_t* devOut, const uint32_t* devIn, size_t items, GpuManagement& gpu);
    void applyKernelToEvaluateAbsItems(int32_t* devOut, const int32_t* devIn, size_t items, GpuManagement& gpu);

    void applyKernelToEvaluateExpItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateExpItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateExpItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateExpItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);


    void applyKernelToEvaluateLogItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLogItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLogItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLogItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);


    void applyKernelToEvaluateInvItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);

    void applyKernelToEvaluateSquareItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSquareItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSquareItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSquareItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);

    void applyKernelToEvaluateSqrtItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSqrtItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSqrtItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSqrtItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);

    void applyKernelToEvaluateInvSquareItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvSquareItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvSquareItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateInvSquareItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);

    void applyKernelToEvaluateSigmoidFn(double* devOut, const double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSigmoidFn(float* devOut, const float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSigmoidFn(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateSigmoidFn(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpuManagement);

    void applyKernelToEvaluateNegItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNegItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNegItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateNegItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu);

    void applyKernelToEvaluateAppendItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu, OperationFlags flags);
    void applyKernelToEvaluateAppendItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu, OperationFlags flags);
    void applyKernelToEvaluateAppendItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu, OperationFlags flags);
    void applyKernelToEvaluateAppendItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu, OperationFlags flags);

    void applyKernelToEvaluateSubItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateSubItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateSubItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateSubItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu);

    void applyKernelToEvaluateMultiplyItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateMultiplyItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateMultiplyItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateMultiplyItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu);
    
    void applyKernelToEvaluateDivItems(double* devOut, const double* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateDivItems(float* devOut, const float* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateDivItems(uint32_t* devOut, const uint32_t* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToEvaluateDivItems(int32_t* devOut, const int32_t* devIn, size_t length, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToEvaluateSignOfItems(double* devOut, const double* devIn, size_t length, const double posSignValue, const double negSignValue, GpuManagement& gpu);
    void applyKernelToEvaluateSignOfItems(float* devOut, const float* devIn, size_t length, const float posSignValue, const float negSignValue, GpuManagement& gpu);
    void applyKernelToEvaluateSignOfItems(uint32_t* devOut, const uint32_t* devIn, size_t length, const uint32_t posSignValue, const uint32_t negSignValue, GpuManagement& gpu);
    void applyKernelToEvaluateSignOfItems(int32_t* devOut, const int32_t* devIn, size_t length, const int32_t posSignValue, const int32_t negSignValue, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToSetAllItemsToValue(double* devOut, size_t length, double value, GpuManagement& gpu);
    void applyKernelToSetAllItemsToValue(float* devOut, size_t length, float value, GpuManagement& gpu);
    void applyKernelToSetAllItemsToValue(uint32_t* devOut, size_t length, uint32_t value, GpuManagement& gpu);
    void applyKernelToSetAllItemsToValue(int32_t* devOut, size_t length, int32_t value, GpuManagement& gpu);

    void applyKernelToEvaluateMutiplyItemsByFactor(double* devInOut, size_t length, double factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateMutiplyItemsByFactor(float* devInOut, size_t length, float factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateMutiplyItemsByFactor(uint32_t* devInOut, size_t length, uint32_t factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateMutiplyItemsByFactor(int32_t* devInOut, size_t length, int32_t factor, GpuManagement& gpu, OperationFlags mode);

    void applyKernelToEvaluateDivItemsByFactor(double* devInOut, size_t length, double factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateDivItemsByFactor(float* devInOut, size_t length, float factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateDivItemsByFactor(uint32_t* devInOut, size_t length, uint32_t factor, GpuManagement& gpu, OperationFlags mode);
    void applyKernelToEvaluateDivItemsByFactor(int32_t* devInOut, size_t length, int32_t factor, GpuManagement& gpu, OperationFlags mode);

    void applyKernelToZeroOutItems(double* devInOut, size_t length, double eps, GpuManagement& gpuManagement);
    void applyKernelToZeroOutItems(float* devInOut, size_t length, float eps, GpuManagement& gpuManagement);
    void applyKernelToZeroOutItems(uint32_t* devInOut, size_t length, uint32_t eps, GpuManagement& gpuManagement);
    void applyKernelToZeroOutItems(int32_t* devInOut, size_t length, int32_t eps, GpuManagement& gpuManagement);
}

namespace dopt
{
    void applyKernelToReducedDotProduct(double* devOut, double* devIn1, double* devIn2, size_t length, GpuManagement& gpu);
    void applyKernelToReducedDotProduct(float* devOut, float* devIn1, float* devIn2, size_t length, GpuManagement& gpu);
    void applyKernelToReducedDotProduct(uint32_t* devOut, uint32_t* devIn1, uint32_t* devIn2, size_t length, GpuManagement& gpu);
    void applyKernelToReducedDotProduct(int32_t* devOut, int32_t* devIn1, int32_t* devIn2, size_t length, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToEvaluateLpNormToPowerP(double* devOut, double* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm);
    void applyKernelToEvaluateLpNormToPowerP(float* devOut, float* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm);
    void applyKernelToEvaluateLpNormToPowerP(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm);
    void applyKernelToEvaluateLpNormToPowerP(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpuManagement, uint32_t pnorm);

    void applyKernelToEvaluateLInfNorm(double* devOut, double* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLInfNorm(float* devOut, float* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLInfNorm(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpuManagement);
    void applyKernelToEvaluateLInfNorm(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpuManagement);
}

namespace dopt
{
    void applyKernelToClampItems(double* devInOut, size_t length, double lower, double upper, GpuManagement& gpuManagement);
    void applyKernelToClampItems(float* devInOut, size_t length, float lower, float upper, GpuManagement& gpuManagement);
    void applyKernelToClampItems(uint32_t* devInOut, size_t length, uint32_t lower, uint32_t upper, GpuManagement& gpuManagement);
    void applyKernelToClampItems(int32_t* devInOut, size_t length, int32_t lower, int32_t upper, GpuManagement& gpuManagement);

    void applyKernelToScaledDifferenceWithEye(double* devInOut, size_t length, double a, double multiple, GpuManagement& gpu);
    void applyKernelToScaledDifferenceWithEye(float* devInOut, size_t length, float a, float multiple, GpuManagement& gpu);
    void applyKernelToScaledDifferenceWithEye(uint32_t* devInOut, size_t length, uint32_t a, uint32_t multiple, GpuManagement& gpu);
    void applyKernelToScaledDifferenceWithEye(int32_t* devInOut, size_t length, int32_t a, int32_t multiple, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToClampItemsVectorized(double* devInOut, size_t length, double* devLower, double* devUpper, GpuManagement& gpuManagement);
    void applyKernelToClampItemsVectorized(float* devInOut, size_t length, float* devLower, float* devUpper, GpuManagement& gpuManagement);
    void applyKernelToClampItemsVectorized(uint32_t* devInOut, size_t length, uint32_t* devLower, uint32_t* devUpper, GpuManagement& gpuManagement);
    void applyKernelToClampItemsVectorized(int32_t* devInOut, size_t length, int32_t* devLower, int32_t* devUpper, GpuManagement& gpuManagement);
}

namespace dopt
{
    void applyKernelToEvaluateAppendItemsWithMultiplier(double* devOut, double* devIn, size_t length, double multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAppendItemsWithMultiplier(float* devOut, float* devIn, size_t length, float multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAppendItemsWithMultiplier(uint32_t* devOut, uint32_t* devIn, size_t length, uint32_t multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAppendItemsWithMultiplier(int32_t* devOut, int32_t* devIn, size_t length, int32_t multiple, GpuManagement& gpu);

    void applyKernelToEvaluateAssignItemsWithMultiplier(double* devOut, double* devIn, size_t length, double multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAssignItemsWithMultiplier(float* devOut, float* devIn, size_t length, float multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAssignItemsWithMultiplier(uint32_t* devOut, uint32_t* devIn, size_t length, uint32_t multiple, GpuManagement& gpu);
    void applyKernelToEvaluateAssignItemsWithMultiplier(int32_t* devOut, int32_t* devIn, size_t length, int32_t multiple, GpuManagement& gpu);
}

namespace dopt
{
    void applyKernelToMakeNaturalCompressor(double* devOut, double* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToMakeNaturalCompressor(float* devOut, float* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToMakeNaturalCompressor(uint32_t* devOut, uint32_t* devIn, size_t length, GpuManagement& gpu);
    void applyKernelToMakeNaturalCompressor(int32_t* devOut, int32_t* devIn, size_t length, GpuManagement& gpu);
}
