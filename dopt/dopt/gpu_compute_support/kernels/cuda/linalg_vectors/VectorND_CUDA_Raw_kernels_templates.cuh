#pragma once

#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/LinalgComputePreprocessing.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/FloatUtilsCUDA.cuh"

#include <stdint.h>
#include <assert.h>

//========================================================================================//
template <class T>
KR_DEV_AND_HOST_FN T powerNaturalHelper(const T a, uint32_t naturalPower)
{
    if (naturalPower == 0) {
        return T(1);
    } else if (naturalPower % 2 == 0) {
        T tmp = powerNaturalHelper(a, naturalPower / 2);
        return tmp * tmp;
    } else {
        T tmp = powerNaturalHelper(a, (naturalPower - 1) / 2);
        return tmp * tmp * a;
    }
}
//========================================================================================//

template<dopt::BinaryReductionOperation reductionFunction,
    dopt::PreprocessForBinaryReductionOperation reductionPreprocessing,
    size_t BLOCKS_SIZE,
    size_t K_UNROLL_FACTOR,
    class T>

KR_KERNEL_ENTRY_FN void krReductionWith_1_VectorInput(T* g_odata, 
                                                      const T* g_idata, 
                                                      unsigned int n)
{
    static_assert(BLOCKS_SIZE == 1024 || BLOCKS_SIZE == 512 || BLOCKS_SIZE == 256 || 
                  BLOCKS_SIZE == 128 || BLOCKS_SIZE == 64, "BLOCKS_SIZE must be 1024, 512, 256, 128, 64");

    // Static shared memory
    KR_SHARED_MEM_PREFIX T smem[BLOCKS_SIZE];     

    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    // We enable each thread to handle four K data elements:
    // - The first step is recalculating the offset in the global input data based on the block and thread index of each thread
    // - Global index, K_UNROLL_FACTOR blocks of input data processed at a time
    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    // Pre-reduce across K_UNROLL_FACTOR elements from several blocks
    // unrolling all blocks
    T ai[K_UNROLL_FACTOR];

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        ai[i] = dopt::neutralForOperation<reductionFunction, T>();
    }

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < n)
        {
            ai[i] = dopt::preprocessItem<reductionPreprocessing>(g_idata[idx + i * BLOCKS_SIZE]);
        }
    }

    T tmpSum = ai[0];

    #pragma unroll
    for (size_t i = 1; i < K_UNROLL_FACTOR; ++i)
    {
        tmpSum = dopt::binaryOp<reductionFunction>(tmpSum, ai[i]);
    }

    // tmpSum is then used to initialize shared memory, rather than initializing it directly from global memory
    smem[tid] = tmpSum;

    // syncronize at this point view of shared memory
    KR_SYNC_THREAD_BLOCK();

    // in-place reduction in shared memory
    if (BLOCKS_SIZE >= 1024 && tid < 512) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 512]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 512 && tid < 256) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 256]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 256 && tid < 128) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 128]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 128 && tid < 64) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 64]);
    KR_SYNC_THREAD_BLOCK();

    // reduction with all strides > 32 should be completed up to this point
    volatile T* vsmem = smem;

    if (BLOCKS_SIZE >= 64 && tid < 32)
    {
        T v = vsmem[tid];                                          KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 32]); KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 16]); KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 8]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 4]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 2]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 1]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        if (tid == 0)
        {
            dopt::myAtomicApplyOperation<reductionFunction>(g_odata, v);
        }
    }
}

//==========================================================================================

template <dopt::UnaryOperation functionSelector, size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, 
          dopt::OperationFlags mode, class T>
KR_KERNEL_ENTRY_FN void krSimpleEwFunction_1_VectorInput(volatile T* g_odata, const T* g_idata, size_t length)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            if (mode == dopt::OperationFlags::eOperationNone)
            {
                switch (functionSelector)
                {
                    case dopt::UnaryOperation::eAbsEw:
                    {
                        T in = g_idata[idx + i * BLOCKS_SIZE];
                        T out = in > 0 ? in : -in;
                        g_odata[idx + i * BLOCKS_SIZE] = out;
                        break;
                    }
                    case dopt::UnaryOperation::eNegEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = -(g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eAppendEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] += (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eSubEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] -= (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eMulEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] *= (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eDivEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] /= (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }

                    case dopt::UnaryOperation::eExpEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = exp(double(g_idata[idx + i * BLOCKS_SIZE]));
                        break;
                    }
                    case dopt::UnaryOperation::eLogEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = log(double(g_idata[idx + i * BLOCKS_SIZE]));
                        break;
                    }
                    case dopt::UnaryOperation::eInvEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = T(1) / (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eSquareEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = (g_idata[idx + i * BLOCKS_SIZE]) * (g_idata[idx + i * BLOCKS_SIZE]);
                        break;
                    }
                    case dopt::UnaryOperation::eSqrtEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = sqrt(double(g_idata[idx + i * BLOCKS_SIZE]));
                        break;
                    }
                    case dopt::UnaryOperation::eInvSquareEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = T(1) / ((g_idata[idx + i * BLOCKS_SIZE]) * (g_idata[idx + i * BLOCKS_SIZE]));
                        break;
                    }
                    case dopt::UnaryOperation::eSigmoidEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = T(1) / (T(1) + exp(-double(g_idata[idx + i * BLOCKS_SIZE])));
                        break;
                    }
                    case dopt::UnaryOperation::eAssignEw:
                    {
                        g_odata[idx + i * BLOCKS_SIZE] = g_idata[idx + i * BLOCKS_SIZE];
                        break;
                    }
                    default:
                    {
                        assert(!"UNKNOWN FUNCTION TO APPLY");
                        break;
                    }
                }
            }
            else if (mode == dopt::OperationFlags::eMakeOperationAtomic)
            {
                T in = T();
                T out = T();
                T prev_out = T();

                do
                {
                    in = g_idata[idx + i * BLOCKS_SIZE];
                    prev_out = g_odata[idx + i * BLOCKS_SIZE];
                    
                    out = prev_out;

                    switch (functionSelector)
                    {
                        case dopt::UnaryOperation::eAbsEw:
                        {
                            out = in > 0 ? in : -in;
                            break;
                        }
                        
                        case dopt::UnaryOperation::eNegEw:
                        {
                            out = -in;
                            break;
                        }
                        case dopt::UnaryOperation::eAppendEw:
                        {
                            out += in;
                            break;
                        }
                        case dopt::UnaryOperation::eSubEw:
                        {
                            out -= in;
                            break;
                        }
                        case dopt::UnaryOperation::eMulEw:
                        {
                            out *= in;
                            break;
                        }
                        case dopt::UnaryOperation::eDivEw:
                        {
                            out /= in;
                            break;
                        }
                        case dopt::UnaryOperation::eExpEw:
                        {
                            out = exp(double(in));
                            break;
                        }                   
                        case dopt::UnaryOperation::eLogEw:
                        {
                            out = log(double(in));
                            break;
                        }
                        case dopt::UnaryOperation::eInvEw:
                        {
                            out = T(1) / in;
                            break;
                        }
                        case dopt::UnaryOperation::eSquareEw:
                        {
                            out = in * in;
                            break;
                        }                  
                        case dopt::UnaryOperation::eSqrtEw:
                        {
                            out = sqrt(double(in));
                            break;
                        }                 
                        case dopt::UnaryOperation::eInvSquareEw:
                        {
                            out = T(1) / (in * in);
                            break;
                        }                  
                        case dopt::UnaryOperation::eSigmoidEw:
                        {
                            out = T(1) / (T(1) + exp(-double(in)));
                            break;
                        }
                        case dopt::UnaryOperation::eAssignEw:
                        {
                            out = in;
                            break;
                        }
                        default:
                        {
                            assert(!"UNKNOWN FUNCTION TO APPLY");
                            break;
                        }
                    }
                } while (! ::dopt::myCAS(&(g_odata[idx + i * BLOCKS_SIZE]), prev_out, out) );
            }
        }
    }
}

//==========================================================================================
template <size_t BLOCKS_SIZE, 
          size_t K_UNROLL_FACTOR, 
          class T>

KR_KERNEL_ENTRY_FN void krSimpleEwFunction_1_VectorInput_and_2_scalars(T* g_odata, const T* g_idata, size_t length, const T posSignValue, const T negSignValue)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            T in = g_idata[idx + i * BLOCKS_SIZE];
            T out = in > 0 ? posSignValue : negSignValue;
            g_odata[idx + i * BLOCKS_SIZE] = out;
        }
    }
}
//==========================================================================================
template <dopt::SingleArgUnaryOperation functionSelector, size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, 
          dopt::OperationFlags mode, class T>
KR_KERNEL_ENTRY_FN void krSimpleEwFunction_1_VectorInput_and_1_scalar(volatile T* g_io_data, T arg, size_t length)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            if (mode == dopt::OperationFlags::eOperationNone)
            {
                switch (functionSelector)
                {
                    case dopt::SingleArgUnaryOperation::eMultByValue:
                    {
                        T in = g_io_data[idx + i * BLOCKS_SIZE];
                        T out = in * arg;
                        g_io_data[idx + i * BLOCKS_SIZE] = out;
                        break;
                    }
                    case dopt::SingleArgUnaryOperation::eDivByValue:
                    {
                        T in = g_io_data[idx + i * BLOCKS_SIZE];
                        T out = in / arg;
                        g_io_data[idx + i * BLOCKS_SIZE] = out;
                        break;
                    }
                    case dopt::SingleArgUnaryOperation::eSetToValue:
                    {
                        g_io_data[idx + i * BLOCKS_SIZE] = arg;
                        break;
                    }
                    case dopt::SingleArgUnaryOperation::eZeroOutItems:
                    {
                        if (g_io_data[idx + i * BLOCKS_SIZE] >= -arg && g_io_data[idx + i * BLOCKS_SIZE] <= arg)
                        {
                            g_io_data[idx + i * BLOCKS_SIZE] = T();
                        }
                        break;
                    }
                    case dopt::SingleArgUnaryOperation::eAddValue:
                    {
                        T in = g_io_data[idx + i * BLOCKS_SIZE];
                        T out = in + arg;
                        g_io_data[idx + i * BLOCKS_SIZE] = out;
                        break;
                    }
                    default:
                    {
                        assert(!"UNKNOWN FUNCTION TO APPLY");
                        break;
                    }
                }
            }
            else if (mode == dopt::OperationFlags::eMakeOperationAtomic) 
            {
                T new_in_out = T();
                T prev_in_out = T();
                // g_io_data[idx + i * BLOCKS_SIZE] <---- write place
                do
                {
                    prev_in_out = g_io_data[idx + i * BLOCKS_SIZE];

                    switch (functionSelector)
                    {
                        case dopt::SingleArgUnaryOperation::eMultByValue:
                        {
                            new_in_out = prev_in_out * arg;
                            break;
                        }
                        case dopt::SingleArgUnaryOperation::eDivByValue:
                        {
                            new_in_out = prev_in_out / arg;
                            break;
                        }
                        case dopt::SingleArgUnaryOperation::eSetToValue:
                        {
                            new_in_out = arg;
                            break;
                        }
                        case dopt::SingleArgUnaryOperation::eZeroOutItems:
                        {
                            if (prev_in_out >= -arg && prev_in_out <= arg)
                            {
                                new_in_out = T();
                            }
                            break;
                        }
                        case dopt::SingleArgUnaryOperation::eAddValue:
                        {
                            new_in_out = prev_in_out + arg;
                            break;
                        }
                        default:
                        {
                            assert(!"UNKNOWN FUNCTION TO APPLY");
                            break;
                        }
                    }
                } while (! ::dopt::myCAS(&(g_io_data[idx + i * BLOCKS_SIZE]), prev_in_out, new_in_out));
            }
        }
    }
}
//==========================================================================================

template <dopt::BinaryReductionOperation reductionFunction,
    size_t BLOCKS_SIZE,
    size_t K_UNROLL_FACTOR,
    class T>
KR_KERNEL_ENTRY_FN void krReductionWith_2_VectorInput(T* g_odata,
                                                         const T* g_idata_arg_1,
                                                         const T* g_idata_arg_2,
                                                         unsigned int n)
{
    static_assert(BLOCKS_SIZE == 1024 || BLOCKS_SIZE == 512 || BLOCKS_SIZE == 256 || 
                  BLOCKS_SIZE == 128 || BLOCKS_SIZE == 64, "BLOCKS_SIZE must be 1024, 512, 256, 128, 64");

    KR_SHARED_MEM_PREFIX T smem[BLOCKS_SIZE];     // static shared memory

    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    // We enable each thread to handle four K data elements:
    // - The first step is recalculating the offset in the global input data based on the block and thread index of each thread
    // - Global index, K_UNROLL_FACTOR blocks of input data processed at a time
    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    // Pre-reduce across K_UNROLL_FACTOR elements from several blocks
    // unrolling all blocks
    T ai[K_UNROLL_FACTOR];

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        ai[i] = dopt::neutralForOperation<reductionFunction, T>();
    }

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < n)
        {
            ai[i] = g_idata_arg_1[idx + i * BLOCKS_SIZE] * g_idata_arg_2[idx + i * BLOCKS_SIZE];
        }
    }

    T tmpSum = ai[0];

    #pragma unroll
    for (size_t i = 1; i < K_UNROLL_FACTOR; ++i)
    {
        tmpSum = dopt::binaryOp<reductionFunction>(tmpSum, ai[i]);
    }

    // tmpSum is then used to initialize shared memory, rather than initializing it directly from global memory
    smem[tid] = tmpSum;

    // syncronize at this point view of shared memory
    KR_SYNC_THREAD_BLOCK();

    // in-place reduction in shared memory
    if (BLOCKS_SIZE >= 1024 && tid < 512) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 512]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 512 && tid < 256) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 256]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 256 && tid < 128) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 128]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 128 && tid < 64) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 64]);
    KR_SYNC_THREAD_BLOCK();

    // reduction with all strides > 32 should be completed up to this point
    volatile T* vsmem = smem;

    if (BLOCKS_SIZE >= 64 && tid < 32)
    {
        T v = vsmem[tid];                                          KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 32]); KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 16]); KR_SYNC_WARP();
        vsmem[tid] = v;                                             KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 8]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 4]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 2]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 1]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        if (tid == 0)
        {
            dopt::myAtomicApplyOperation<reductionFunction>(g_odata, v);
        }
    }
}
//========================================================================================
template<dopt::BinaryReductionOperation reductionFunction, 
         size_t BLOCKS_SIZE, 
         size_t K_UNROLL_FACTOR, 
         class T>
KR_KERNEL_ENTRY_FN void krApplyKernelToEvaluateLpNormHelperTemplate(T* g_odata,
                                                                    const T* g_idata_arg,
                                                                    unsigned int n,
                                                                    uint32_t p)
{
    static_assert(BLOCKS_SIZE == 1024 || BLOCKS_SIZE == 512 || BLOCKS_SIZE == 256 || 
                  BLOCKS_SIZE == 128 || BLOCKS_SIZE == 64, "BLOCKS_SIZE must be 1024, 512, 256, 128, 64");

    KR_SHARED_MEM_PREFIX T smem[BLOCKS_SIZE];     // static shared memory

    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    // We enable each thread to handle four K data elements:
    // - The first step is recalculating the offset in the global input data based on the block and thread index of each thread
    // - Global index, K_UNROLL_FACTOR blocks of input data processed at a time
    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    // Pre-reduce across K_UNROLL_FACTOR elements from several blocks
    // unrolling all blocks
    T ai[K_UNROLL_FACTOR];

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        ai[i] = dopt::neutralForOperation<reductionFunction, T>();
    }

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < n)
        {
            T arg = g_idata_arg[idx + i * BLOCKS_SIZE];
            if (arg < 0)
                arg = -arg;
            ai[i] = powerNaturalHelper(arg, p);
        }
    }

    T tmpSum = ai[0];

    #pragma unroll
    for (size_t i = 1; i < K_UNROLL_FACTOR; ++i)
    {
        tmpSum = dopt::binaryOp<reductionFunction>(tmpSum, ai[i]);
    }

    // tmpSum is then used to initialize shared memory, rather than initializing it directly from global memory
    smem[tid] = tmpSum;

    // syncronize at this point view of shared memory
    KR_SYNC_THREAD_BLOCK();

    // in-place reduction in shared memory
    if (BLOCKS_SIZE >= 1024 && tid < 512) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 512]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 512 && tid < 256) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 256]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 256 && tid < 128) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 128]);
    KR_SYNC_THREAD_BLOCK();

    if (BLOCKS_SIZE >= 128 && tid < 64) smem[tid] = dopt::binaryOp<reductionFunction>(smem[tid], smem[tid + 64]);
    KR_SYNC_THREAD_BLOCK();

    // reduction with all strides > 32 should be completed up to this point
    volatile T* vsmem = smem;

    if (BLOCKS_SIZE >= 64 && tid < 32)
    {
        T v = vsmem[tid];                                          KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 32]); KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 16]); KR_SYNC_WARP();
        vsmem[tid] = v;                                             KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 8]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 4]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 2]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        v = dopt::binaryOp<reductionFunction>(v, vsmem[tid + 1]);  KR_SYNC_WARP();
        vsmem[tid] = v;                                            KR_SYNC_WARP();

        if (tid == 0)
        {
            dopt::myAtomicApplyOperation<reductionFunction>(g_odata, v);
        }
    }
}
//========================================================================================
template <dopt::TwoArgUnaryOperation functionSelector, size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, class T>
KR_KERNEL_ENTRY_FN void krApplyKernelToTwoArgFunctionVectorizedTemplate(T* g_io_data, T* g_arg1, T* g_arg2, size_t length)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            switch (functionSelector)
                {
                case dopt::TwoArgUnaryOperation::eClamp:
                {
                    T tmp = g_io_data[idx + i * BLOCKS_SIZE];
                    T arg1 = g_arg1[idx + i * BLOCKS_SIZE];
                    T arg2 = g_arg2[idx + i * BLOCKS_SIZE];

                    if (tmp <= arg1)
                        tmp = arg1;
                    else if (tmp >= arg2)
                        tmp = arg2;

                    g_io_data[idx + i * BLOCKS_SIZE] = tmp;

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
//========================================================================================

template <dopt::UnaryOperation functionSelector, size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, class T>
KR_KERNEL_ENTRY_FN void krSimpleEwFunctionWithMultiplier_1_VectorInput_1_scalar(T* g_odata, const T* g_idata, size_t length, T extraMultiplier)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            switch (functionSelector)
            {
                case dopt::UnaryOperation::eAppendEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] += extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }
                case dopt::UnaryOperation::eSubEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] -= extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }
                case dopt::UnaryOperation::eMulEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] *= extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }
                case dopt::UnaryOperation::eDivEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] /= extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }

                case dopt::UnaryOperation::eExpEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * exp(double(g_idata[idx + i * BLOCKS_SIZE]));
                    break;
                }
                case dopt::UnaryOperation::eLogEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * log(double(g_idata[idx + i * BLOCKS_SIZE]));
                    break;
                }
                case dopt::UnaryOperation::eInvEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * T(1) / (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }
                case dopt::UnaryOperation::eSquareEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]) * (g_idata[idx + i * BLOCKS_SIZE]);
                    break;
                }
                case dopt::UnaryOperation::eSqrtEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * sqrt(double(g_idata[idx + i * BLOCKS_SIZE]));
                    break;
                }
                case dopt::UnaryOperation::eInvSquareEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * T(1) / ((g_idata[idx + i * BLOCKS_SIZE]) * (g_idata[idx + i * BLOCKS_SIZE]));
                    break;
                }
                case dopt::UnaryOperation::eSigmoidEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * T(1) / (T(1) + exp(-double(g_idata[idx + i * BLOCKS_SIZE])));
                    break;
                }
                case dopt::UnaryOperation::eAssignEw:
                {
                    g_odata[idx + i * BLOCKS_SIZE] = extraMultiplier * (g_idata[idx + i * BLOCKS_SIZE]);
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

//========================================================================================

template <dopt::TwoArgUnaryOperation functionSelector, size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, class T>
KR_KERNEL_ENTRY_FN void krApplyKernelToTwoArgFunctionTemplate(T* g_io_data, T arg1, T arg2, size_t length)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            switch (functionSelector)
            {
                case dopt::TwoArgUnaryOperation::eClamp:
                {
                    T tmp = g_io_data[idx + i * BLOCKS_SIZE];

                    if (tmp <= arg1)
                        tmp = arg1;
                    else if (tmp >= arg2)
                        tmp = arg2;;

                    g_io_data[idx + i * BLOCKS_SIZE] = tmp;

                    break;
                }
                case dopt::TwoArgUnaryOperation::eScaledDiff:
                {
                    T tmp = g_io_data[idx + i * BLOCKS_SIZE];
                    T out = (arg1 - tmp) * arg2;
                    g_io_data[idx + i * BLOCKS_SIZE] = out;
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
//========================================================================================

template <size_t BLOCKS_SIZE, size_t K_UNROLL_FACTOR, class T, bool DOPT_GPU_ARCH_LITTLE_ENDIAN = true>
KR_KERNEL_ENTRY_FN void krNaturalCompressor(T* g_odata, const T* g_idata, size_t length)
{
    int tid = KR_LOCAL_THREAD_ID_1D();
    int bid = KR_LOCAL_BLOCK_ID_1D();

    int idx = bid * (BLOCKS_SIZE * K_UNROLL_FACTOR) + tid;

    #pragma unroll
    for (size_t i = 0; i < K_UNROLL_FACTOR; ++i)
    {
        if (idx + i * BLOCKS_SIZE < length)
        {
            auto pack = ::cuda4dopt::getFloatPointPack<DOPT_GPU_ARCH_LITTLE_ENDIAN>(g_idata[idx + i * BLOCKS_SIZE]);
            pack.components.mantissa = 0;
            g_odata[idx + i * BLOCKS_SIZE] = pack.real_value_repr;
        }
    }
}
