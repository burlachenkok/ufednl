#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>
#include <sstream>

// Prefix for shared memory. It is on the chip. Something similar to CPU L1 cache.
// It is captured when thread block is start to execute and it is released at the end of execution of ThreadBlock.
#define KR_SHARED_MEM_PREFIX __shared__

/** Thread block syncronization barrier.
* - Eeach thread in the same thread block must wait until all other threads in that thread block have reached this synchronization point.
* - All global and shared memory accesses made by all threads prior to this barrier will be visible to all other threads in the thread block after the barrier.
* - The function is used to coordinate communication between threads in the same block
* - It can negatively affect performance by forcing warps to become idle.
*/
#define KR_SYNC_THREAD_BLOCK() __syncthreads()

// Syncronize Warp.
// Assumptions that warp code is executed in lockstep or that reads/writes from separate threads are 
// visible across a warp without synchronization are invalid since CC [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x] 
#define KR_SYNC_WARP() __syncwarp()

// The warpSize built-in variable contains the number of threads per warp.
#define KR_WARP_SIZE (warpSize)

// For current thread it's local 1d coordinate in thread-block
#define KR_LOCAL_THREAD_ID_1D() (threadIdx.x)

// For current thread it's local 2d coordinate in thread-block
#define KR_LOCAL_THREAD_ID_2D() (threadIdx.y)

// For current thread it's thread block 1d coordinate in whole compute grid
#define KR_LOCAL_BLOCK_ID_1D() (blockIdx.x)

// For current thread it's thread block 2d coordinate in whole compute grid
#define KR_LOCAL_BLOCK_ID_2D() (blockIdx.y)

// For current thread it's thread block 1d coordinate in whole compute grid
#define KR_BLOCK_DIM_IN_THREADS_1D() (blockDim.x)

// For current thread it's thread block 2d coordinate in whole compute grid
#define KR_BLOCK_DIM_IN_THREADS_2D() (blockDim.y)

//==================================================================================================================
// Functions prefixies:

// Prefix for Kernel functions: Executed on the device & Callable from the host.
#define KR_KERNEL_ENTRY_FN __global__

// Prefix for Device functions: Executed on the device & Callable from the device(only).
#define KR_DEV_FN __device__

// Prefix for Device and Host functions: Executed on the device (or host) & Callable from the device (or host).
#define KR_DEV_AND_HOST_FN __device__ __host__

// Dump information about current thread in kernel
#define KR_DEBUG_THREAD_INFO() \
do {printf("debug: threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);} while(0)

// Prefix for Device functions: Executed on the host & Callable from the device(only).
#define HOST_FN __host__

// Execure CUDA Runtime call with sanity check
// Remark: The cudaGetErrorString function is analogous to the Standard C strerror function
#define CHECK_CUDA_CALL(call) do { const cudaError_t error = call; \
                                   if (error != cudaSuccess) \
                                   { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", int(error), cudaGetErrorString(error)); exit(-1); } \
                                 } while(0)

// Execure CUDA Runtime syncronization. 
// Remark: For DEBUG only. In final release can be removed.
#define CHECK_CUDA_DEBUG_SYNC(call) do { const cudaError_t error = call; \
                                      if (error != cudaSuccess) \
                                       { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", int(error), cudaGetErrorString(error)); exit(-1); } \
                                       } while(0)


namespace dopt
{
    template <class TStream>
    inline int getSPcores(TStream& out, const cudaDeviceProp& devProp)
    {
        int cores = 0;
        int mp = devProp.multiProcessorCount;

        switch (devProp.major)
        {
        case 2: // Fermi
            if (devProp.minor == 1)
            {
                cores = mp * 48;
            }
            else
            {
                cores = mp * 32;
            }
            break;

        case 3: // Kepler
            cores = mp * 192;
            break;

        case 5: // Maxwell
            cores = mp * 128; // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
            break;

        case 6: // Pascal
            if (devProp.minor >= 1)
                cores = mp * 128;
            else if (devProp.minor == 0)
                cores = mp * 64;
            break;
        case 7: // Volta (from this microarchitecture there are separate cores for FP32, FP64, INT operations)
            cores = mp * 64;
            break;
        case 8:
            if (devProp.minor >= 1)
                cores = mp * 128;  // Ada Lovelace
            else
                cores = mp * 64;   // Ampere (in this microarchitecture there are separate cores for FP32, FP64, INT operations)
            break;

        case 9: // Hooper
            cores = mp * 128;
            break;

        default:
            out << "Unknown device type for get Scalar Processor Count\n";
            break;
        }

        return cores;
    }

    inline int getWarpSchedulersPerStreamMultiprocessor(const cudaDeviceProp& devProp)
    {
        switch (devProp.major)
        {
            case 2: // Fermi
            {
                return 2;
            }

            case 3: // Kepler
            {
                return 4;
            }

            case 5: // Maxwell [16.4.1]
            {
                return 4;
            }

            case 6: // Pascal [16.5.1]
            {
                if (devProp.minor == 0)
                    return 2;
                else
                    return 4;
            }
            case 7: // Volta, Turing [16.6.1]
            {
                return 4;
            }
            case 8: // Ampere, ADA [16.7.1]
            {
                return 4;
            }
            case 9:
            {
                // Hooper [16.8.1]
                return 4;
            }
            default:
            {
                return 4;
            }
        }
    }


    inline std::string getDeviceString(const cudaDeviceProp& deviceProp, int deviceIndex)
    {
        std::stringstream out;

        const char* arch_names[] = { "" /*0*/,
                            "TESLA"  /*1*/,
                            "FERMI"  /*2*/,
                            "KEPLER" /*3*/,
                            ""       /*4*/,
                            "MAXWELL"/*5*/,
                            "PASCAL" /*6*/,
                            "VOLTA"  /*7*/,    // >= 7.5 -- TURING.
                            "AMPERE" /*8*/,    // >= 8.9 -- ADA_LOVELACE
                            "HOOPER" /*9*/ };

        const char* archName = "[UNKNOWN]";
        if (deviceProp.major >= sizeof(arch_names) / sizeof(arch_names[0]))
        {
            archName = "[UNKNOWN]";
        }
        else
        {
            archName = arch_names[deviceProp.major];
            if (deviceProp.major == 7 && deviceProp.minor >= 5)
                archName = "TURING";
            else if (deviceProp.major == 8 && deviceProp.minor >= 9)
                archName = "ADA-LOVELACE";
        }

        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level
        // TODO: Figure out how exactly compute peak FP32 -- does number of warp schedulers matters ?

        int kInstructionPerCycle = 0;

        if (deviceProp.major == 6)
            kInstructionPerCycle = 2;
        else
            kInstructionPerCycle = 1;

        int kWarpSchedulers = getWarpSchedulersPerStreamMultiprocessor(deviceProp);

        double peakPerformance_tflops = (double(deviceProp.clockRate /*in KHz*/) * getSPcores(out, deviceProp) * kInstructionPerCycle * kWarpSchedulers) / 1e+9;

        auto my_round = [](double value)->int {
            if (value - int(value) >= 0.5)
                return (int)value + 1;
            else
                return (int)value;
            };

        out << "[" << deviceIndex << "]"
            << " "
            << "" << deviceProp.name << ""
            << "/"
            << archName
            << "/CUDA Compute Capability: "
            << deviceProp.major << "." << deviceProp.minor
            << "/"
            << my_round((double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GBytes DRAM"
            << "/"
            << my_round(peakPerformance_tflops) << " TFLOPS@FP32 "
            << (deviceIndex == 0 ? "[DEFAULT]" : "") << "\n";

        return out.str();
    }
}
