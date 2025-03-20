#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/system/include/CompilerInfo.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

#include <assert.h>
#include <iostream>

KR_KERNEL_ENTRY_FN void testSumKernelGPU(float* x, float* y, float* z, int iterations)
{
    // checkIndex();
    int i = KR_LOCAL_BLOCK_ID_1D() * KR_BLOCK_DIM_IN_THREADS_1D() + KR_LOCAL_THREAD_ID_1D();
    volatile float xi = x[i];
    volatile float yi = y[i];
    volatile float zi = 0;

    for (int j = 0; j < iterations; ++j)
    {
        zi = xi + yi;
    }
    
    z[i] = zi;
}

KR_KERNEL_ENTRY_FN void testCopyKernelGPU(float* dst, float* src)
{
    int i = KR_LOCAL_BLOCK_ID_1D() * KR_BLOCK_DIM_IN_THREADS_1D() + KR_LOCAL_THREAD_ID_1D();
    dst[i] = src[i];
}

namespace benchmark
{
    void testSumKernel(float* d_x, float* d_y, float* d_z, int iterations, size_t dim, size_t blockSize)
    {
        assert(dim % blockSize == 0);
        testSumKernelGPU<<< dim / blockSize, blockSize >>>(d_x, d_y, d_z, iterations);
        cudaError_t status = cudaDeviceSynchronize();
        assert(status == cudaSuccess);

        if (status != cudaSuccess)
        {
            std::cout << "testSumKernelGPU failed: " << cudaGetErrorString(status) << " [FAILED]\n";
        }
    }

    void testCopyKernel(float* dst, float* src, int iterations, size_t dim, size_t blockSize)
    {
        assert(dim % blockSize == 0);

        for (size_t i = 0; i < iterations; ++i)
        {
            testCopyKernelGPU<<< dim / blockSize, blockSize >>>(dst, src);
            cudaError_t status = cudaDeviceSynchronize();
            assert(status == cudaSuccess);
            if (status != cudaSuccess)
            {
                std::cout << "testCopyKernelGPU failed: " << cudaGetErrorString(status) << " [FAILED]\n";
            }
        }
    }

    union test_endian
    {
        long testWord;
        char testWordInBytes[sizeof(long)];
    } u = {};
    
    KR_DEV_FN bool devIsLittleEndian()
    {
        test_endian u;
        u.testWord = 1;
        bool rightToLeft = (u.testWordInBytes[0] == 1);
        return rightToLeft;
    }

    KR_KERNEL_ENTRY_FN void printSizeOfMainCppTypesInDevice(int devNumber)
    {
        constexpr int a = 1;      
        printf("CUDA Device Execution [device|gpu]. Device [%i]\n\n", int(devNumber));
        KR_DEBUG_THREAD_INFO();

        printf("   sizeof(bool): %i\n", int(sizeof(bool)));
        printf("   sizeof(unsigned short): %i\n", int(sizeof(unsigned short)));
        printf("   sizeof(unsigned int): %i\n", int(sizeof(unsigned int)));
        printf("   sizeof(unsigned long): %i\n", int(sizeof(unsigned long)));
        printf("   sizeof(unsigned long long): %i\n", int(sizeof(unsigned long long)));
        printf("   sizeof(float): %i\n", int(sizeof(float)));
        printf("   sizeof(double): %i\n", int(sizeof(double)));
        printf("   sizeof(size_t): %i\n", int(sizeof(size_t)));
        printf("   sizeof(void*): %i\n", int(sizeof(void*)));
        printf("   warp size: %i\n", int(KR_WARP_SIZE));
        printf("   __cplusplus version: %i\n", int(__cplusplus));
       
        printf("   endian [device]: %s\n", devIsLittleEndian() ? "little-endian [left <= right]" : "big-endian [left => right]\n");
        printf("\n");
        
        #if defined(__CUDACC_DEBUG__)
            #define CUDACC_DEBUG_IS_ON true
        #else
            #define CUDACC_DEBUG_IS_ON false
        #endif  

        printf("   compiling CUDA source files in the device - debug mode (__CUDACC_DEBUG__): %s\n", CUDACC_DEBUG_IS_ON ? "[YES]" : "[NO]");
        printf("   compiling CUDA source files with: NVCC %i.%i.%i\n", int(__CUDACC_VER_MAJOR__), int(__CUDACC_VER_MINOR__), int(__CUDACC_VER_BUILD__));
    }

    // Small fast tests for C++11,14,17,20 CUDA compilation [To eliminate problems with C++ language selection]
    
    constexpr int compile_test_cpp_11(int a, int b) {
        return a + b;
    }
    consteval unsigned char compile_test_cpp_14() {
        unsigned char a = 0b00110011;
        return a;
    }
    consteval void compile_test_cpp_17() {
        if (int a = 1; a == 1) {
            return;
        }
    }
    
    void printCudaDebugInformation(int devNumber, const cudaDeviceProp& deviceProp)
    {
        constexpr int a = 1;

        printf("CUDA Runtime Execution [host|cpu]. Device [%i]\n", int(devNumber));
        printf("   sizeof(bool): %i\n", int(sizeof(bool)));
        printf("   sizeof(unsigned short): %i\n", int(sizeof(unsigned short)));
        printf("   sizeof(unsigned int): %i\n", int(sizeof(unsigned int)));
        printf("   sizeof(unsigned long): %i\n", int(sizeof(unsigned long)));
        printf("   sizeof(unsigned long long): %i\n", int(sizeof(unsigned long long)));
        printf("   sizeof(float): %i\n", int(sizeof(float)));
        printf("   sizeof(double): %i\n", int(sizeof(double)));
        printf("   sizeof(size_t): %i\n", int(sizeof(size_t)));
        printf("   sizeof(void*): %i\n", int(sizeof(void*)));
        printf("   warp size: %i\n", int(deviceProp.warpSize));
        printf("   __cplusplus version: %i\n", int(__cplusplus));
        test_endian u;
        u.testWord = 1;
        bool rightToLeft = (u.testWordInBytes[0] == 1);
        printf("   endian [host]: %s\n", rightToLeft ? "little-endian [left <= right]" : "big-endian [left => right]\n");
        printf("\n");
        printf("   compiling HOST source files with: %s\n", dopt::compilerCppVersion().c_str());
        printf("\n");
        
        printSizeOfMainCppTypesInDevice <<<1, 1 >>> (devNumber);
    }
}
