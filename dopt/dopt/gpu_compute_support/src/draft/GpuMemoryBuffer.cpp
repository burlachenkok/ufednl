#if 0
#include "dopt/gpu_compute_support/include/GpuMemoryBuffer.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

namespace dopt
{
    
    GPUPointer GPUMemoryBuffer::rawAllocate(size_t szInBytes)
    {
        void* device_pointer = 0;
        CHECK_CUDA_CALL(cudaMalloc((void**)&device_pointer, szInBytes));
        return device_pointer;
    }

    void GPUMemoryBuffer::rawDeallocate(GPUPointer ptr)
    {
        CHECK_CUDA_CALL(cudaFree(ptr));
    }

    void GPUMemoryBuffer::rawCopyDevice2Device(GPUPointer dst, GPUPointer src, size_t szInBytes)
    {
        CHECK_CUDA_CALL(cudaMemcpy(dst, src, szInBytes, cudaMemcpyDeviceToDevice));
    }
}
#endif
