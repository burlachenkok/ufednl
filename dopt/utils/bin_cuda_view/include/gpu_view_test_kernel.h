#pragma once

#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

namespace benchmark
{
    /** Kernel to test sum of long vectors d_x and and d_y and put results into d_z.
    * @param d_x pointer in devices memory to dense vector X
    * @param d_y pointer in devices memory to dense vector Y
    * @param d_z pointer in devices memory to dense vector Z
    * @param iterations number of times for performing ADD and WRITE for benchmarking
    * @param dim input dimension in items
    * @param blockSize block size for launching grid in items
    */
    void testSumKernel(float* d_x, float* d_y, float* d_z, int iterations, size_t dim, size_t blockSize);

    /** Kernel to test copy of long vectors src to dst
    * @param dst pointer in devices memory to dense vector destination
    * @param src pointer in devices memory to dense vector source
    * @param iterations number of times for performing copy
    * @param dim input dimension in items
    * @param blockSize block size for launching grid in items
    */
    void testCopyKernel(float* dst, float* src, int iterations, size_t dim, size_t blockSize);

    /** Debug information about current device
    */
    void printCudaDebugInformation(int devNumber, const cudaDeviceProp& deviceProp);
}
