cmake_minimum_required(VERSION 3.12)

include(DoptVersionGit)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})

if (DOPT_CUDA_SUPPORT)
    project(${ProjectId} LANGUAGES CXX C CUDA)
    file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "src/*.cu" "kernels/*.cu" "kernels/*.cpp" "kernels/*.cxx" "kernels/*.c" "kernels/*.cuh")
    file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp" "kernels/*.h" "kernels/*.hpp")
endif()

if (DOPT_OPENCL_SUPPORT)
    project(${ProjectId} LANGUAGES CXX C)
    file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "kernels/*.cpp" "kernels/*.cxx" "kernels/*.c")
    file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp" "kernels/*.h" "kernels/*.hpp")
endif()


if(original_src)
    createSourceGrouping(${original_src})
endif()

if (original_headers)
    createHeadersGrouping(${original_headers})
endif()

#============= BUILD TARGETS =================================================================
if (DOPT_CUDA_SUPPORT)
  cuda_add_library(${PROJECT_NAME} STATIC ${original_src} ${original_headers})
  target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
else()
  add_library(${PROJECT_NAME} STATIC ${original_src} ${original_headers})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#============= BUILD TARGETS =================================================================

target_link_libraries(${PROJECT_NAME} linalg_vectors)

# Configure CPU/Host compiler flags for project
configureCompileFlags()

#if (DOPT_CUDA_SUPPORT)
  # Passing Host compiler flags via NVCC
  # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768/2
#  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${CMAKE_CXX_FLAGS}>)
#endif()

#get_target_property(PROJ_CFLAGS ${PROJECT_NAME} COMPILE_DEFINITIONS)
#message(" ++Target compiler flags for ${PROJECT_NAME}: ${PROJ_CFLAGS}")

# NVIDIA GPU/CUDA relative
#=============================================================================================
# CUDA compilation relative flags:
# List of compiler flags: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
#
# -arch sm_20 --- The switch causes the compiler to generate device code for the specific architecture.
# -Xcompiler  --- Specify options directly to the compiler/preprocessor.
# -G          --- Generate debug information for device code.
# -g          --- Generate debug information for host code.
# --dopt=on   --- Enable device code optimization [by default is on]
# -O3         --- Optimization level of host code
# --ptxas-options=-v --- Registers per thread and shared memory per block resource usage can be obtained from nvcc
# �maxrregcount=NUM  --- The option tells the compiler to not use more than NUM registers per thread. 
# -rdc               --- Enable or disable the generation of relocatable device code.
# -Xptxas -v,-abi=no --- Print the number of registers, bytes of shared memory, and bytes of constant memory used by each thread.
# -Xptxas -dlcm=cg   --- Flags inform the compiler to disable the L1 cache for GPU DRAM access. Memory request is serviced by a 32-byte  memory transaction.
# -Xptxas -dlcm=ca   --- Flags inform the compiler to enable the L1 cache for GPU DRAM access. Memory request is serviced by a 128-byte  memory transaction.

# CUDA relative info:
# 1. In Linux NVIDIA devices are available in /dev/. To check number of NVIDIA GPUS one can execute `ls -l /dev/nvidia*`
# 2. printf function is only supported on architectures starting with Fermi GPUs, thefore -arch=sm_20 at least maybe necessary
# 3. List NVIDIA GPU devices: `nvidia-smi -L`
# 4. Details about GPU #0: `nvidia-smi -q -i 0` 
# 5. Detailed utilization info: `nvidia-smi -q -i 0 -d UTILIZATION`
# 6. Number of effective warp schedulers defines how much warps can be execute at the clock.
# 7. DRAM GPU Memory is conneced to all SMs.
# 8. The only safe way to synchronize across blocks is to use the global synchronization point at the end of every kernel execution
# 9. cudaDeviceSynchronize() can be used in CUDA/Kernels to sync all child grids launched in this block. However in fact the Kernel launch typically should be started by ne thread in thread block.
# 10. GPU DRAM access: If both L1 and L2 caches are used, a memory access is serviced by a 128-byte memory transaction.
# 11. GPU DRAM access: If L2 caches are used only , a memory access is serviced by a 32-byte memory transaction.
# 12. GPU L1 cache line size: 128 bytes
# 14. GPU DRAM paths: L1/L2 Cache, Constant Cache, Read-only cache.
# 15. The CPU L1 cache is optimized for both spatial and temporal locality. 
# 16. The GPU L1 cache is designed for spatial but not temporal locality. Frequent access to a cached L1 memory location does not increase the probability for cache hit.

# Speedups (fundamentally)
# 1. Instruction-level parallelism (ILP): More independent instructions within a thread (both memory and ALU)
# 2. Thread-level parallelism (TLP): More concurrently eligible threads
# 3. Unrolling in CUDA - Improving performance by reducing instruction overheads and creating more independent instructions to schedule. More concurrent operations are added to the pipeline leading to higher saturation of instruction and memory bandwidth. 
# 4. An optimal execution confi guration is a matter of striking a balance between latency hiding and resource utilization
# 5. The number of registers used by a kernel can have a signifi cant impact on the number of resident warps [see --maxregcount flag of nvcc]
# 6. Keep the number of threads per block a multiple of warp size (32).
# 7. Avoid small block sizes.
# 8. Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device.
# 9. cyclic partitioning: Each thread works on more than one data block
# 10. Using fewer registers in your kernels may allow more thread blocks to reside on an SM.
# 11. A basic principle of CUDA programming, you should always be thinking of ways to minimize hostdevice transfers. PCI-E 8GB/s, but in-device 144 GB/s.
# 12. Batching many small transfers into one larger transfer improves performance because it reduces per-transfer overhead.
# 13. Data transfers between the host and device can sometimes be overlapped with kernel execution.
# 14. You should either minimize or overlap data transfers between the host and device whenever possible.
# 15. Memory transaction effi ciency: Use the least number of transactions to service the maximum number of memory requests.
# 16. Minimize load of unrequested bytes.
# 17. Aligned and coalesced memory accesses that reduce wasted bandwidth.
# 18. Sufficient concurrent memory operations to hide memory latency (e.g. via loop unrolling)
# 19. For I/O-bound kernel exposing sufficient memory access parallelism is a high priority.


# CUDA profilers:
#  https://developer.nvidia.com/nsight-compute
#  https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual-profiler
#  https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof

# To Think About:
#  -- For a given kernel, trying different grid and block dimensions may yield better performance.
#=============================================================================================
