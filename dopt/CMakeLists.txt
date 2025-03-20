cmake_minimum_required(VERSION 3.12)

# Policy regarding SWIG
cmake_policy(SET CMP0086 NEW)
cmake_policy(SET CMP0078 NEW)

# Policy regarding CUDA
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

#===================================================================================================================
# https://cmake.org/cmake/help/latest/prop_gbl/USE_FOLDERS.html
# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#===================================================================================================================
# CUDA TOOLS [IGNORE FOR NON-CUDA/NON-OPENCL BUILDS]
#===================================================================================================================
set(CUDA_PATH_WINDOWS_DEF "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4") # CUDA Toolkit path in Windows
set(CUDA_PATH_LINUX_DEF "/sw/rl9g/cuda/12.2/rl9_binary")                              # CUDA Toolkit path in Linux
set(CMAKE_CUDA_ARCH_BIN "60")                                                         # Manually select target NVIDIA GPU architeture

#===================================================================================================================
# START OF CONFIGUTATION
#===================================================================================================================
# Debug Building Itself
option(DOPT_TRACE_DEBUG_PROJECT_BUILDING  "Turn on debug information about project building" OFF)

# Build configutation: Build configuration for C++
option(DOPT_USE_STATIC_CRT                "Use static CRT version" OFF)
option(REMOVE_RTTI_SUPPORT_CPP            "Remove RTTI - run time type information from C++" ON)
option(REMOVE_EXCEPTION_SUPPORT_CPP       "Remove exception support from C++" ON)
option(LINK_TIME_OPTIMIZATION_CPP         "Link time Optimization or Whole program Optimization" ON)
option(COMPILE_TIME_OPTIMIZATION_USE_PCH  "Use precompiled headers to improve compile time" OFF)
option(DOPT_VERBOSE_BUILD                 "Turn on all verbose messages during build" ON)

# If use Whole Program Optimization the compiler will have more information about the entire program, and it may decide to omit frame pointers based on its analysis.
# Generally don't need to explicitly enable or disable frame pointer omission.
option(COMPILE_TIME_OPT_OMIT_FRAME_PTR    "Compile time Optimization with omitting frame pointer" OFF)

# Build configutation: What to include into build
option(DOPT_INCLUDE_UNITTESTS   "Include unittests into build" ON)
option(DOPT_INCLUDE_UTILS       "Include utils into build" ON)

# Build configutation: Compute features
option(DOPT_CUDA_SUPPORT           "Build with CUDA support" OFF)
option(DOPT_OPENCL_SUPPORT         "Build with OPENCL support" OFF)

option(SUPPORT_CPU_SSE2_128_bits   "Target CPU support SSE2 instruction set with 128 bits registers"   OFF)
option(SUPPORT_CPU_AVX_256_bits    "Target CPU support AVX2 instruction set with 256 bits registers"   ON )
option(SUPPORT_CPU_AVX_512_bits    "Target CPU support AVX512 instruction set with 512 bits registers" OFF)
option(SUPPORT_CPU_CPP_TS_V2_SIMD  "Target compiler support C++ SIMD Extension. Use it." OFF)

option(SUPPORT_CPU_FMA_EXT         "Target CPU support x86/FMA3 instruction" OFF)
option(SUPPORT_CPU_LOAD_STORE_PART "Use store and load partial functionality for SIMD code instead of usual CPU code" OFF)
#===================================================================================================================

# TODO: Eliminate link time optimization for CUDA projects, NVIDIA tools incompatible with Whole Program Otpimization
if (DOPT_CUDA_SUPPORT)
    set(LINK_TIME_OPTIMIZATION_CPP OFF)
endif()

# Build configutation: Implementation Variants
#====================================================================================================================================================================
option(DOPT_INCLUDE_VECTORIZED_CPU_IMP_VECS        "Include into the build vectorized implementation of dense vectors" ON)
option(DOPT_INCLUDE_VECTORIZED_CPU_IMP_MATS        "Include into the build vectorized implementation of dense matrices" ON)
option(DOPT_INCLUDE_VECTORIZED_CPU_TRANSPOSE_MATS  "Include into the build vectorized implementation of matrices tranpose" ON)
option(DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL          "Use implementation dependent logic that exploits the fact that under technical cond. hessians are symmetric" OFF)
option(DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS   "Use kernel object sempaphore during waiting updates from the clients in a local FedNL implementation" OFF)
option(DOPT_FIX_TOPK_CONTRACTION_FACTOR            "Fix computed contraction factor delta/alpha with subtle computation aspect w.r.t. to diagonal elements" ON)
#===================================================================================================================

# Build configutation:  Instrumentation
#===================================================================================================================
option(DOPT_EXTRA_DEBUG                            "Turn on debugging and tracking various quantities. Currently disrepancy for linear solve and difference between learnable and centralized hessians for FEDNL" OFF)
option(OPT_CODE_COVERAGE_GCOV_IS_ON                "Use GNU gcov code coveraging tool" OFF)
option(DOPT_LLVM_OPT_VIEWER                        "Use LLVM optimization remarks. Please turn off link time optimization manually." OFF)

# Build configutation:  Wrappers
#===================================================================================================================
option(DOPT_SWIG_INTERFACE_GENERATOR               "Use SWIG to generate API for library." OFF)
option(DOPT_BUILD_SHARED_LIBRARIES                 "Build Shared Libraries." OFF)

# LLVM Optimization Remarks - Helping the Compiler Generate Better Code - Ofek Shilon - CppCon 2022 06 January 2022: https://www.youtube.com/watch?v=qmEsx4MbKoc&t=2407s
# LLVM Optimizer Viewer project: https://github.com/OfekShilon/optview2
# How to use: 
#  Step-1: Turn option ON and build project with CLANG
#  Step-2: python3 opt-viewer.py -j10 --output-dir ./out --source-dir ./../project_name/build_release/
#
# Remark: The tool provides some way of dialog with the source writer and compiler.
#===================================================================================================================
# END OF CONFIGUTATION
#===================================================================================================================

if(DOPT_TRACE_DEBUG_PROJECT_BUILDING)
    # Turn on for debugging purposes
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

project("dopt" LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake)
include(HelpFunctions)
include(ToolChainConfiguration)

set(DOPT_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

messageNormal("============================================================")
messageNormal("PROJECT_ROOT: " ${DOPT_PROJECT_ROOT})
messageNormal("OUTPUT BINARY DIRECTORY: " ${CMAKE_CURRENT_BINARY_DIR})

messageNormal("BUILD TYPE: " ${CMAKE_BUILD_TYPE})
messageNormal("PROJECT NAME: " ${PROJECT_NAME})
messageNormal("BUILD PROJECT WITH CUDA SUPPORT: " ${DOPT_CUDA_SUPPORT})
messageNormal("BUILD PROJECT WITH OPENCL SUPPORT: " ${DOPT_OPENCL_SUPPORT})

messageNormal("CMAKE VERSION: " ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
messageNormal("CMAKE VERSION: " ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
messageNormal("CMAKE DEFAULT MODULE PATH (include, find_package): " ${CMAKE_MODULE_PATH})
messageNormal("CMAKE INSTALLATION DIRECTORY: " ${CMAKE_ROOT})

messageNormal("C++ COMPILER: " ${CMAKE_CXX_COMPILER})
messageNormal("C COMPILER: " ${CMAKE_C_COMPILER})
messageNormal("C++ STANDARD: " ${CXX_STANDARD})
messageNormal("TRACE BUILD COMMANDS: " ${DOPT_TRACE_DEBUG_PROJECT_BUILDING})
messageNormal("============================================================")

#==============================================================================================================
# Turn off x86 extension for ARM exlpicitly they are not supported in ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch")
    messageNormal("Target CPU Architecture: ${CMAKE_SYSTEM_PROCESSOR}. Turning off x86_64 extensions")
    set(SUPPORT_CPU_SSE2_128_bits OFF)
    set(SUPPORT_CPU_AVX_256_bits OFF)
    set(SUPPORT_CPU_AVX_512_bits OFF)
else()
    messageNormal("Target Architecture Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Setup variable based on build configuration
#==============================================================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DOPT_DEBUG_BUILD 1)
else()
    set(DOPT_DEBUG_BUILD 0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(DOPT_RELEASE_BUILD 1)
else()
    set(DOPT_RELEASE_BUILD 0)
endif()

# Based on type of C++ runtime selected for project specify option for build unittests and protobuf library
#==============================================================================================================
if(DOPT_USE_STATIC_CRT)
    set(gtest_force_shared_crt 0)
else()
    set(gtest_force_shared_crt 1)
endif()  

if (DOPT_INCLUDE_UNITTESTS)
    add_subdirectory(dopt/3rdparty/gtest-1.12.1)
endif()

if (DOPT_CUDA_SUPPORT)

    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH_WINDOWS_DEF} CACHE FILEPATH "")

    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH_LINUX_DEF} CACHE FILEPATH "")

        # set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
        # set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Macos" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH_LINUX_DEF} CACHE FILEPATH "")

        # set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
        #  set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
    endif()

    if (DOPT_TRACE_DEBUG_PROJECT_BUILDING)
        #For debugging of builds
        set(CUDA_VERBOSE_BUILD 1)
    endif()

    set(CUDA_NVCC_FLAGS "-std=c++${CXX_STANDARD}")

    # -G -- Generate debug information for device code. 
    # -g -- Generate debug information for host code. (maybe not needed)
    # -O -- Specify optimization level for host code. (maybe not needed)
    set(CUDA_NVCC_FLAGS_DEBUG -G)

    # -lineinfo -- Generate line-number information for device code.
    # -O3       -- Specify optimization level for host code. (maybe not needed)
    set(CUDA_NVCC_FLAGS_RELEASE -lineinfo)

    # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
    find_package(CUDA REQUIRED)

    # Automatically select architeture:
    #cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS "Auto")

    #   Auto: detects local machine GPU compute arch at runtime.
    #   All: cover common and entire subsets of architectures.

    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
    message(STATUS "+ CUDA Flags: ${CUDA_NVCC_FLAGS}")
    message(STATUS "+ CUDA Version: ${CUDA_VERSION}")
    message(STATUS "+ CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")

endif()

if (DOPT_OPENCL_SUPPORT)

    if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(OpenCL_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
    else()
        # None
    endif()

    find_package(OpenCL REQUIRED)

endif()


if(DOPT_SWIG_INTERFACE_GENERATOR)

    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(SWIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dopt/3rdparty/windows/swigwin-4.2.1/Lib")
        set(SWIG_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/dopt/3rdparty/windows/swigwin-4.2.1/swig.exe")
    endif()


    find_package(SWIG REQUIRED) # Ensures SWIG is found on the system.
    include(UseSWIG)            # Loads CMake's SWIG support.
    set(CMAKE_SWIG_FLAG "")

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

endif()


# Print extra information
#==============================================================================================================
printInfo()


# Add all subprojects
#==============================================================================================================
add_subdirectory(dopt/cmdline)
add_subdirectory(dopt/timers)
add_subdirectory(dopt/fs)
add_subdirectory(dopt/random)
add_subdirectory(dopt/copylocal)
add_subdirectory(dopt/linalg_vectors) 
add_subdirectory(dopt/linalg_matrices)
add_subdirectory(dopt/linalg_linsolvers) 
add_subdirectory(dopt/math_routines)
add_subdirectory(dopt/numerics)
add_subdirectory(dopt/system) 
add_subdirectory(dopt/optimization_problems) 

# Add GPU compute support
if(DOPT_CUDA_SUPPORT)
    add_subdirectory(dopt/gpu_compute_support)
endif()

# Add unittests
if (DOPT_INCLUDE_UNITTESTS)
    add_subdirectory(bin_tests)
endif()

# Add utils
if (DOPT_INCLUDE_UTILS)
    add_subdirectory(utils/bin_opt_problem_generator)
    add_subdirectory(utils/bin_host_view)

    if(DOPT_CUDA_SUPPORT)
        add_subdirectory(utils/bin_cuda_view)
    endif()

    if(DOPT_OPENCL_SUPPORT)
        add_subdirectory(utils/bin_opencl_view)
    endif()

endif()
#==============================================================================================================

# Add solvers
#==============================================================================================================
add_subdirectory(bin_fednl_local)
add_subdirectory(bin_fednl_distr_master)
add_subdirectory(bin_fednl_distr_client)

add_subdirectory(bin_fednl_local_pp)
add_subdirectory(bin_fednl_distr_master_pp)
add_subdirectory(bin_fednl_distr_client_pp)

add_subdirectory(bin_split)
#==============================================================================================================
