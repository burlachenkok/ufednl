#include <stdio.h>

#include "dopt/system/include/CpuInfo.h"
#include "dopt/system/include/CompilerInfo.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include "gtest/gtest.h"
#include  <time.h>

#if DOPT_CUDA_SUPPORT
#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#endif

/** In Google Test:
* 
*    1. Filter setup via comdline ( i.e. with `--gtest_filter` ) has priority over filter setup in source code.
*    2. With flag: `--gtest_list_tests` it's possible to get list of tests
*    3. Google test filter uses `*` wildcard symbol to denote any sequence of characters.
*    4. Finally, you can use multiple rules seperated by `:`. The symbol `:` connects rules in logical and.
*    5. Exclusions of a name from filter is identified by `-`` sign. The minus sign `-` affects all rules spearated by `:`.
*
*  Remark:
* 
*    To exclude a test from execution, append `DISABLED_` prefix to its name.
*/

int main(int argc, char** argv)
{
    // ::testing::GTEST_FLAG(break_on_failure) = true;
    //::testing::GTEST_FLAG(output) = "xml:./unit_and_perf_tests.xml";
    ::testing::GTEST_FLAG(repeat) = 1;
    ::testing::GTEST_FLAG(color) = "yes";
    ::testing::GTEST_FLAG(filter) = "*";

    ::testing::InitGoogleTest(&argc, argv);

    const clock_t timestampStartTests = clock();

    int result = RUN_ALL_TESTS();

    const clock_t timestampEndTests = clock();

    double deltaSeconds = (timestampEndTests - timestampStartTests) / double(CLOCKS_PER_SEC);

    printf("=======================================================================\n");
    printf("Binary: %s\n", argv[0]);
    printf("Time spent to execution unit tests: %lf seconds\n", deltaSeconds);
    printf("Filter for test execution: %s\n", ::testing::GTEST_FLAG(filter).c_str());
    printf("\n");
    printf("Size of data 'pointer' in bytes: %i\n", static_cast<int>(sizeof(void*)));
    printf("Size of data with 'int' type in bytes: %i\n", static_cast<int>(sizeof(int)));
    printf("Size of data with 'long' type in bytes: %i\n", static_cast<int>(sizeof(long)));
    printf("\n");

#if DOPT_DEBUG_BUILD && !DOPT_RELEASE_BUILD
    printf("Build type: DEBUG\n");
#elif !DOPT_DEBUG_BUILD && DOPT_RELEASE_BUILD
    printf("Build type: RELEASE\n");
#else
    printf("Build type: UNDEFINED\n");
#endif

#if DOPT_USE_STATIC_CRT
    printf("Compiled with: static CRT\n");
#else
    printf("Compiled with: dynamic CRT\n");
#endif
    

#if SUPPORT_CPU_SSE2_128_bits
    printf("Compiled with: SSE2 - 128 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_AVX_256_bits
    printf("Compiled with: AVX2 (AVX256) - 256 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_AVX_512_bits
    printf("Compiled with: AVX512 - 512 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_CPP_TS_V2_SIMD
    printf("Compiled with: C++ Extensions for Parallelism V2\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#else
    printf("Compiled with: No special CPU operations\n");
#endif
    
    printf("Target Architecture: %s\n", DOPT_ARCH_NAME);
    
#if DOPT_CUDA_SUPPORT
    printf("Compiled with: support CUDA for GPU [YES]\n");
    
    printf(" > Number of available CUDA compatible GPUS: %i\n", dopt::GpuManagement::getNumberOfInstalledGPU());
    
    std::string defGPUDescr = dopt::GpuManagement::getShortDeviceDescription(dopt::GpuManagement::defaultGPUDevice());
    printf(" > Default GPU: %s\n", defGPUDescr.c_str());
#else
    printf("Compiled with: support CUDA for GPU [NO]\n");
#endif

#if DOPT_OPENCL_SUPPORT
    printf("  Compiled with: support OpenCL for GPU/CPU [YES]\n");
#else
    printf("  Compiled with: support OpenCL for GPU/CPU [NO]\n");
#endif

    printf("Date and time for build unittest: " __DATE__ "/" __TIME__ "\n");

#if DOPT_WINDOWS
    printf("Operating System: Windows\n");
#elif DOPT_LINUX
    printf("Operating System: Linux\n");
#elif DOPT_MACOS
    printf("Operating System: macOS\n");
#else
    printf("Operating System: Unknown\n");
#endif

    printf("=======================================================================\n");
    printf("Infromation about installed CPU                                        \n");
    printf("\n");
    printf(" Physical Cores: %i\n", dopt::physicalProcessorsInSystem());
    printf(" Logical Cores: %i\n", dopt::logicalProcessorsInSystem());
    printf("=======================================================================\n");
    printf(" Return value from unit tests: %i\n", result);
    printf("=======================================================================\n");
    printf("  __cplusplus version: %i\n", int(__cplusplus));
    printf("  compiler:name: %s\n", dopt::compilerCppVersion().c_str());
    printf("  SIMD support at compile time: %s\n", dopt::isSimdComputeSupportedAtCompileTime() ? "[YES]" : "[NO]");
    printf("=======================================================================\n");

    // Sleep
    //getchar();
    return result;
}
