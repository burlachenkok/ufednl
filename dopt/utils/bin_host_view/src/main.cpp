#include "dopt/system/include/ProcessInfo.h"
#include "dopt/system/include/CpuInfo.h"
#include "dopt/system/include/MemInfo.h"
#include "dopt/system/include/CompilerInfo.h"
#include "dopt/system/include/Version.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/system/include/ProcessInfo.h"
#include "dopt/cmdline/include/CmdLineParser.h"
#include "dopt/system/include/Version.h"

#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include  <time.h>

int main(int argc, char** argv)
{
    dopt::CmdLine cmdline(argc, argv);

    dopt::printInformationAboutBuild(argv[0]);

    printf("  Size of data 'pointer' in bytes: %i\n", static_cast<int>(sizeof(void*)));
    printf("  Size of data with 'int' type in bytes: %i\n", static_cast<int>(sizeof(int)));
    printf("  Size of data with 'long' type in bytes: %i\n\n", static_cast<int>(sizeof(long)));

#if DOPT_USE_STATIC_CRT
    printf("  Compiled with: static CRT\n");
#else
    printf("  Compiled with: dynamic CRT\n");
#endif

    
#if DOPT_CUDA_SUPPORT
    printf("  Compiled with: support CUDA for GPU [YES]\n");
#else
    printf("  Compiled with: support CUDA for GPU [NO]\n");
#endif

#if DOPT_OPENCL_SUPPORT
    printf("  Compiled with: support OpenCL for GPU/CPU [YES]\n");
#else
    printf("  Compiled with: support OpenCL for GPU/CPU [NO]\n");
#endif

    printf("  Date and time for build unittest: " __DATE__ "/" __TIME__ "\n");

#if DOPT_WINDOWS
    printf("  Operating System: Windows\n");
#elif DOPT_LINUX
    printf("  Operating System: Linux\n");
#elif DOPT_MACOS
    printf("  Operating System: macOS\n");
#else
    printf("  Operating System: Unknown\n");
#endif
    
    printf("=======================================================================\n");
    printf("Infromation about used vector registers in CPU                         \n");
    printf("=======================================================================\n");
#if SUPPORT_CPU_SSE2_128_bits
    printf("  Compiled with: SSE2 - 128 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("  SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_AVX_256_bits
    printf("  Compiled with: AVX2 (AVX256) - 256 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("  SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_AVX_512_bits
    printf("  Compiled with: AVX512 - 512 bits SIMD\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("  SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#elif SUPPORT_CPU_CPP_TS_V2_SIMD
    printf("  Compiled with: C++ Extensions for Parallelism V2\n");
    size_t kSimdSizeInBits = (dopt::VectorSimdTraits<double, dopt::cpu_extension>::VecType::size()) * sizeof(double) * 8;
    printf("  SIMD vector size for FP64: %i Bits\n", int(kSimdSizeInBits));
#else
    printf("  Compiled with: No special CPU operations\n");
#endif

    printf("  Target Architecture: %s\n", DOPT_ARCH_NAME);

    printf("=======================================================================\n");
    printf("Infromation about installed CPU                                        \n");
    printf("=======================================================================\n");
    printf("  Physical Cores: %i\n", dopt::physicalProcessorsInSystem());
    printf("  Logical Cores: %i\n", dopt::logicalProcessorsInSystem());
    printf("  Architecture Name: %s\n", DOPT_ARCH_NAME);
    
    uint64_t baseFreq = 0, maxFreq = 0;
    if (dopt::getCPUBaseAndMaxFrequencyInMhz(baseFreq, maxFreq)) 
    {
        printf("  Frequency of the core is in the range: [%i Ghz - %i Ghz]\n", int(baseFreq/1024), int(maxFreq/1024));
    }

    printf("  CPU cache size (C++): %i Bytes\n", dopt::cacheLineSizeForProcessorFromCpp());
    printf("  CPU cache size  (OS): %i Bytes\n", dopt::cacheLineSizeForProcessorFromOS());
    printf("        Page size (OS): %i Bytes\n", dopt::virtualPageSize());

    printf("=======================================================================\n");
    printf("Infromation about available CPU Extensions                             \n");
    printf("=======================================================================\n");    
    dopt::printExtensionForInstalledCPU([](const char* msg) {printf("  %s", msg); });
    printf("=======================================================================\n");
    printf("Compiler Information                                                    \n");
    printf("=======================================================================\n");
    printf("  compiler:name: %s\n", dopt::compilerCppVersion().c_str());
    printf("  compiler:size of an empty class is not zero: %s\n", dopt::compilerSizeOfEmptyIsNotNull() ? "[OK]" : "[NO]");
    printf("  compiler:different objects have different addresses: %s\n", dopt::compilerDifferentObjectsAddrDiffer() ? "[OK]" : "[NO]");
    printf("  compiler:long double equal to size of double: %s\n", dopt::isLongDoubleSameAsDouble() ? "[YES]" : "[NO]");
    printf("  runtime check. compiler:byte order in processor is right 2 left (little endian, Intel): %s\n", dopt::isByteOrderRight2Left() ? "[YES]" : "[NO]");
    printf("  runtime check. compiler:byte order in processor is left 2 right (big endian, Motorola): %s\n", dopt::isByteOrderLeft2Right() ? "[YES]" : "[NO]");
    
    printf("  compiletime order is consistent. compiler:byte order in processor is right 2 left (little endian): %s\n", (dopt::isByteOrderRight2Left() == DOPT_ARCH_LITTLE_ENDIAN) ? "[OK]" : "[ERROR!]");
    printf("  compiletime order is consistent. compiler:byte order in processor is left 2 right (big endian): %s\n", (dopt::isByteOrderLeft2Right() == DOPT_ARCH_BIG_ENDIAN) ? "[OK]" : "[ERROR!]");

    printf("  compiler:char type constist of 8 bits: %s\n", dopt::isCharConsistOf8Bits() ? "[YES]" : "[NO]");
    printf("  compiler:Matryoshka layout - derived class and first member of base share identical address: %s\n", dopt::memLayoutIsMatryoshka() ? "[YES]" : "[NO]");
    printf("  compiler:char type is signed: %s\n", dopt::compilerIsCharTypeSigned() ? "[YES]" : "[NO]");
    printf("  compiler:empty base class need not be represented by a separate byte: %s\n", dopt::compilerOptimizedEmptyBaseClass() ? "[YES]" : "[NO]");
    printf("  compiler:__cplusplus: %i\n", int(__cplusplus));
    printf("  compiler:SIMD support at compile time: %s\n", dopt::isSimdComputeSupportedAtCompileTime() ? "[YES]" : "[NO]");
    printf("=======================================================================\n");
    printf("Infromation about physical DRAM Memory                                  \n");
    printf("=======================================================================\n");
    printf("  Amount of DRAM memory: %i GBytes\n", int(dopt::installedPhysicalMemoryInBytes() / 1024/1024/1024));
    printf("  Amount of free DRAM memory: %i GBytes\n", int(dopt::availablalePhysicalMemoryInBytes() / 1024 / 1024 / 1024));
    const dopt::ProcessStatistics stats = dopt::getProcessStatistics();
    printf("=======================================================================\n");
    printf("Memory statistics for process\n");
    printf("=======================================================================\n");
    printf("  Used Memory\n");
    printf("   - images of binary files: %i KBytes\n", int(stats.memoryForImages / 1024));
    printf("   - memory for maped files: %i KBytes\n", int(stats.memoryForMappedFiles / 1024));
    printf("   - private memory for process: %i KBytes\n\n", int(stats.memoryPrivateForProcess / 1024));

    printf("  Physical Memory for process for store anything needed: %i KBytes\n", 
              int(dopt::physicalMemoryForProcess() / 1024) );
    printf("  Virtual and Physical Committed Memory for process: %i KBytes\n", 
              int(dopt::totalVirtualAndPhysicalMemoryForProcess() / 1024) );
    printf("=======================================================================\n");
    printf("  Process ID: %u\n", unsigned(dopt::currentProcessId()));
    printf("  Thread ID: %u\n", unsigned(dopt::currentThreadId()));
    printf("=======================================================================\n");
    printf("Compiletime Exra Information                                           \n");
    printf("=======================================================================\n");
    printf(" compiler: CPU Architecture (from CMake): %s\n", dopt::cmakeTargetCPUArch);
    printf(" compiler: C++ Compiler path: %s\n", dopt::cmakeToolchainCppCompiler);
    printf(" compiler: C++ Compiler flags: %s\n", dopt::cmakeCompilerCppBuildFlags);
    printf(" linker: Linker flags: %s\n", dopt::cmakeLinkerExecutableBuildFlags);

    //printf()
    if (cmdline.isFlagSetuped("wait-for-input"))
        getchar();

    return 0;
}
