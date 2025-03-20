#pragma once

#include "dopt/system/include/CompilerInfo.h"
#include "dopt/system/include/Version.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

template <class TStream>
inline bool preambleForViewer(TStream& out)
{
    out << "GENERAL INFORMATION\n";
#if defined(DOPT_WINDOWS)
    out << " Operation System: Windows\n";
#elif defined(DOPT_LINUX)
    out << " Operation System: Linux\n";
#elif defined(DOPT_MACOS)
    out << " Operation System: macOS\n";
#endif
    out << " Host Compiler Name: " << dopt::compilerCppVersion().c_str() << '\n';
    out << " Cplusplus version: " << int(__cplusplus) << '\n';
    out << " Compiletime. CUDA Toolkit version:" << CUDA_VERSION / 1000 << "." << (CUDA_VERSION / 10) % 100 << '\n';
    out << " Compiletime. CUDA Runtime version:" << CUDART_VERSION / 1000 << "." << (CUDART_VERSION / 10) % 100 << '\n';
    out << " Compiletime. CUDA Toolkit path: " << dopt::cmakeCudaToolkitPath << '\n';
    
    out << '\n';
    out << " Build data and time: " << __DATE__ << "/" << __TIME__ << '\n';
    out << '\n';

    return true;
}
