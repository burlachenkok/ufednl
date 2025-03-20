# u-FedNL: Advanced Configutations

----

The goal of this document describe how you can configure the build and use some extra tools.

----

# Options to Build Various Parts

These options are turned on and off in the main CMakeLists.txt configuration build file.
Are they turned `on` or `off`. With these options, you can turn `off` and `on`.

| **Option**                  | **Goal**                                                                                                                                                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DOPT_INCLUDE_UTILS          | Include build utils programs - dataset generators, system view, GPU viewers.                                                                                                                                               |
| DOPT_INCLUDE_UNITTESTS      | Add unit tests. The unit test covers almost all functionality.                                                                                                                                                             | 
| DOPT_CUDA_SUPPORT           | Build with CUDA support. CUDA is a C/C++ Language Extension from NVIDIA which is the main language to have the ability to write logic for NVIDIA GPU.                                                                      |
| DOPT_OPENCL_SUPPORT         | Build the project with OpenCL support. OpenCL is universal and supports various embedded and discrete GPUs across various vendors: Intel, NVIDIA, Apple, and ARM. However, the standard version which is supported varies. |
| DOPT_SWIG_INTERFACE_GENERATOR | Use SWIG to generate API and binding for Python and others supported by SWIG.                         |
| DOPT_BUILD_SHARED_LIBRARIES   | Build Shared clients and servers as share libraries including executable applications           |

In addition, several options are not part of the CMakeLists.txt but are part of the tools above it, however, they will help to improve build time:

* `project_script.py -use_ninja`  -- use Ninja build system (can be faster than GNU Make).
* `project_script.py -parallel_jobs` -- provide the ability to use a specific number of cores during the build.
* `project_script.py -unity_build` -- use unity build to speed up compilation time.

See also document `README_3_BUILD_LOCALLY.md`.

# Options that Affect Internal Logic. An alternative view is Implementation Variants.

These options are turned on and off in the main CMakeLists.txt configuration build file.
Are they turned `on` or `off` the buildable code obtain information about these C macroses.

| **Option**                                 | **Goal**                                                                                                     |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| DOPT_INCLUDE_VECTORIZED_CPU_IMP_VECS       | Build specialized SIMD vector implementation for dense vectors and vector views.                             |
| DOPT_INCLUDE_VECTORIZED_CPU_IMP_MATS       | Used specialized SIMD implementation for some internals of dense matrices                                    | 
| DOPT_DEBUG_BUILD                           | Are the current buildable binaries targeted to debug versions? In some code paths, it induces extra checks.  |
| DOPT_RELEASE_BUILD                         | Are the current buildable binaries targeted to be released in the latest version?                            |
| DOPT_INCLUDE_VECTORIZED_CPU_TRANSPOSE_MATS | Include vectorized implementation of matrices transpose in the build.                                      |                                          
| DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL         | Use implementation-dependent logic that exploits the fact that under technical cond. Hessians are symmetric. |    
| DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS  | Use kernel object semaphore during waiting updates from the clients in a local FedNL implementation.         |           
| DOPT_FIX_TOPK_CONTRACTION_FACTOR           | Fix computed contraction factor delta/alpha with subtle computation aspect w.r.t. to diagonal elements.      |          

# Options that are CPU and Instruction Set Architecture Specific

We try to detect which options are appropriate automatically, but please in case of problems manually select the needed options.

| **Option**                                 | **Goal**                                                                                                             |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| SUPPORT_CPU_FMA_EXT                        | Target CPU support x86/FMA3 instruction. And even more, code is encouraged to use it directly.                       | 
| SUPPORT_CPU_LOAD_STORE_PART                | Target CPU support partial store and load for partially operates on SIMD registers.                                  |
| SUPPORT_CPU_CPP_TS_V2_SIMD                 | Target compiler supports the C++ Extensions for Parallelism Version 2, ISO/IEC TS 19570:2018. Use e.g. for ARM Neon. |
| SUPPORT_CPU_SSE2_128_bits                  | Target CPU support SSE2 instruction set with 128-bit registers.                                                      |                                                
| SUPPORT_CPU_AVX_256_bits                   | Target CPU support AVX2 instruction set with 256-bit registers.                                                      |                                                
| SUPPORT_CPU_AVX_512_bits                   | Target CPU support AVX512 instruction set with 512-bit registers.                                                    |                                              

# Options which Affect Compiler and Liner C++ Program Behavior Itself

These are pretty low-level options to play with. Most people involved in Computer Games and systems Programming are aware of all of them.
However, even if you know C++, but use it outside high-performance domains you may not know all of them.
Outside we wrap up all options in a friendly way independent of toolchains (mainly compiler + linker) to build programs.

| **Option**                        | **Goal**                                                                                                                                                                                                                                              |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DOPT_USE_STATIC_CRT               | Use static CRT version                                                                                                                                                                                                                                | 
| REMOVE_RTTI_SUPPORT_CPP           | Remove RTTI - run-time type information from C++                                                                                                                                                                                                      |
| REMOVE_EXCEPTION_SUPPORT_CPP      | Remove exception support from C++                                                                                                                                                                                                                     |
| LINK_TIME_OPTIMIZATION_CPP        | Link time Optimization or Whole Program Optimization. Very strong toolchain optimization.                                                                                                                                                             | 
| COMPILE_TIME_OPTIMIZATION_USE_PCH | Use precompiled headers to improve compile time.                                                                                                                                                                                                      |
| COMPILE_TIME_OPT_OMIT_FRAME_PTR   | Compile time Optimization with omitting frame pointer. If use LINK_TIME_OPTIMIZATION_CPP this option is not needed. The compiler will have more information about the entire program, and it may decide to omit frame pointers based on its analysis. |

# Instrumentation

Please use these options if you want to carry out specific instrumentation and analysis of the program.

| **Option**                   | **Goal**                                                                                       |
|------------------------------|------------------------------------------------------------------------------------------------|
| OPT_CODE_COVERAGE_GCOV_IS_ON | Use GNU gcov code coverage tool                                                                |
| DOPT_LLVM_OPT_VIEWER         | Use CLANG/LLVM optimization remarks. Please turn off link time optimization manually.          |
| DOPT_EXTRA_DEBUG             | Turn on/off debugging and tracking various quantities. Currently discrepancy for linear solve. |
| DOPT_TRACE_DEBUG_PROJECT_BUILDING | Turn on debug information about the project building itself                                   |
| DOPT_VERBOSE_BUILD           | Turn on all verbose messages during build                                                      |

# Autogeneration Another Languages Binding Interfaces with SWIG

SWIG (Simplified Wrapper and Interface Generator) is a tool that helps create bindings between C/C++ code and various high-level programming languages.

## Installation for Ubuntu

```bash
sudo apt-get install swig python3-dev
```

## Installation for macOS
```bash
brew install swig
brew install python
```

## Installation for Windows OS

- Builds are available here: https://www.swig.org/download.html
- Install Python interpreter
- Install `python -m pip install python-dev-tools`

## Turn on/off option in CMakeLists.txt

The option in charge of generating Python binding for distributed shared library of functionality is located in `DOPT_SWIG_INTERFACE_GENERATOR`.


# Short Survey about Using Google Tests

Detailed information can be obtained from:
* https://github.com/google/googletest/blob/main/docs/advanced.md

* https://google.github.io/googletest/advanced.html

Below we provide a summary from official Google Documentation.

Google test filter uses `*` wildcard symbol to denote any sequence of characters. Exclusions of a name from the filter are identified by `-` sign. 
Finally, you can say multiple rules separated by `:`.

| **Command Line**      | **Meaning** |
|-----------------------|-------------|
|`./foo_test` | It has no flag and thus runs all its tests.|
| `./foo_test --gtest_filter=*` | Also runs everything, due to the single match-everything * value.|
|`./foo_test --gtest_filter=FooTest.*` | Runs everything in a test suite FooTest .|
|`./foo_test --gtest_filter=*Null*:*Constructor*` | Runs any test whose full name contains either "Null" or "Constructor".|
|`./foo_test --gtest_filter=-*DeathTest.*` | Runs all non-death tests.|
|`./foo_test --gtest_filter=FooTest.*-FooTest.Bar` | Runs everything in the test suite FooTest except FooTest.Bar.|
|`./foo_test --gtest_filter=FooTest.*:BarTest.*-FooTest.Bar:BarTest.Foo` | Runs everything in the test suite FooTest except FooTest.Bar and everything in the test suite BarTest except BarTest.Foo.|
| `./foo_test --gtest_list_tests`	| List available tests |
| `./foo_test --gtest_output="xml:test_report.xml" --gtest_filter=*` | Runs everything, and dumps result to XML file beside printed result to stdout. |
|  `::testing::GTEST_FLAG(filter) = "Utils.SomeTest"; (C++)` | Setup gtest_filter during runtime. Call it **before** ::testing::InitGoogleTest	|
|  `./foo_test --gtest_shuffle --gtest_filter=FooTest.*	` | Runs everything in test case FooTest, but shuffle the test order.	|
|  `./foo_test --gtest_filter=FooTest.*-FooTest.Bar` | Runs everything in test case FooTest except FooTest.Bar. |
| `./foo_test --gtest_filter=*Null*:*Constructor*`	| Runs any test whose full name contains either "Null" or "Constructor."	|
|  `./foo_test --gtest_repeat=10 --gtest_filter=-*DeathTest.*` |  Runs all except tests from test case DeathTest. And run them 10 times. |

In source code in C++ level you can pass different flags. Filter setup via the command-line (if they are presented) has priority over filter setup in the source code. Finally, to exclude a test from execution, append `DISABLED_` prefix to its name.
