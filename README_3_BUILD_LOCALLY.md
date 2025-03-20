# u-FedNL: Build Locally

----

The goal of this document is to describe how after previous steps to build a project locally in your Operating System.

----

# Building in IDE
* In [CLion](https://www.jetbrains.com/clion/) or [QtCreator](https://doc.qt.io/qtcreator/) open `dopt/CMakeLists.txt` and build within them `Debug` or `Release` configurations.
* For [Visual Studio](https://visualstudio.microsoft.com/) you can open generated project solution. To generate it use the following:
```
cd ./dopt/scripts
./project_scripts.py -c -gr
```

# Building and More via Console

## Reference Table

To assist with various subtle aspects of the building process we have created a helper script `project_scripts.py`. Below are descriptions of the flags and you specify as many flags as you wish.
**Note:** At this moment it may be a bit overwhelming because you do not need all of them. However, you may wish to return to this table occasionally.

The table itself can be obtained via:
```
cd ./dopt/scripts
./project_scripts.py -h
```

| #   | Bootstrap script     | Long Argument       | Short Argument | Description                                                        |
|-----|----------------------|---------------------|----------------|--------------------------------------------------------------------|
| 1.  | project_scripts.py   | -help               | -h             | to print this information text into console                        |
| 2.  | project_scripts.py   | -support_nvidia_gpu | -nv            | generate project files with NVIDIA CUDA GPU support                |
| 3.  | project_scripts.py   | -support_opencl     | -ocl           | generate project files with OpenCL GPU support                     |
| 4.  | project_scripts.py   | -generate_debug     | -gd            | generate debug version                                             |
| 5.  | project_scripts.py   | -generate_release   | -gr            | generate release version                                           |
| 6.  | project_scripts.py   | -build_debug        | -bd            | build debug version                                                |
| 7.  | project_scripts.py   | -build_release      | -br            | build release version                                              |
| 8.  | project_scripts.py   | -clean              | -c             | cleanup folder with build artifacts                                |
| 9.  | project_scripts.py   | -tests_debug        | -td            | launch debug unit tests                                            |
| 10. | project_scripts.py   | -tests_release      | -tr            | launch release unit tests                                          |
| 11. | project_scripts.py   | -info_debug         | -id            | launch debug system info                                           |
| 12. | project_scripts.py   | -info_release       | -ir            | launch release system info                                         |
| 13. | project_scripts.py   | -vc_version         | -vs            | provide specific version of Visual Studio (2022,2019,...)          |
| 14. | project_scripts.py   | -use_clang_for_vc   | -uc            | use clang toolset for Visual Studio (2022,2019,...)                |
| 15. | project_scripts.py   | -use_ninja          | -un            | use Ninja build system (can be faster than GNU Make)               |
| 16. | project_scripts.py   | -parallel_jobs      | -j             | use a specific number of CPU cores in the building process         |
| 17. | project_scripts.py   | -unity_build        | -ub            | use unity build to speedup compilation time                        |
| 18. | project_scripts.py   | -use_pch            | -up            | use precompile headers o speedup compilation time (g++ has issues) |
| 19. | project_scripts.py   | -gnu_code_coverage  | -gc            | turn on generating code coverage                                   |
| 20. | project_scripts.py   | -llvm_opt_viewer    | -ov            | turn on LLVM optimization compilers remarks to study them          |
| 21. | project_scripts.py   | -doxygen            |                | generate doxygen automatic documentation                           |

## How to: Build Release Implementation and Run Tests

The first step is to switch to the directory and then invoke the need commands:
```bash
cd ./dopt/scripts
```

Then you can to build a project with the following command line:

`./project_scripts.py -c -gr -br -tr -j 48`
* [-c]  Clean build directory
* [-gr] Generate project files for release build
* [-bd] Start release builds
* [-tr] After the build is finished launch unit tests for functionality
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

**Note:** This command will select some default C and C++ compilers from your system. This choice is made by CMake.


## How to: Build Debug Implementation and Debug Tests

The first step is to switch to the directory and then invoke the need commands:
```bash
cd ./dopt/scripts
```

Then you can build a project with the following command line:

`./project_scripts.py -c -gd -bd -td -j 8`
* [-c]  Clean build directory
* [-gr] Generate project files for release build
* [-bd] Start release builds
* [-tr] After the build is finished launch unit tests for functionality
* [-j 8] During the compilation process allow the build tool to create a pool of compilation processes equal to 8

**Note:** This command will select some default C and C++ compilers from your system. This choice is made by CMake.

##  How to: Clean the Folder from Any Local Artefacts

Clean working directory: `./project_scripts.py -c`

The project can be built with vectorized register support for two most popular CPU compute architectures:
* `AArch64`
* `x86_64`

**Note:** Once you build a project for `ARM/AArch64` the building with `AVX` and `SSE2` will be automatically turned off
with logic from `./dopt/CMakeLists.txt`. In this context, it does not make sense.

## How to: Build Debug and Release Implementations

The following two commands separately build debug and release versions. They will utilize only 1 CPU core.

```bash
./project_scripts.py -gd -bd
./project_scripts.py -gr -br
```

##  How to: Improve Build Time with Ninja

`./project_scripts.py -c -un -gd -bd -j 48`
* [-c]  Clean build directory
* [-un] Utilize Ninja build system
* [-gd] Generate project files for debug build
* [-bd] Start debug builds
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

##  How to: Improve Build Time with (Jumbo) Unity Build

`./project_scripts.py -c -ub -gr -br -j 48`
* [-c]  Clean build directory
* [-ub] Use Unity build to speed up compilation time. It is a compile optimization technique where multiple source files are combined into a single large source file before compilation.
* [-gr] Generate project files for release build
* [-br] Start release builds
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

##  How to: Build with custom C and C++ Compilers

The environment variables `CXX` and `CC` are commonly used in various build systems including GNU Make, CMake, Visual Studio, etc.
They specify the C++ and C compilers used, respectively. And typically it's happening in all Operating Systems.

```bash
export CXX="/usr/bin/clang++-17"
export CC="/usr/bin/clang-17"
./project_scripts.py -c -gd -bd -j 48
```

* [-c]  Clean build directory
* [-gd] Generate project files for release build
* [-bd] Start debug build
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

##  How to: Invoke Generation of Documentation

The following command invokes automatically documentation generation:

```
./project_scripts.py -doxygen
```

##  How to: Build with OpenCL Support

If your compute platform supports OpenCL you build one utility to view the platform and project itself using OpenCL compute kernels in the following way:
```
./project_scripts.py -gd -bd -ocl -j8
```

##  How to: Build with CUDA Support

If your compute platform supports CUDA you build one utility to view the platform and project itself using OpenCL compute kernels in the following way:
```
./project_scripts.py -gd -bd -cuda -j8
```

##  How to: Pass extra arguments for CMake and more Sublte Control

For more subtle control over the build process, you may wish to pass several options to CMake:
```
cmake -D<OPTION NAME>=1 source-dir
```

To accomplish this you need to provide all space-separated opinions as line in:
```
EXTRA_CMAKE_ARGS
```


# Appendix

##  Build with Optimization Remarks

```bash
export CXX="/usr/bin/clang++-17"
export CC="/usr/bin/clang-17"
./project_scripts.py -c -ov -gd -bd -j 48
```

* [-c]  Clean build directory
* [-ov] LLVM optimization compilers remarks to study them
* [-gd] Generate project files for release build
* [-bd] Start debug build
* [-j 48] During the compilation process allow the build tool to create a pool of compilation processes equal to 48

If your research includes reporting about passed seconds for your algorithm in your field and you need to report this number,
and you are using C, C++, Java, Fortran, Rust, Scala, Swift, or any other language supported by a set of compiler and toolchain technologies LLVM (https://llvm.org/)
then the following information can be interesting for you because it can assist you with polishing your implementation and gain extra speedup.

The LLVM Clang compiler can produce a special form of feedback information for the author of the code in the form of output files in the format `*.opt.yaml`.

Essentially, LLVM optimization compilers' remarks offer a dialog between:
(1) Person who writes algorithms in high-level languages (Higher than Assembly, but lower than Scripting Languages)
(2) Compiler that converts high-level instruction into instructions for a specific Instruction Set Architecture (ISA) of Computing Device or another machine-dependent representation.

With this tool, it's possible to obtain information that the compiler tried to optimize the source code, but it failed due to the writer violates
some subtle principles of the Language that touch on questions about code optimization.

After building the program we suggest using one of the available tools to read the remarks. One of such tool is [OptViewer](https://github.com/OfekShilon/optview2)
