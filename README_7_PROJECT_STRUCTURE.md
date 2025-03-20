# u-FedNL: Project Structure of FedNL

----

The goal of this document is to provide high-level guidelines for parts of the project.

----

# Components of Self-Contained Framework

| **Component Name**                        | **Goal**                                                                                                                                                                              | **Type**       |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| dopt/CMakeLists.txt                       | Main root CMake project configuration file.                                                                                                                                           | Configuration  |
| dopt/scripts/project_scripts.py           | Script to help launch builds, test, docs, and run some utils (in a platform where the project was built)                                                                              | Script         |
| dopt/scripts/doxygen/doxygen_generate.py  | Automatically generate documentation for the project from doxygen annotated source code                                                                                               | Script         |
| copylocal                                 | Low-level utilities to work with bytes, and bits and locally copy them effectively.                                                                                                   | Static Library | 
| cmdline                                   | C++ cross-platform implementation of useful command line parsing mechanisms.                                                                                                          | Static Library |
| fs                                        | Wrappers to work with string conversion routines, operate with files and filenames, and memory map in a system-independent way. Well optimizized.                                     | Static Library |
| linalg_linsolvers                         | Collection of dense and iterative and specialized linear solvers in CPU.                                                                                                              | Static Library |
| math_routines                             | Special math routines - matrix sparsification, convex optimization. However, there are also some data structures and algorithms: sorting, tries, heaps, and indexed heaps.            | Static Library |
| numerics                                  | Number differentiation helps to evaluate derivatives, gradients, and hessian numerically.                                                                                             | Static Library |
| timers                                    | Various timers - systems, good systems, from C++ runtime (something like CPUs clocks=>secs), low-level wrappers to calculate clocks for x86 and AArch64 .                             | Static Library |
| system/include/PlatformSpecificMacroses.h | Pretty detailed C/C++ include files with information about OS, Compiler versions, CPU Architecture                                                                                    | Include File   |
| system/include/digest                     | Digest to check bit-bit equivalence CRC-32-IEEE 802.3, MD5 RFC1321 Algorithms (system part)                                                                                           | Static Library | 
| system/include/network                    | TCP/IPv4, TCP/IPv5, UDP/IPv4, UPP/IPv6 wrappers  for Windows and POSIX API    (system part)                                                                                           | Static Library |
| system                                    | Memory pools, OS Memory Allocators, Low Level operating on Float/Double scalars                                                                                                       | Static Library |
| random                                    | calculate central statistics (mean, variance), uniform pseudo-random generators - Liner Cong. generator, Mersene, and several others, R.V. generators, shuffling with early stopping. | Static Linrary | 
| linalg_vectors                            | Dense vectors and light vector views with carefully designed API include vectors with two different underlying storages and custom implementation for CPU with SIMD.                  | Static Linrary | 
| linalg_matrices                           | Dense matrix implementation for BLAS operations and also Cholesky and QR factorization.                                                                                               | Static Linrary | 
| linalg_linsolvers                         | Several linear systems solvers: Jacobi, Gauss-Seidel, Conjugate-Gradient, Gauss-Elimination, 3-diagonal solver, backward and forward substitution (with SIMD optimizations).          | Static Linrary | 
| gpu_compute_support                       | The Project partially supports CUDA GPU Computation. It contains dense vector, dense matrix operation, GPU memory management, wrappers over low-level GPU code invocation             | Static Linrary |
| numerics                                  | Tools for numerical differentiation to validate the correctness of Hessian and gradient computations from function oracles.                                                           | Static Library |
| optimization_problems                     | Implements optimization problems, including logistic regression and quadratic minimization.                                                                                           | Static Library |

----


# Datasets

All datasets for experiments are distributed with this solution and located in `./dopt/datasets`.

# Auxiliray Binary Applications to Assist Follow-Up Work

To have the ability to build the following various tools the option `DOPT_INCLUDE_UTILS` should be turned on.

If your platform supports `OpenCL` please turn on the `DOPT_OPENCL_SUPPORT` option this includes GPUs for Apple Hardware, ARM Mali GPU, and Intel HD Graphics.

If your platform supports `CUDA` please turn on the `DOPT_CUDA_SUPPORT` option. This includes NVIDIA Hardware.

| **Auxiliray Binary**            | **Goal**                                                                                                                                                                             | **Type**   |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| bin_tests                       | Google Unit tests. Total number of tests is 102, but each test is pretty big. Tests cover all projects. (See separate help document to work with Google Tests)                       | Executable |
| utils/bin_host_view             | Binary application to check used compiler name, flags, version, linker flags, information about OS, DRAM, and CPU extensions.                                                        | Executable |
| utils/bin_opencl_view           | Observe platforms for OpenCL, devices available that support computation via OpenCL in the platform, its type, and available extensions if you will provide `show-extensions`        | Executable |
| utils/bin_cuda_view             | Observe NVIDIA CUDA Compatible devices, show peek computation and memory throughput limits, Available DRAM, and features. Run simple tests to verify. Flags: `verbose`, `benchmark`. | Executable |
| utils/bin_opt_problem_generator | Optional synthetics optimization problem generator. Can be used for debugging purposes.                                                                                              | Executable |
| utils/bin_split                 | Binary program to take one dataset in text, optionally reshuffle, add intercept, obtain information about several clients, and split it into several files.                          | Executable |    

## Extra about Optimization Problem Generator

**Remark.** To synthetically generate data you should specify via the command line the following arguments:

* `--dimension <INT>` -- dimension of of optimization problem
* `--samples <INT>` -- number of samples
* `--seed <INT>` -- seed for Pseudo-Random-Generator
* `--feature-generator <u_in_0_1> | <u_in_[-1_1]> | <u_in_[-10_10]>` -- encodes how features are filled. If you need specific distribution please write it by yourself.
* `--classes "-1,1"` -- next please specify labels for positive and negative class (any)
* `--classes_fractions "0.5,0.5"` -- Next please specify the fraction of samples of the first and second class
* `--out my.txt` -- the file into which the result will be saved

# Main Binary Applications

## Single-Node Multi-Core Runners

| **Project**                     | **Goal**                                                                                                                                                                                                                              | **Type**   |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| bin_fednl_local                 | Local simulation in the local machine FedNL, FedNL-LS x Compressors mentioned in the paper.                                                                                                                                           | Executable |
| bin_fednl_local_pp              | Local simulation in the local machine FedNL, FedNL-LS x Compressors mentioned in the paper for Partial Participation.  | Executable |

## Multiple-Node FL Training via TCP/IP Network

| **Project**               | **Goal**                                                                                      | **Type**   |
|---------------------------|-----------------------------------------------------------------------------------------------|------------|
| bin_fednl_distr_client    | Client application to participate in FedNL, and FedNL-LS algorithms with various compressors. | Executable |
| bin_fednl_distr_client_pp | Client application to participate in FedNL, and FedNL-LS algorithms with various compressors. | Executable |
| bin_fednl_distr_master    | Server application to control FedNL, FedNL-LS training.                                       | Executable |
| bin_fednl_distr_client_pp | Server application to control FedNL, FedNL-LS training.                                       | Executable |
