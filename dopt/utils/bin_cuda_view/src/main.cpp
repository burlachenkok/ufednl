#include <cuda_runtime_api.h>
#include <cuda.h>
#include <assert.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <array>

#include "dopt/system/include/CompilerInfo.h"
#include "dopt/cmdline/include/CmdLineParser.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "utils/bin_cuda_view/include/compiletime_help_functions.h"
#include "utils/bin_cuda_view/include/gpu_view_test_kernel.h"

#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"
#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"

//=========================================================================================================================
// LIST CUDA DEVICES
//=========================================================================================================================
template <class TStream>
bool listCudaDevices(TStream& out, bool showDebugInformationFromDevice)
{
    int device_count = dopt::GpuManagement::getNumberOfInstalledGPU();

    cudaDeviceProp deviceProp = {};

    out << "DETAILS FOR INSTALLED " << device_count << " GPUS" << "\n\n";
    
    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        // cudaSetDevice does not cause host synchronization
        cudaSetDevice(device_index);

        CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_index));

        out << "Device: " << device_index << "\"" << deviceProp.name << "\" " << (device_index == 0 ? "[DEFAULT]" : "") << "\n";
        
        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        out << "\n";      
        out << "  GPU GENERAL INFORMATION\n";

        const char* arch_names[] = { "" /*0*/,
                                    "TESLA"  /*1*/,
                                    "FERMI"  /*2*/,
                                    "KEPLER" /*3*/,
                                    ""       /*4*/,
                                    "MAXWELL"/*5*/,
                                    "PASCAL" /*6*/,
                                    "VOLTA"  /*7*/,    // >= 7.5 -- TURING.
                                    "AMPERE" /*8*/,    // >= 8.9 -- ADA_LOVELACE
                                    "HOOPER" /*9*/ };

        if (deviceProp.major >= sizeof(arch_names) / sizeof(arch_names[0]))
        {
            out << "   GPU Micro Architecture: " << "[UNKNOWN]" << "\n";
        }
        else
        {
            const char* archName = arch_names[deviceProp.major];
            
            if (deviceProp.major == 7 && deviceProp.minor >= 5)
                archName = "TURING";
            else if (deviceProp.major == 8 && deviceProp.minor >= 9)
                archName = "ADA-LOVELACE";
            
            out << "   GPU Architecture: " << archName << "\n";
        }
        out << "   CUDA Driver Version [latest CUDA version supported by the driver]: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << "\n";
        out << "   CUDA Runtime Version [runtime version for device]: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << "\n";
        out << "   CUDA Compute Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor << "\n";
        out << "\n";

        out << "  GPU MEMORY RESOURCES\n";
        out << "   Total amount of Global Memory on Device: " << (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GBytes" << "\n";
        out << "   Total amount of Constant Memory on Device: " << (double)deviceProp.totalConstMem/1024.0 << " KBytes" << "\n";
        out << "   Total amount of Shared Memory per Block: " << (double)deviceProp.sharedMemPerBlock/1024.0 << " KBytes" << "\n";
        out << "   Total number of Register Memory available per Block: " << (deviceProp.regsPerBlock * 4.0)/1024.0 << " KBytes" << "\n";
        if (deviceProp.l2CacheSize)
            out << "   L2 Cache Size: " << deviceProp.l2CacheSize/1024.0  << " KBytes" << "\n";
        out << "   Maximum memory pitch: " << deviceProp.memPitch << " Bytes" << "\n";

        out << "\n";
        out << "  COMPUTE/MEMORY EXECUTION UNITS PER GPU\n";
        out << "   GPU Scalar Cores Clock rate: " << deviceProp.clockRate * 1e-6f << " GHz" << "\n";
        out << "   Memory Clock rate: " << deviceProp.memoryClockRate * 1e-3f << " Mhz" << "\n";
        out << "   Memory Bus Width: " << deviceProp.memoryBusWidth << " Bits" << "\n";
        out << "   Number of Stream Multiprocessors on device: " << deviceProp.multiProcessorCount << "\n";
        out << "   Number of CUDA Scalar FP32 Cores on device: " << dopt::getSPcores(out, deviceProp) << "\n";
        out << "   Warp size: " << deviceProp.warpSize << "\n";

        int kWarpSchedulers = dopt::getWarpSchedulersPerStreamMultiprocessor(deviceProp);
        out << "   Warp Schedulers per Stream Multiprocessor: " << kWarpSchedulers << "\n";


        out << "\n";
        out << "  COMPUTE/MEMORY EXECUTION UNITS PER STREAM MULTIPROCESSOR\n";
        out << "   Warp size in threads: " << deviceProp.warpSize << "\n";
        out << "   Maximum number of resident threads / per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor <<  "\n";
        out << "   Maximum number of resident warps / per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize << "\n";
        out << "   Maximum number of threads / per block: " << deviceProp.maxThreadsPerBlock << "\n";
        out << "   Maximum sizes of each dimension of a block in threads: " << "[" << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1] << "," << deviceProp.maxThreadsDim[2] << "]" << "\n";
        out << "   Maximum sizes of each dimension of a grid in blocks: " << "[" << deviceProp.maxGridSize[0] << "," << deviceProp.maxGridSize[1] << "," << deviceProp.maxGridSize[2] << "]" << "\n";

        out << "\n";
        out << "  KERNEL LAUNCH CONFIGURATION RELATIVE\n";

        cudaSharedMemConfig bankCfg = cudaSharedMemBankSizeDefault;
        CHECK_CUDA_CALL(cudaDeviceGetSharedMemConfig(&bankCfg));
        
        const char* bankCfgStr = "";

        switch (bankCfg)
        {
        case cudaSharedMemBankSizeDefault:
            bankCfgStr = "cudaSharedMemBankSizeDefault";
            break;
        case cudaSharedMemBankSizeFourByte:
            bankCfgStr = "cudaSharedMemBankSize 4 Byte";
            break;
        case cudaSharedMemBankSizeEightByte:
            bankCfgStr = "cudaSharedMemBankSize 8 Byte";
            break;
        }

        out << "   Bank size for shared memory: " <<  bankCfgStr << "\n";
        size_t limitValue = 0;
        if (cudaDeviceGetLimit(&limitValue, cudaLimitStackSize) == cudaSuccess)
            out << "   Limit: Stack size for each GPU thread: " << limitValue/1024.0 << " KBytes" << "\n";

        if (cudaDeviceGetLimit(&limitValue, cudaLimitPrintfFifoSize) == cudaSuccess)
            out << "   Limit: Size of the shared FIFO used by the GPU printf(): " << limitValue/1024.0 << " KBytes" << "\n";

        if (cudaDeviceGetLimit(&limitValue, cudaLimitMallocHeapSize) == cudaSuccess)
            out << "   Limit: Size of the heap used by the GPU malloc() and free(): " << limitValue / 1024.0 << " KBytes" << "\n";

        out << "\n";
        out << "  THEORETICAL COMPUTE AND MEMORY LIMITS\n";

        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level        

        int kInstructionPerCycle = 0;

        if (deviceProp.major == 6)
            kInstructionPerCycle = 2;
        else
            kInstructionPerCycle = 1;       

        // https://stackoverflow.com/questions/15055877/how-to-get-memory-bandwidth-from-memory-clock-memory-speed
        const int kDDR3_PumpRate = 2;  // For HBM1/HBM2 memory and GDDR3 memory
        const int kDDR5_PumpRate = 4;  // For GDDR5 memory
        const int kDDR5X_PumpRate = 8; // For GDDR5X memory
        
        double peakPerformance_tflops = (double(deviceProp.clockRate /*in KHz*/) * dopt::getSPcores(out, deviceProp) * kInstructionPerCycle * kWarpSchedulers) / 1e+9;
        out << "   Estimated Peak Single Precision FP32: " << peakPerformance_tflops << " TFLOPS\n";

        double peakBandwidth_gb_sec = (double(deviceProp.memoryClockRate /*in KHz*/) * (deviceProp.memoryBusWidth / 8.0) * kDDR3_PumpRate) / 1e+6;
        out << "   Estimated Peak Memory Bandwidth: " << peakBandwidth_gb_sec << " GB/second\n";
        out << "*******************************************************************************************************************\n";

        if (showDebugInformationFromDevice)
        {
            out << "\n";
            out << "EXTRA DEBUG INFORMATION\n";
            benchmark::printCudaDebugInformation(device_index, deviceProp);
        }
    }
    out << "\n";
    out << "  PEER TO PEER INFORMATION COMPUTE AND MEMORY LIMITS\n";
    if (device_count <= 1)
    {
        out << "   SYSTEM HAS LESS THEN 2 GPUs. PEER TO PEER ACCESS HAS NO SENSE IN THIS SETTING\n";
    }
    
    // If GPUs are connected to the same PCIe root node. If peer-to-peer access can be enabled the you can copy data directly between two devices.
    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        CHECK_CUDA_CALL(cudaSetDevice(device_index));
        
        for (int target_device = 0; target_device < device_count; ++target_device)
        {
            if (target_device == device_index)
                continue;

            int peer_access_available = 0;
            cudaError_t reqPeer2PeerStatus = cudaDeviceCanAccessPeer(&peer_access_available, device_index, target_device);
            
            if (reqPeer2PeerStatus == cudaSuccess)
            {
                out << "    GPU " << device_index
                    << " can access GPU "
                    << target_device << " directly/peer-to-peer: "
                    << (peer_access_available ? "[YES]" : "[NO]") << "\n";
            }
        }
    }
    out << "\n";    
    out << "*******************************************************************************************************************\n";
    out << "Reference Information\n";
    out << " NVIDIA GPU ARCHITECTURES: Fermi 2.* => Kepler 3.* => Maxwell 5.* => Pascal 6.* => Volta/Turing 7.* => Ampeter/Ada 8.* => Hooper 9* \n";
    out << " NVIDIA GPU/Chipsets: " << "https://en.wikipedia.org/wiki/CUDA" << "\n";
    out << " Technical Specifications: " << "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications" << "\n";
    out << " Arithmetic Instructions: " << "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions" << "\n";
    out << "*******************************************************************************************************************\n";

    return true;
}

template <class TStream>
bool listCudaDevicesShort(TStream& out, bool showDebugInformationFromDevice)
{
    int device_count = dopt::GpuManagement::getNumberOfInstalledGPU();

    cudaDeviceProp deviceProp = {};
    out << " LIST OF INSTALLED CUDA COMPATIBLE " << device_count << " GPUS" << "\n\n";
    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        // Remark: cudaSetDevice does not cause host synchronization
        CHECK_CUDA_CALL(cudaSetDevice(device_index));
        CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_index));
       
        out << dopt::getDeviceString(deviceProp, device_index);

        if (showDebugInformationFromDevice)
        {
            out << "\n";
            out << "EXTRA DEBUG INFORMATION\n";
            benchmark::printCudaDebugInformation(device_index, deviceProp);
        }
    }

    return true;
}

//=========================================================================================================================
// MICRO BENCHMARKS
//=========================================================================================================================
std::string computeBenchmark()
{
    std::stringstream out;   
    const size_t N = 100 * 1024 * 1024; // 400 MBytes buffers
    const size_t buffersInBytes = N * sizeof(float); // Size in bytes for buffer

    // Allocate host buffer
    float* x = 0;
    x = (float*)malloc(buffersInBytes);
    assert(x != nullptr);
    for (size_t i = 0; i < N; ++i)
        x[i] = (float)i;

    // Allocate three device buffers
    float* d_x = 0;
    float* d_y = 0;
    float* d_z = 0;

    CHECK_CUDA_CALL(cudaMalloc((void**)&d_x, buffersInBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_y, buffersInBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_z, buffersInBytes));
    CHECK_CUDA_CALL(cudaMemcpy(d_x, x, buffersInBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_y, x, buffersInBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    dopt::HighPrecisionTimer tm;

    const size_t kRepeats = 200;

    out << '\n';
    
    {
        // Warmup
        benchmark::testSumKernel(d_x, d_y, d_z, kRepeats, N, 1);

        // Iterate through different blocks
        std::array<int, 4> blockSizes = { 128, 
                                          256, 
                                          512, 
                                          1024};
        
        for (auto blk : blockSizes)
        {
            tm.reset();
            benchmark::testSumKernel(d_x, d_y, d_z, kRepeats, N, blk);
            double eclapsedSecond = tm.getTimeSec();
            out << " Computational Performance [benchmark:vec add]. Block [" << blk << "]: " 
                << ( (kRepeats * double(N)) * (1.0/*add*/) ) / (eclapsedSecond * 1e+12)
                << " TFLOPS@FP32" << "\n";
        }
    }
    
    CHECK_CUDA_CALL(cudaMemcpy(x, d_z, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Check correctness
    {
        double maxError = 0.0f;
        for (int i = 0; i < N; i++)
        {
            maxError = fabs(x[i] - (i + i)) > maxError ? fabs(x[i] - (i + i)) : maxError;
        }
        out << " -Maximum error for [benchmark:vec add] " << maxError << ".  " << (maxError < 1e-6 ? "[PASSED]" : "[FAILED]") << "\n";
    }

    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    free(x);

    return out.str();
}

std::string copyBenchmark()
{  
    std::stringstream out;
    
    const size_t N = 100 * 1024 * 1024; // 400 MBytes buffers
    
    float* x = 0;
    
    float* d_x = 0;
    float* d_y = 0;

    const size_t buffersInBytes = N * sizeof(float);
    
    x = (float*)malloc(buffersInBytes);
    
    assert(x != nullptr);
    
    CHECK_CUDA_CALL(cudaMalloc((void**) &d_x, buffersInBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**) &d_y, buffersInBytes));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    dopt::HighPrecisionTimer tm;

    const size_t kRepeats = 10;

    {
        tm.reset();
        for (size_t i = 0; i < kRepeats; ++i)
        {
            CHECK_CUDA_CALL(cudaMemcpy(d_x, x, buffersInBytes, cudaMemcpyHostToDevice));          
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }
        double eclapsedSecond = tm.getTimeSec();
        out << " Host [Pageable] => Device [DRAM]   | Communication Speed: " << (buffersInBytes * kRepeats) / (eclapsedSecond) / (1024.0 * 1024.0 * 1024.0) << " GBytes/Sec" << "\n";
    }
    
    {
        tm.reset();
        for (size_t i = 0; i < kRepeats; ++i)
        {
            CHECK_CUDA_CALL(cudaMemcpy(x, d_x, buffersInBytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }
        double eclapsedSecond = tm.getTimeSec();
        out << " Device [DRAM]   => Host [PAGE]     | Communication Speed: " << (buffersInBytes * kRepeats) / (eclapsedSecond) / (1024.0 * 1024.0 * 1024.0) << " GBytes/Sec" << "\n";
    }

    {
        tm.reset();
        for (size_t i = 0; i < kRepeats; ++i)
        {
            CHECK_CUDA_CALL(cudaMemcpy(d_y, d_x, buffersInBytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }
        double eclapsedSecond = tm.getTimeSec();
        out << " Device [DRAM]   => Device [DRAM]   | Communication Speed: " << (buffersInBytes * kRepeats) / (eclapsedSecond) / (1024.0 * 1024.0 * 1024.0) << " GBytes/Sec" << "\n";
    }

    float* h_pinned = 0;
    CHECK_CUDA_CALL(cudaMallocHost((void**)&h_pinned, buffersInBytes));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    {
        tm.reset();
        for (size_t i = 0; i < kRepeats; ++i)
        {
            CHECK_CUDA_CALL(cudaMemcpy(d_x, h_pinned, buffersInBytes, cudaMemcpyHostToDevice));
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }
        double eclapsedSecond = tm.getTimeSec();
        out << " Host [NON-PAGE] => Device [DRAM]   | Communication Speed: " << (buffersInBytes * kRepeats) / (eclapsedSecond) / (1024.0 * 1024.0 * 1024.0) << " GBytes/Sec" << "\n";
    }

    {
        tm.reset();
        for (size_t i = 0; i < kRepeats; ++i)
        {
            CHECK_CUDA_CALL(cudaMemcpy(h_pinned, d_x, buffersInBytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }
        double eclapsedSecond = tm.getTimeSec();
        out << " Device [DRAM]   => Host [NON-PAGE] | Communication Speed: " << (buffersInBytes * kRepeats) / (eclapsedSecond) / (1024.0 * 1024.0 * 1024.0) << " GBytes/Sec" << "\n";
    }
    
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_CALL(cudaFreeHost(h_pinned));
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));   

    free(x);

    return out.str();
}

template <class TStream>
bool carryBenchmarks(TStream& out)
{
    out << "\n";
    out << "GPU INFORMATION BASED ON BENCHMARKS\n";

    int device_count = 0;
    CHECK_CUDA_CALL(cudaGetDeviceCount(&device_count));

    cudaDeviceProp deviceProp = {};

    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        // cudaSetDevice does not cause host synchronization
        cudaSetDevice(device_index);
        CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_index));
        out << dopt::getDeviceString(deviceProp, device_index);
        out << copyBenchmark();
        out << computeBenchmark();
    }

    return true;
}
//=========================================================================================================================

int main(int argc, char** argv)
{
    dopt::CmdLine cmdline(argc, argv);
    bool showDebugInformationFromDevice = cmdline.isFlagSetuped("gpu-from-inside") || true;
        
    preambleForViewer(std::cout);
    
    if (cmdline.isFlagSetuped("verbose"))
        listCudaDevices(std::cout, showDebugInformationFromDevice);
    else
        listCudaDevicesShort(std::cout, showDebugInformationFromDevice);

    if (cmdline.isFlagSetuped("benchmark"))
    {
        carryBenchmarks(std::cout);
    }

    // Destroy all allocations and reset all state on the current device in the current process.
    int device_count = dopt::GpuManagement::getNumberOfInstalledGPU();
    
    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        CHECK_CUDA_CALL(cudaSetDevice(device_index));
        CHECK_CUDA_CALL(cudaDeviceReset());
    }

    return 0;
}
//=========================================================================================================================
