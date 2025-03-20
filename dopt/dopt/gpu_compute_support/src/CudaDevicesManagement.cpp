#include "dopt/gpu_compute_support/include/CudaDevicesManagement.h"
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

namespace dopt
{
    thread_local GPUDevice GpuManagement::currentUsedDevice = GpuManagement::defaultGPUDevice();
    
    std::string GpuManagement::getShortDeviceDescription(GPUDevice device)
    {
        cudaDeviceProp deviceProp = {};
        CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device.deviceNumber));
        std::string res = dopt::getDeviceString(deviceProp, device.deviceNumber);
        return res;
    }
    
    int GpuManagement::getNumberOfInstalledGPU()
    {
        int device_count = 0;        
        cudaError_t status = cudaGetDeviceCount(&device_count);
        assert(status == cudaSuccess);
        return device_count;
    }

    void GpuManagement::selectGpu(GPUDevice device)
    {
        if (currentUsedDevice.deviceNumber != device.deviceNumber)
        {
            cudaError_t status = cudaSetDevice(device.deviceNumber);
            assert(status == cudaSuccess);
        }
    }

    bool GpuManagement::isGpuSelected(GPUDevice device)
    {
        return currentUsedDevice.deviceNumber == device.deviceNumber;
#if 0
        int device_index_in_cuda = 0;
        CHECK_CUDA_CALL(cudaGetDevice(&device_index_in_cuda));
        return device_index_in_cuda == device.deviceNumber;
#endif
    }

    GPUDevice GpuManagement::currentDevice()
    {
        return currentUsedDevice;
#if 0
        int device_index_in_cuda = 0;
        CHECK_CUDA_CALL(cudaGetDevice(&device_index_in_cuda));        
        GPUDevice dev = { device_index_in_cuda };       
        return dev;
#endif
    }

    cudaStream_t GpuManagement::getCurrentStream() const
    {
        return currentStream;
    }

    void GpuManagement::selectThisGpu()
    {
        selectGpu(device);
    }

    GpuManagement& GpuManagement::setCurrentStream(cudaStream_t newStreamForOperations)
    {
        currentStream = newStreamForOperations;
        return *this;
    }

    GpuManagement& GpuManagement::setCurrentStreamToDefault()
    {
        currentStream = (cudaStream_t)0;
        return *this;
    }

    void GpuManagement::syncronizeStream()
    {
        CHECK_CUDA_CALL(cudaStreamSynchronize(currentStream));
    }

    void GpuManagement::syncronizeDevice()
    {
        GPUDevice prevDev = currentDevice();
        selectGpu(device);
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        selectGpu(prevDev);
    }

    GpuManagement::GpuManagement(GPUDevice theDevice)
    : currentStream((cudaStream_t)0)
    , device(theDevice)
    , maxThreadsPerBlock_(0)
    , maxResidentThreadsPerSm_(0)
    , sharedMemoryPerBlock_(0)
    , smNumber_(0)
    {
        GPUDevice prevDev = currentDevice();
        selectGpu(theDevice);

        // https://stackoverflow.com/questions/31458016/in-cuda-is-it-guaranteed-that-the-default-stream-equals-nullptr
        cudaDeviceProp deviceProp;
        CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device.deviceNumber));
        
        warpSize_ = deviceProp.warpSize;
        maxThreadsPerBlock_ = deviceProp.maxThreadsPerBlock;
        maxResidentThreadsPerSm_ = deviceProp.maxThreadsPerMultiProcessor;
        sharedMemoryPerBlock_ = deviceProp.sharedMemPerBlock;
        smNumber_ = deviceProp.multiProcessorCount;
        
        selectGpu(prevDev);
    }
    
    GpuManagement::~GpuManagement()
    {
#if 0
        GPUDevice prevDev = currentDevice();
        selectGpu(device);
        CHECK_CUDA_CALL(cudaDeviceReset());
        selectGpu(prevDev);
#endif
    }

    void GpuManagement::notificationKernelIsLaunching(const char* kernelName) const
    {
    }

    void GpuManagement::notificationKernelWasLaunched(const char* kernelName) const
    {
        CHECK_CUDA_DEBUG_SYNC(cudaGetLastError());
        CHECK_CUDA_DEBUG_SYNC(cudaDeviceSynchronize());
    }
}
