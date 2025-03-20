#pragma once

#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"
#include <assert.h>

#include <string>
#include <utility>
#include <iostream>

namespace dopt
{
    struct GPUDevice
    {           
        int deviceNumber = 0;
    };
   
    class GpuManagement
    {
    public:

        /** Get number of CUDA Compatible devices in the system
        * @return number of devices
        */
        static int getNumberOfInstalledGPU();

        /** Get default GPU device
        * @return default device
        */
        static constexpr GPUDevice defaultGPUDevice()
        {
            GPUDevice dev = { 0 };
            return dev;            
        }

        /** Get short device description
        * @return custom string which hold important information about GPU
        */
        static std::string getShortDeviceDescription(GPUDevice device);

        /** Select this gpu as current one for CUDA runtime
        * @param device - device to select
        * @remark Once a current device is selected, all CUDA operations will be applied to that device, such as
        * - Any device memory allocated from the host thread
        * - Any host memory allocated with CUDA runtime functions
        * - Any streams or events created from the host thread will be associated with that device
        * - Any kernels launched from the host thread will be executed on that device
        */
        static void selectGpu(GPUDevice device);

        /** Check that current GPU selected in CUDA runtime is this one
        * @param device - device to check
        * @return true if current device is the device
        */
        static bool isGpuSelected(GPUDevice device);

        /** Get current GPU device
        * @return instance of current device
        */
        static GPUDevice currentDevice();
        
        /** Wait for completion of all operations in current device. Explicit synchronization between host and device.
        * @note Blocks until the device has completed all preceding requested tasks.
        */
        void syncronizeDevice();

        /** Wait for completion of all operations in current stream
        * @sa getCurrentStream
        */
        void syncronizeStream();

        /** Get handle for stream associated with this device
        * @return stream handle for current device
        */
        cudaStream_t getCurrentStream() const;
        
        /** Set device associated with this device manager as current
        */
        void selectThisGpu();

        /** Set handle for stream associated with this device
        * @param newStreamForOperations - stream handle for current device
        * @return *this
        */
        GpuManagement& setCurrentStream(cudaStream_t newStreamForOperations);

        /** Reset stream stream associated with this device with default stream
        * @return *this
        */
        GpuManagement& setCurrentStreamToDefault();

        /** Get the item from GPU syncronously
        * @param devPtr - pointer to the item in GPU memory
        * @return copy of *devPtr object in host/cpu memory
        * @remark Implicit synchronization at the host side is performed
        */
        template<class T>
        T getItemInGpu(const T* devPtr)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);

            T res = T();

            CHECK_CUDA_CALL(cudaMemcpy(&res, devPtr, sizeof(res), cudaMemcpyDeviceToHost));
            
            selectGpu(prevDev);

#if 0
            // Returns the last error that has been produced by any of the runtime calls in the same host thread. 
            // Note that this call does not reset the error to cudaSuccess like cudaGetLastError().
            CHECK_CUDA_CALL(cudaPeekAtLastError());

            // Blocks until the device has completed all preceding requested tasks. 
            // cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed.
            // If the cudaDeviceScheduleBlockingSync flag was set for this device, the host thread will block until the device has finished its work.
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
#endif
            return res;
        }

        /** Set the item in GPU syncronously
        * @param devPtr - pointer to the item in GPU memory
        * @param item - item to set in GPU memory which will be copied to devPtr. The item is store in CPU memory.
        * @remark Implicit synchronization at the host side is performed.
        */
        template<class T>
        void setItemInGpu(T* devPtr, const T& item)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);
            T res = T();
            CHECK_CUDA_CALL(cudaMemcpy(devPtr, &item, sizeof(res), cudaMemcpyHostToDevice));
            selectGpu(prevDev);
        }

        template<class T>
        void copyHost2DeviceSync(T* devOutput, const T* hostInput, size_t items)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);

            size_t numberOfBytesToCopy = items * sizeof(T);
            CHECK_CUDA_CALL(cudaMemcpy(devOutput, hostInput, numberOfBytesToCopy, cudaMemcpyHostToDevice));
            // CHECK_CUDA_CALL(cudaPeekAtLastError());
            // CHECK_CUDA_CALL(cudaDeviceSynchronize());
            selectGpu(prevDev);
        }

        template<class T>
        void copyDevice2HostSync(T* hostOutput, const T* devInput, size_t items)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);
            size_t numberOfBytesToCopy = items * sizeof(T);
            CHECK_CUDA_CALL(cudaMemcpy(hostOutput, devInput, numberOfBytesToCopy, cudaMemcpyDeviceToHost));
            selectGpu(prevDev);
        }

        template<class T>
        void copyDevice2DeviceAsync(T* devOutput, const T* devInput, size_t items)
        {
            // TODO: If memory from different devices
            assert(devOutput != devInput);
            size_t numberOfBytesToCopy = items * sizeof(T);
            CHECK_CUDA_CALL(cudaMemcpyAsync(devOutput, devInput, numberOfBytesToCopy, cudaMemcpyDeviceToDevice, currentStream));
            // CHECK_CUDA_CALL(cudaPeekAtLastError());
            // CHECK_CUDA_CALL(cudaDeviceSynchronize());
        }

        template<class T>
        void copyDevice2DeviceSync(T* devOutput, const T* devInput, size_t items)
        {
            // TODO: If memory from different devices
            assert(devOutput != devInput);
            size_t numberOfBytesToCopy = items * sizeof(T);
            CHECK_CUDA_CALL(cudaMemcpy(devOutput, devInput, numberOfBytesToCopy, cudaMemcpyDeviceToDevice));
        }

        template<class T>
        void setDeviceMemoryToZero(T* devOutput, size_t items)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);
            size_t numberOfBytesToSet = items * sizeof(T);
            CHECK_CUDA_CALL(cudaMemset(devOutput, 0, numberOfBytesToSet));
            selectGpu(prevDev);
        }
        
        /* Allocate pinned/non-pagable host memory
        * @param szInBytes number of bytes to allocate in the host memory
        * @remark Allocated pinned host memory can be read and written with much higher bandwidth than pageable memory.
        * @reamrk Allocating excessive amounts of pinned memory might degrade host system performance.
        */
        void* allocatePinnedBytesInHost(size_t szInBytes)
        {
            void* cpuMemPtr = 0;
            CHECK_CUDA_CALL(cudaMallocHost(&cpuMemPtr, szInBytes));
            assert(cpuMemPtr != 0);
            return cpuMemPtr;
        }        

        /** Free previously allocated bytes in host(CPU) DRAM pinned memory pool
        * @param rawCPUPointer poiter into the memory allocated in the host
        * @sa allocatePinnedBytesInHost
        */
        bool freePinnedBytesInHost(void* rawCPUPointer/*, size_t szInBytes*/)
        {
            CHECK_CUDA_CALL(cudaFreeHost(rawCPUPointer));
            return true;
        }      

        /** Allocate szInBytes bytes in GPU memory
        * @param szInBytes number of requested bytes
        * @return device pointer in the device
        * @sa freeMemoryInDevice
        */
        void* allocateBytesInDevice(size_t szInBytes)
        {
            GPUDevice prevDev = currentDevice();           
            selectGpu(device);
            
            // Allocate GPU global memory. 
            // This function allocates a linear range of device memory with the specified size in bytes.
            void* devPtr = 0;
            CHECK_CUDA_CALL(cudaMalloc(&devPtr, szInBytes));
            assert(devPtr != 0);            

            selectGpu(prevDev);
            return devPtr;
        }

        /** Free prevouisly allocated bytes in the device(GPU) DRAM memory
        * @param rawDevPointer poiter into the memory allocated in the device
        * @sa allocateBytesInDevice
        */
        bool freeMemoryInDevice(void* rawDevPointer/*, size_t szInBytes*/)
        {
            GPUDevice prevDev = currentDevice();
            selectGpu(device);

            // Deallocate memory in GPU
            CHECK_CUDA_CALL(cudaFree(rawDevPointer));

            selectGpu(prevDev);
            return true;
        }
        
        /* Get the Warp size.
        * @return warp size
        * @remark In NVIDIA GPUs it is typically (almost for sure) 32
        * @remark In AMD GPUs it is typically (almost for sure) 64
        * @remark Warp is a bundle of threads that are physically and really executed in parallel. 
        * Each thread within a warp has its own ALU, but they all share a common Control Unit.
        */
        int warpSize() const {
            return warpSize_;
        }

        /* Get the maximum number of threads per block
        * @return maximum number of threads per block
        * @remark In NVIDIA GPUs it is typically 1024
        */
        int maxThreadsPerBlock() const  {
            return maxThreadsPerBlock_;
        }

        int maxResidentThreadsPerSm() const {
            return maxResidentThreadsPerSm_;
        }

        int smNumber() const {
            return smNumber_;
        }

        int sharedMemoryPerBlock() const {
            return sharedMemoryPerBlock_;
        }
        
        bool hasLittleEndianAdressing() const {
            return hasLittleEndian_;
        }

        GpuManagement(GPUDevice theDevice = GPUDevice());

        ~GpuManagement();

        void notificationKernelIsLaunching(const char* kernelName) const;
        void notificationKernelWasLaunched(const char* kernelName) const;

    private:
        static thread_local GPUDevice currentUsedDevice;  ///< Current selected device
        
        GPUDevice device;             ///< Device
        cudaStream_t currentStream;   ///< Stream for operations on GPU associated with this device

        int warpSize_;                ///< The warp size [typically 32 in NVIDIA GPUs, and 64 in AMD GPUs]
        int maxThreadsPerBlock_;      ///< The maximum number of threads per block [typically 1024]
        int maxResidentThreadsPerSm_; ///< The maximum number of threads per SM
        int sharedMemoryPerBlock_;    ///< The shared memory available per block in bytes
        int smNumber_;                ///< Number of multiprocessors on device. See also: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

        bool hasLittleEndian_ = true; ///< TODO: Check and fill-in more carefully        
    };
}
