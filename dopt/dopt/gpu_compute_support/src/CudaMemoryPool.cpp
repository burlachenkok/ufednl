#if 0

#include "dopt/gpu_compute_support/include/CudaMemoryPool.h"

#include "lw_rawdata/headers_public/storage/MemoryPoolWithGpu.h"
#include "lw_rawdata/headers_public/storage/GpuManagement.h"

#include <malloc.h>
#include <memory.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <sstream>


#include <cuda_runtime_api.h> 

//const bool lw_rawdata::MemoryPoolWithGpu::kAllocatePinnedHostMemory;

namespace
{
    struct MemBlockWithGPU;

    struct MemBlockWithGPU
    {

    private:
        MemBlockWithGPU(const MemBlockWithGPU& rhs);
        MemBlockWithGPU& operator = (const MemBlockWithGPU& rhs);
    public:
        void* allocateBytes(size_t sz) {
            void* memPtr = 0;

            // Allocate pinned host memory
            if (isAllocatePinnedHostMemory)
            {
                cudaError_t allocStatus = cudaMallocHost(&memPtr, sz);         // Allocated pinned host memory
                assert(allocStatus == cudaSuccess);
            }
            else
            {
                memPtr = malloc(sz);
            }

            // Force setup ll bytes to zero (just for safe)
            memset(memPtr, 0, sz);
            return memPtr;
        }

        void* allocateBytesGpu(size_t sz) {
            void* dePtr = 0;
            cudaError_t allocStatus = cudaMalloc(&dePtr, sz);             // Allocate GPU global memory
            assert(allocStatus == cudaSuccess);
            //cudaError_t memsetStatus = cudaMemset(dePtr, 0, sz);
            //assert(memsetStatus == cudaSuccess);
            return dePtr;
        }

        void deallocateBytes(void* ptr) {
            if (isAllocatePinnedHostMemory)
            {
                cudaError_t deallocStatus = cudaFreeHost(ptr);             // Deallocate pinned host memory
                assert(deallocStatus == cudaSuccess);
            }
            else
            {
                free(ptr);
            }
        }

        static void deallocateBytesGpu(void* devPtr) {
            cudaError_t deallocStatus = cudaFree(devPtr);                 // Deallocate memory in GPU
            assert(deallocStatus == cudaSuccess);
        }

        bool              gpuMemoryHaveBeenModified;    ///< gpu memory was in dirty state
        bool              cpuMemoryHaveBeenModified;    ///< cpu memory was in dirty state

        uint8_t* gpuPool;   ///< pointer to the start of memory block in the GPU device memory
        uint8_t* pool;      ///< pointer to the start of memory block
        MemBlockWithGPU* nextBlock;  ///< pointer to next block
        size_t       itemSize;     ///< size of item to allocate in bytes
        size_t  blockSizeInBytes;  ///< block size in bytes. I.e. used memory for memory block.

        void* firstAvailable;     ///< pointer to first available items in block
        void* lastAvailable;     ///< pointer to last available items in block

        void* freeElements;  ///< pointer to head of special list of free items. Free list contains in data contains: null pointer (end of list) or pointer to next item in list

        bool isAllocatePinnedHostMemory;        ///< Flag which describe that under the hood host memory is pinned

        bool isPerformAsyncCopying;             ///< Flag which describe that during syncing in fact all will be performing in asyncornous fasion

        /** Ctor. Allocate raw memory block via crt.
        * @param theItemSize size of one item. Internal implementation require theItemSize to be great or equal to sizeof(void*) for performace point of view.
        * @param maxBlockSize block size in bytes
        */
        MemBlockWithGPU(bool isAllocatePinnedHostMemory, bool isPerformAsyncCopying, size_t theItemSize, size_t maxBlockSize)
        {
            this->isAllocatePinnedHostMemory = isAllocatePinnedHostMemory;
            this->isPerformAsyncCopying = isPerformAsyncCopying;

            if (theItemSize < sizeof(void*))
            {
                assert(!"Items of memory pool should be at least big as sizeof(void*)");
                theItemSize = sizeof(void*);
            }

            itemSize = theItemSize;
            blockSizeInBytes = maxBlockSize;

            pool = (uint8_t*)allocateBytes(maxBlockSize);
            gpuPool = (uint8_t*)allocateBytesGpu(maxBlockSize);

            firstAvailable = pool;
            lastAvailable = maxBlockSize - itemSize + pool;
            freeElements = NULL;
            nextBlock = NULL;

            gpuMemoryHaveBeenModified = false;
            cpuMemoryHaveBeenModified = false;
        }

        /** Dtor. Give memory block back to CRT heap. No dtor's for real objects will be called!
        */
        ~MemBlockWithGPU()
        {
            deallocateBytes(pool);
            deallocateBytesGpu(gpuPool);
        }

        /** Mark memory block as whole free. No dtor's for real objects will be called!
        */
        void freeAll()
        {
            firstAvailable = pool;
            freeElements = NULL;
        }

        /** Check that for ptr address have already been freed via freeItem(ptr)
        * @see freeItem()
        * @param ptr pointer to some memory
        * @return true if ptr item have been freed via freeItem()
        */
        bool haveBeenFree(void* ptr) const
        {
            void* next = freeElements;
            while (next)
            {
                if (next == ptr)
                    return true;
                else
                    next = *(void**)next;
            }
            return false;
        }

        /** Check that ptr belong to memory blocks, and is not free
        * @see freeItem()
        * @see allocItem()
        * @param ptr pointer to some memory
        * @return true if ptr point to allocated item, or potenional allocated item in future
        */
        bool has(void* ptr) const
        {
            if (ptr < pool || ptr > lastAvailable)
                return false;
            return haveBeenFree(ptr) == false;
        }

        /** Allocate memory for one item. If there is some element in free list take first.
        * @return pointer to allocated item
        */
        void* allocItem()
        {
            if (freeElements == nullptr)
            {
                // there are no free items
                if (firstAvailable <= lastAvailable)
                {
                    void* ptr = firstAvailable;
                    firstAvailable = itemSize + (uint8_t*)(firstAvailable);
                    return ptr;
                }
                else
                {
                    return nullptr;
                }
            }
            else
            {
                // there are no some free items, let's reuse it
                // A bit hardcore is here
                void* res = freeElements;             // take one free item from the head of the list
                freeElements = *(void**)freeElements; // move head of the list
                return res;
            }
        }

        /** Free item from memory pool.
        * @param item remove item from memory pool
        */
        void freeItem(void* item)
        {
            // Implementation - append free item to head of the list.
            // 0. Item will be new head
            *(void**)item = freeElements; // 1. Write into plaaned head of the list pointer to previous head of the list.
            freeElements = item;           // 2. update head of the list to new one
        }
    };
}

namespace lw_rawdata
{
    MemoryPoolWithGpu::MemoryPoolWithGpu(InitFlags flags, size_t theItemSize, size_t theItemsCountInBlock)
    {
        if (flags == InitFlags::eSyncronousMode)
        {
            isAllocatePinnedHostMemory = false;
            isPerformAsyncCopying = false;
        }
        else
        {
            isAllocatePinnedHostMemory = true;
            isPerformAsyncCopying = true;
        }

        if (theItemSize < sizeof(void*))
        {
            theItemSize = sizeof(void*);
        }

        itemSize = theItemSize;
        blockSize = theItemsCountInBlock * itemSize;
        firstMemBlock = new MemBlockWithGPU(isAllocatePinnedHostMemory, isPerformAsyncCopying, theItemSize, blockSize);
        itemsAllocated = 0;
    }

    MemoryPoolWithGpu::~MemoryPoolWithGpu()
    {
        MemBlockWithGPU* next = nullptr;
        for (MemBlockWithGPU* cur = (MemBlockWithGPU*)firstMemBlock; cur != nullptr; cur = next)
        {
            next = cur->nextBlock;
            delete cur;
        }
        firstMemBlock = nullptr;
    }

    void MemoryPoolWithGpu::freeAll()
    {
        MemBlockWithGPU* next = nullptr;
        for (MemBlockWithGPU* cur = (MemBlockWithGPU*)firstMemBlock; cur != nullptr; cur = next)
        {
            next = cur->nextBlock;
            cur->freeAll();
        }
        itemsAllocated = 0;
    }

    void* MemoryPoolWithGpu::allocItem()
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;

        itemsAllocated++;
        void* item = block->allocItem();
        if (item)
            return item; // successfully allocated within current block


        // if we do not allocate item in current memory block, try allocate in next block
        MemBlockWithGPU* cur = block->nextBlock;

        // block ount trversed so far
        int blockCount = 1;

        for (; !item && cur; cur = cur->nextBlock, ++blockCount)
        {
            item = cur->allocItem();
        }

        // allocate new block and make it with size blockSize * NUMBER OF BLOCKS TRAVERSED SO FAR
        if (item == nullptr)
        {
            cur = block;                                                // setup cur for firstMemBlockWithGPU
            block = new MemBlockWithGPU(isAllocatePinnedHostMemory, isPerformAsyncCopying, itemSize, blockSize * (blockCount));   // create new block
            block->nextBlock = cur;                                     // setup pointer to next block to previously "firstMemBlockWithGPU"
            firstMemBlock = block;                                      // update first memory block
            item = block->allocItem();                                  // finally allocate item
        }

        return item;
    }

    bool MemoryPoolWithGpu::notifyStateChangedInCPU(void* cpuAddress)
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;

        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            if (cur->has(cpuAddress))
            {
                cur->cpuMemoryHaveBeenModified = true;
                return true;
            }
        }
        return false;
    }

    bool MemoryPoolWithGpu::notifyStateChangedInGPU(void* gpuAddress)
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;

        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            size_t byteOffset = ((uint8_t*)gpuAddress - cur->gpuPool);
            if (cur->has(cur->pool + byteOffset))
            {
                cur->gpuMemoryHaveBeenModified = true;
                return true;
            }
        }
        return false;
    }


    bool MemoryPoolWithGpu::sync()
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            if (cur->gpuMemoryHaveBeenModified && cur->cpuMemoryHaveBeenModified)
            {
                assert(!"INCONSISTEN STATE. CAN NOT DECIDE IN WHICH DIRECTION MEMORY SHOULD BE COPIED");
                return false;
            }
            else if (cur->gpuMemoryHaveBeenModified)
            {
                cudaError_t cudaMemcpyStatus = cudaSuccess;
                if (isPerformAsyncCopying)
                {
                    cudaMemcpyStatus = cudaMemcpyAsync(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyDeviceToHost, GpuManagement::currentGpu().getCurrentStream());
                }
                else
                {
                    cudaMemcpyStatus = cudaMemcpy(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyDeviceToHost);
                }

                assert(cudaMemcpyStatus == cudaSuccess);
                cur->gpuMemoryHaveBeenModified = false;
            }
            else if (cur->cpuMemoryHaveBeenModified)
            {
                cudaError_t cudaMemcpyStatus = cudaSuccess;
                if (isPerformAsyncCopying)
                {
                    cudaMemcpyStatus = cudaMemcpyAsync(cur->gpuPool, cur->pool, cur->blockSizeInBytes, cudaMemcpyHostToDevice, GpuManagement::currentGpu().getCurrentStream());
                }
                else
                {
                    cudaMemcpyStatus = cudaMemcpy(cur->gpuPool, cur->pool, cur->blockSizeInBytes, cudaMemcpyHostToDevice);
                }

                assert(cudaMemcpyStatus == cudaSuccess);
                cur->cpuMemoryHaveBeenModified = false;
            }
            else
            {
                ;
            }
        }
        return true;
    }

    void MemoryPoolWithGpu::syncForceCopy2GPU()
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            cudaError_t cudaMemcpyStatus = cudaSuccess;

            if (isPerformAsyncCopying)
            {
                cudaMemcpyStatus = cudaMemcpyAsync(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyHostToDevice, GpuManagement::currentGpu().getCurrentStream());
            }
            else
            {
                cudaMemcpyStatus = cudaMemcpy(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyHostToDevice);
            }

            assert(cudaMemcpyStatus == cudaSuccess);
            cur->gpuMemoryHaveBeenModified = false;
            cur->cpuMemoryHaveBeenModified = false;
        }
    }

    void MemoryPoolWithGpu::syncForceCopy2CPU()
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            cudaError_t cudaMemcpyStatus = cudaSuccess;

            if (isPerformAsyncCopying)
            {
                cudaMemcpyStatus = cudaMemcpyAsync(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyDeviceToHost, GpuManagement::currentGpu().getCurrentStream());
            }
            else
            {
                cudaMemcpyStatus = cudaMemcpy(cur->pool, cur->gpuPool, cur->blockSizeInBytes, cudaMemcpyDeviceToHost);
            }

            assert(cudaMemcpyStatus == cudaSuccess);
            cur->gpuMemoryHaveBeenModified = false;
            cur->cpuMemoryHaveBeenModified = false;
        }
    }

    uint8_t* MemoryPoolWithGpu::convertCputPointerToGpuPointer(void* cpuAddress)
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            if (cur->has(cpuAddress))
            {
                size_t byteOffset = ((uint8_t*)cpuAddress - cur->pool);
                return cur->gpuPool + byteOffset;
            }
        }
        return nullptr;
    }

    uint8_t* MemoryPoolWithGpu::converGputPointerToCpuPointer(void* gpuAddress)
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;

        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            size_t byteOffset = ((uint8_t*)gpuAddress - cur->gpuPool);
            if (cur->has(cur->pool + byteOffset))
            {
                return cur->pool + byteOffset;
            }
        }
        return nullptr;
    }

    bool MemoryPoolWithGpu::freeItem(void* item)
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;

        for (MemBlockWithGPU* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            if (cur->has(item))
            {
                cur->freeItem(item);
                itemsAllocated--;
                return true;
            }
        }
        return false;
    }

    bool MemoryPoolWithGpu::isValidForFree(void* item) const
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        for (MemBlockWithGPU* cur = block; cur != NULL; cur = cur->nextBlock)
        {
            if (cur->has(item))
                return true;
        }
        return false;
    }

    size_t MemoryPoolWithGpu::getItemSizeInBytes() const
    {
        return itemSize;
    }

    size_t MemoryPoolWithGpu::getBlockSizeInBytes() const
    {
        return blockSize;
    }

    size_t MemoryPoolWithGpu::getTotalAllocatedItems() const
    {
        return itemsAllocated;
    }

    size_t MemoryPoolWithGpu::getTotalBlocksCount() const
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        size_t count = 0;
        for (MemBlockWithGPU* cur = block; cur != NULL; cur = cur->nextBlock)
        {
            count++;
        }

        return count;
    }

    size_t MemoryPoolWithGpu::getUsedMemorByMemBlocks() const
    {
        MemBlockWithGPU* block = (MemBlockWithGPU*)firstMemBlock;
        size_t number_of_bytes = 0;
        for (MemBlockWithGPU* cur = block; cur != NULL; cur = cur->nextBlock)
        {
            number_of_bytes += cur->blockSizeInBytes;
            // number_of_bytes += sizeof(MemBlockWithGPU); // it doesn't include this quantity
        }

        return number_of_bytes;
    }

    bool MemoryPoolWithGpu::isEmpty() const
    {
        return itemsAllocated == 0;
    }

    std::string MemoryPoolWithGpu::deviceInfo()
    {
        std::stringstream str;

        cudaDeviceProp deviceProp;
        cudaError_t status;
        int device_count = 0;
        status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess) {
            str << "cudaGetDeviceCount() failed: " << cudaGetErrorString(status);
            return str.str();
        }
        str << "CUDA-capable devices: " << device_count << " / ";
        int device_index = 0;
        cudaGetDevice(&device_index);
        str << "CUDA device current: " << device_index << " / ";
        status = cudaGetDeviceProperties(&deviceProp, device_index);
        if (status != cudaSuccess)
        {
            str << "cudaGetDeviceProperties() for device failed: " << cudaGetErrorString(status);
            return str.str();
        }
        str << deviceProp.name << "/";
        const char* arch_names[] = { "" /*0*/,  ""  /*1*/,  "FERMI"  /*2*/,  "KEPLER" /*3*/,  ""       /*4*/, "MAXWELL"/*5*/,  "PASCAL" /*6*/, "VOLTA"  /*7*/ };

        if (deviceProp.major < sizeof(arch_names) / sizeof(arch_names[0]))
            str << arch_names[deviceProp.major];
        return str.str();
    }
}
#endif
