#if 0
#pragma once

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    using GPUPointer = void*;
    
    class GPUMemoryBuffer
    {
    public:
        GPUMemoryBuffer(size_t szInBytes)
        : m_pData(rawAllocate(szInBytes))
        , m_size(szInBytes)
        {
        }

        GPUMemoryBuffer(const GPUMemoryBuffer& rhs)
        {
            m_pData = rawAllocate(rhs.m_size);
            m_size = rhs.m_size;
            rawCopy(m_pData, rhs.m_pData, m_size);           
        }

        GPUMemoryBuffer(GPUMemoryBuffer&& rhs)
        {
            m_pData = rhs.m_pData;
            m_size = rhs.m_size;
            rhs.m_pData = nullptr;
            rhs.m_size = 0;
        }

        GPUMemoryBuffer& operator = (GPUMemoryBuffer&& rhs)
        {
            assert(this != &rhs);
            
            free();
            
            m_pData = rhs.m_pData;
            m_size = rhs.m_size;
            
            rhs.m_pData = nullptr;
            rhs.m_size = 0;
            
            return *this;
        }

        ~GPUMemoryBuffer() {
            free();
        }

        void reallocate(size_t szInBytes)
        {
            rawDeallocate(m_pData);
            m_pData = rawAllocate(szInBytes);
            assert(m_pData != nullptr);
            m_size = szInBytes;
        }

        void free()
        {
            rawDeallocate(m_pData);
            m_pData = nullptr;
            m_size = 0;
        }

        GPUPointer rawGpuPointer() const {
            return m_pData;
        }

        size_t sizeInBytes() const {
            size_t m_size;
        }

        void loadFromHostCPUMemory(void* pSrcHostData, size_t szInBytes)
        {
            rawCopyDevice2Device(GPUPointer dst, GPUPointer src, size_t szInBytes);
        }

        void storeToHostCPUMemory(void* pDstHostData, size_t szInBytes)
        {
        }
        

    protected:
        
        /** Allocates a linear range of device memory with the specified size in bytes.
        * @param szInBytes The size of the memory range to allocate in bytes.
        * @return pointer to memory range allocated on the device.
        */
        GPUPointer rawAllocate(size_t szInBytes);
        
        void rawDeallocate(GPUPointer ptr);
        
        /* The function used to transfer data between device and device
        * @param dst The destination pointer on the device.
        * @param src The source pointer on the device.
        * @param szInBytes The size of the memory range to copy in bytes.
        */
        void rawCopyDevice2Device(GPUPointer dst, GPUPointer src, size_t szInBytes);

    private:
        void* m_pData;
        size_t m_size;
    };
}
#endif