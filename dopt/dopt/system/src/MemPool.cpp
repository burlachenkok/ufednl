#include "dopt/system/include/MemoryPool.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/system/include/SystemMemoryAllocate.h"

#include <stdlib.h>

namespace
{
    constexpr bool kUseCRTForAllocations = false;
    constexpr bool kUseSystemVirtualMemoryAllocations = true;
    constexpr bool kUseSystemPinnedMemoryAllocations = false; // Can be problematic due to constraints in OS

    constexpr size_t kSystemPageSize = 4 * 1024;
}

static_assert(int(kUseCRTForAllocations) + int(kUseSystemVirtualMemoryAllocations) + int(kUseSystemPinnedMemoryAllocations) == 1, 
              "Please Select Target Memory for memory allocations");

namespace dopt
{
    struct MemBlock
    {
        void* pool;               ///< pointer to the beginning of the block
        MemBlock* nextBlock;      ///< pointer to the next block
        size_t  itemSize;         ///< element size in bytes
        size_t maxBlockSize;      ///< block size

        void* firstAvailable;     ///< pointer to the first available element in the block
        void* lastAvailable;      ///< pointer to the last available element in the block
        void* freeElements;       ///< pointer to a special list of free elements

        size_t allocatedPages;    ///< number of allocated pages
        
        /** Ctor. Allocate raw memory block via crt.
        * @param theItemSize size of one item. Internal implementation require theItemSize to be great or equal to sizeof(void*) for performance point of view.
        * @param maxBlockSize block size in bytes
        */
        MemBlock(size_t theItemSize, size_t theMaxBlockSize)
        {
            if (theItemSize < sizeof(void*))
            {
                theItemSize = sizeof(void*);
            }

            maxBlockSize = theMaxBlockSize;

            itemSize = theItemSize;

            allocatedPages = (theMaxBlockSize + kSystemPageSize - 1) / (kSystemPageSize);

            if (kUseCRTForAllocations)
            {
                pool = malloc(maxBlockSize);
            }
            else if (kUseSystemVirtualMemoryAllocations)
            {
                pool = dopt::allocateVirtualMemory(kSystemPageSize, allocatedPages);
            }
            else if (kUseSystemPinnedMemoryAllocations)
            {
                pool = dopt::allocateVirtualMemory(kSystemPageSize, allocatedPages);
                bool res = dopt::lockVirtualMemory(pool, kSystemPageSize, allocatedPages);
                assert(res == true);
            }

            assert(pool != 0);
            
            firstAvailable = pool;
            lastAvailable = maxBlockSize - itemSize + (unsigned char*)pool;
            freeElements = nullptr;
            nextBlock = nullptr;
        }

        MemBlock(const MemBlock& rhs) = delete;

        /** Destructor. Give memory block back to CRT heap. No dtor's for real objects will be called!
        */
        ~MemBlock() 
        {
            if (kUseCRTForAllocations)
            {
                free(pool);
            }
            else if (kUseSystemVirtualMemoryAllocations)
            {
                dopt::deallocateVirtualMemory(pool, kSystemPageSize, allocatedPages);
            }
            else if (kUseSystemPinnedMemoryAllocations)
            {
                bool res = dopt::unlockVirtualMemory(pool, kSystemPageSize, allocatedPages);
                assert(res == true);
                dopt::deallocateVirtualMemory(pool, kSystemPageSize, allocatedPages);
            }
        }

        /** Mark memory block as whole free. No destructor's for real objects will be called!
        */
        void freeAll()
        {
            firstAvailable = pool;
            freeElements = NULL;
        }

        /** Check that for ptr address have already been free via freeItem(ptr) and so it is in free list
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
        * @return true if ptr point to allocated item, or potential allocated item in future
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
                if (firstAvailable <= lastAvailable)
                {
                    void* ptr = firstAvailable;
                    firstAvailable = itemSize + (unsigned char*)(firstAvailable);
                    return ptr;
                }
                else
                {
                    return nullptr;
                }
            }
            else
            {
                // for freed elements, the "element data" stores a pointer to the next element
                // even if it's a little hardcore
                void* res = freeElements;
                freeElements = *(void**)freeElements;
                return res;
            }
        }

        /** Free item from memory pool. Implementation - append free item to head of the list.
        * @param item remove item from memory pool
        */
        void freeItem(void* item)
        {
            *(void**)item = freeElements; // writing to the "head of the list" the value of the pointer to the free old element
            freeElements = item;          // offset head of the list
        }
    };


    MemoryPool::MemoryPool()
    {
        itemSize = 0;
        blockSize = 0;
        block = nullptr;
        itemsAllocated = 0;
    }

    MemoryPool::MemoryPool(size_t theItemSize, size_t theItemsCountInBlock)
    {
        if (theItemSize < sizeof(void*))
        {
            theItemSize = sizeof(void*);
        }

        itemSize = theItemSize;
        blockSize = theItemsCountInBlock * itemSize;
        block = new MemBlock(theItemSize, blockSize);
        itemsAllocated = 0;
    }

    MemoryPool::MemoryPool(MemoryPool&& rhs) noexcept
    : blockSize(rhs.blockSize)
    , itemSize(rhs.itemSize)
    , block(nullptr)
    , itemsAllocated(0)
    {
        block = rhs.block;
        itemsAllocated = rhs.itemsAllocated;

        rhs.block = nullptr;
        rhs.itemsAllocated = 0;
    }

    MemoryPool& MemoryPool::operator = (MemoryPool&& rhs) noexcept
    {
        freeAll();
        
        blockSize = rhs.blockSize;
        itemSize = rhs.itemSize;
        block = rhs.block;
        itemsAllocated = rhs.itemsAllocated;

        rhs.block = nullptr;
        rhs.itemsAllocated = 0;

        return *this;
    }

    MemoryPool::~MemoryPool()
    {
        MemBlock* next = nullptr;
        for (MemBlock* cur = block; cur != NULL; cur = next)
        {
            next = cur->nextBlock;
            delete cur;
        }
        block = nullptr;
    }

    void MemoryPool::freeAll()
    {
        MemBlock* next = NULL;
        for (MemBlock* cur = block; cur != NULL; cur = next)
        {
            next = cur->nextBlock;
            cur->freeAll();
        }
        itemsAllocated = 0;
    }

    void* MemoryPool::allocItem()
    {
        itemsAllocated++;
        void* item = block->allocItem();
        if (item)
            return item;

        MemBlock* cur = block->nextBlock;

        // if we do not allocate item in current memory block
        int blockCount = 1;

        for (; !item && cur; cur = cur->nextBlock)
        {
            item = cur->allocItem();
            blockCount++;
        }

        // allocate new block
        if (item == NULL)
        {
            cur = block;
            block = new MemBlock(itemSize, blockSize * (blockCount));
            block->nextBlock = cur;
            return block->allocItem();
        }

        return item;
    }

    bool MemoryPool::freeItem(void* item)
    {
        for (MemBlock* cur = block; cur != NULL; cur = cur->nextBlock)
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

    bool MemoryPool::isValidForFree(void* item) const
    {
        for (MemBlock* cur = block; cur != NULL; cur = cur->nextBlock)
        {
            if (cur->has(item))
                return true;
        }
        return false;
    }

    size_t MemoryPool::getItemSize() const
    {
        return itemSize;
    }

    size_t MemoryPool::getBlockSize() const
    {
        return blockSize;
    }

    size_t MemoryPool::getTotalAllocatedItems() const
    {
        return itemsAllocated;
    }

    size_t MemoryPool::getTotalBlocksCount() const
    {
        size_t count = 0;
        for (MemBlock* cur = block; cur != nullptr; cur = cur->nextBlock)
        {
            count++;
        }

        return count;
    }

    bool MemoryPool::isEmpty() const
    {
        return itemsAllocated == 0;
    }
}
