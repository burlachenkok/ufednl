/** @file
 * C++ cross-platform raw memory pool
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

namespace dopt
{
    struct MemBlock;

    /**
    * @brief Memory pool. For fast allocate/deallocate object with same size
    * @details theItemSize should at list as big as (void*) in bytes. If it's not the case internally always single elements with size as least sizeof(void*) will be allocated.
    */
    class MemoryPool
    {
    public:
        /** default ctor
        */
        MemoryPool();

        /** ctor
        * @param theItemSize at least equal to size of void*, or it will expand to it
        * @param theItemsCountInBlock size of block in elements
        */
        MemoryPool(size_t theItemSize, size_t theItemsCountInBlock);

        /** Copy move constructor
        * @param rhs xvalue expression from which we perform move
        */
        MemoryPool(MemoryPool&& rhs) noexcept;

        /** Assigment move operator
        * @param rhs xvalue expression from which we perform move
        * @return reference to *this
        */
        MemoryPool& operator = (MemoryPool&& rhs) noexcept;

        /** Destructor. Clean memory for under the hood memory blocks.
        */
        ~MemoryPool();

        /** Free all items from memory pool.
        */
        void freeAll();

        /** Allocate one item from memory pool
        * @see freeItem()
        * @return point to allocated item
        */
        void* allocItem();

        /** Free item from memory pool
        * @see allocItem()
        * @param item pointer to item which you're going to free
        * @return true point to allocated item
        */
        bool freeItem(void* item);

        /** Check that memory for ptr have been allocated and now can be free
        * @see freeItem()
        * @param ptr address
        * @return true if element have been allocated
        */
        bool isValidForFree(void* ptr) const;

        /** Get size of allocated item in bytes
        * @return number of bytes per item
        * @remark complexity: ~1
        * @see getBlockSize()
        */
        size_t getItemSize() const;

        /** Get size of block in bytes
        * @return number of bytes per block
        * @remark complexity: ~1
        */
        size_t getBlockSize() const;

        /** Get total allocated items
        * @return number of allocated items
        * @remark complexity: ~1
        */
        size_t getTotalAllocatedItems() const;

        /** Get total number of blocks
        * @return number memory blocks
        * @remark complexity: ~(NUMBER OF BLOCKS)
        */
        size_t getTotalBlocksCount() const;

        /** Empty mean that it is zero allocated items
        * @see getTotalAllocatedItems()
        * @return true if it is zero allocated items
        * @remark complexity: ~1
        */
        bool isEmpty() const;

    private:
        MemoryPool(const MemoryPool& rhs);
        MemoryPool& operator = (const MemoryPool& rhs);

    private:
        size_t blockSize;      ///< block size in bytes = number of elements in block X item size in bytes
        size_t itemSize;       ///< item size in bytes
        MemBlock* block;       ///< pointer to first mem-block
        size_t itemsAllocated; ///< number of allocated items via pool
    };

    /** Redefine allocation mechanism for single element objects
    */
    template<size_t size>
    struct MemoryPoolAllocator
    {
        static void* operator new(size_t sz) { return elementsPool.allocItem(); }
        static void operator delete(void* ptr) { elementsPool.freeItem(ptr); }
        static MemoryPool elementsPool;
    };
    template<size_t size>

    MemoryPool MemoryPoolAllocator<size>::elementsPool(size, 10);
}
