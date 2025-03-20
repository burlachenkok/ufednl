#include "dopt/system/include/MemoryPool.h"

#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <stdint.h>

namespace
{
    struct AIntrernal
    {
        int a;
    };

    struct A final: public dopt::MemoryPoolAllocator<sizeof(AIntrernal)>
    {
        AIntrernal aInternal;
    };

    static_assert(sizeof(A) == sizeof(AIntrernal), "MemoryPoolAllocator specification should contain correct size");
}

TEST(dopt, MemoryPoolGTest)
{
    {
        A* b = new A[10];
        EXPECT_TRUE(b != NULL);
        EXPECT_EQ(0, A::elementsPool.getTotalAllocatedItems());
        delete []b;

        A* a = new A;
        EXPECT_TRUE(a != NULL);
        EXPECT_EQ(1, A::elementsPool.getTotalAllocatedItems());
        delete a;
        EXPECT_EQ(0, A::elementsPool.getTotalAllocatedItems());
    }
    {
	    dopt::MemoryPool pool(sizeof(uint16_t), 5);
        EXPECT_EQ(0, pool.getTotalAllocatedItems());
        EXPECT_TRUE(pool.allocItem() != NULL);
        EXPECT_TRUE(pool.allocItem() != NULL);
        EXPECT_EQ(2, pool.getTotalAllocatedItems());
        pool.freeAll();
        EXPECT_EQ(0, pool.getTotalAllocatedItems());
        EXPECT_TRUE(pool.isEmpty());
        
        void* ptrA = pool.allocItem();
        void* ptrB = pool.allocItem();
        EXPECT_TRUE(ptrA != NULL);
        EXPECT_TRUE(ptrB != NULL);

        EXPECT_TRUE(pool.isValidForFree(ptrA));
        EXPECT_TRUE(pool.freeItem(ptrA));
        EXPECT_FALSE(pool.isValidForFree(ptrA));
        EXPECT_FALSE(pool.freeItem(ptrA));
        EXPECT_FALSE(pool.freeItem(NULL));
        EXPECT_EQ(1, pool.getTotalAllocatedItems());
        EXPECT_TRUE(pool.freeItem(ptrB));
        EXPECT_EQ(0, pool.getTotalAllocatedItems());
        void* ptr = pool.allocItem();
        EXPECT_EQ(1, pool.getTotalAllocatedItems());
        EXPECT_EQ(ptrB, ptr) << "Check MemPool feature with fast allocate & deallocate";
        EXPECT_EQ(1, pool.getTotalBlocksCount()) << "It seems that for such small number of objects in pool no need to create extra blocks";
        EXPECT_TRUE(size_t(ptr) % 2 == 0);
        EXPECT_TRUE(size_t(ptrB) % 2 == 0);
        EXPECT_TRUE(size_t(ptrA) % 2 == 0);
    }

    {
        dopt::MemoryPool pool(sizeof(int16_t), 5);
        for (int i = 0; i < 5; ++i)
            EXPECT_TRUE(pool.allocItem() != NULL);
        EXPECT_EQ(1, pool.getTotalBlocksCount());
        EXPECT_TRUE(pool.allocItem() != NULL);
        EXPECT_EQ(2, pool.getTotalBlocksCount());
        EXPECT_EQ(6, pool.getTotalAllocatedItems());
    }

    {
        void* ptrs[10];
        dopt::MemoryPool pool(sizeof(int64_t), 6);
        for (int i = 0; i < 10; ++i)
        {
            ptrs[i] = pool.allocItem();
            EXPECT_TRUE(ptrs[i] != NULL);
        }
        EXPECT_EQ(2, pool.getTotalBlocksCount());
        EXPECT_EQ(10, pool.getTotalAllocatedItems());
        for (int i = 9; i >= 0; --i)
        {
            EXPECT_TRUE(pool.freeItem(ptrs[i]));
        }
        EXPECT_EQ(0, pool.getTotalAllocatedItems());
    }
}
