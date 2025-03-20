#include "dopt/math_routines/include/MinHeapIndexed.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <vector>
#include <stdint.h>

namespace
{
    int32_t generateInt32()
    {
        typedef int32_t T;
        double val = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        T returnValue = static_cast<T>(val * std::numeric_limits<T>::max());
        return returnValue;
    };
}

TEST(dopt, MinHeapIndexedGTest)
{
    {
        auto cmp = [](int a, int b) {return a > b; };
        dopt::MinHeapIndexed<int, size_t, std::vector<int>, std::vector<size_t>, 2, decltype(cmp)> m(10, cmp);

        EXPECT_TRUE(m.isEmpty());
        EXPECT_TRUE(m.insert(5, 50));
        EXPECT_TRUE(m.hasItem(5));
        EXPECT_FALSE(m.hasItem(4));
        
        EXPECT_FALSE(m.insert(5, 40));
        EXPECT_TRUE(m.insert(1, 100));
        EXPECT_EQ(2, m.size());
        EXPECT_EQ(10, m.capacity());

        EXPECT_EQ(50, m.minElement());
        EXPECT_EQ(5, m.minIndex());
        EXPECT_EQ(50, m.get(5));
        EXPECT_EQ(100, m.get(1));

        m.change(2, 20);
        EXPECT_EQ(3, m.size());
        EXPECT_EQ(20, m.minElement());
        m.change(5, 10);
        EXPECT_EQ(10, m.minElement());
    }

    {
        auto cmp = [](int32_t a, int32_t b) {return a > b; };
        dopt::MinHeapIndexed<int32_t, uint16_t,
                             std::vector<int32_t>, std::vector<uint16_t>, 2, decltype(cmp)> m(1000, cmp);
        for (size_t i = 0; i < 1000; ++i)
        {
            m.insert(i, generateInt32());
            m.change(i, generateInt32());
            EXPECT_EQ(i + 1, m.size());
            EXPECT_EQ(1000, m.capacity());
        }

        {
            std::vector<int32_t> extracted;

            while (!m.isEmpty())
                extracted.push_back(m.extractMinimum());

            EXPECT_EQ(1000, extracted.size());
        }
    }

    {
        auto cmp = [](int32_t a, int32_t b) {return a > b; };
        dopt::MinHeapIndexed<int32_t, uint16_t,
            std::vector<int32_t>, std::vector<uint16_t>, 2, decltype(cmp)> m(1000, cmp);

        // Permutation via using one-to-one mapping in Zp
        for (size_t i = 0; i < 79;++i)
            m.insert(i, (4*i) % 79);

        EXPECT_TRUE(m.peekMinimum() == m.peekMinimum());
        EXPECT_TRUE(m.size() == 79);
        EXPECT_TRUE(m.capacity() == 1000);

        std::vector<int32_t> extracted;
        while (!m.isEmpty())
            extracted.push_back(m.extractMinimum());

        for (size_t i = 0; i < 79; ++i)
            EXPECT_TRUE(extracted[i] == i);

        EXPECT_TRUE(m.insert(0, 12));
        EXPECT_FALSE(m.insert(0, 12));
        EXPECT_TRUE(m.insert(10, 19));
        EXPECT_TRUE(m.insert(8, 10));

        auto heapItems = m.getSetOfItems();

        EXPECT_TRUE(heapItems.size() == 3);

        std::set<int32_t> heapItemsSet = { heapItems.begin(), heapItems.end() };
        std::set<int32_t> set_answer = { 10, 12, 19 };
        std::vector<uint32_t> diff;
        std::set_symmetric_difference(heapItemsSet.begin(), heapItemsSet.end(),set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }
}
