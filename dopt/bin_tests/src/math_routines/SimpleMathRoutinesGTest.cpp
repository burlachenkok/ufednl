#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/math_routines/include/SpecialMathRoutinesForMatrix.h"
#include "dopt/random/include/RandomGenRealLinear.h"

#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"

#include "gtest/gtest.h"

#include <stddef.h>
#include <vector>
#include <algorithm>
#include <set>

TEST(dopt, SimpleMathRoutinesGTest)
{
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m(5, 5);
    
    //             0    1    2   3   4
    m.setRow(0, { -5,  0,   1, -10, 2  });
    m.setRow(1, {  0,  12,  2, -14, 6  });
    m.setRow(2, {  1,  2,   1,   3, 17  });
    m.setRow(3, {-10,  -14, 3,  -9, -10 });
    m.setRow(4, {  2,  6,   17, -10, 2  });
    //                       0    1    2   3   4

    EXPECT_TRUE(m.rows()    == 5);
    EXPECT_TRUE(m.columns() == 5);
    EXPECT_TRUE(m.isSymmetric());
    EXPECT_TRUE(m.isSquareMatrix());

    {
        auto ctr = dopt::indiciesForUpperTriangularPart(m);
        
        std::set<uint32_t> set_idicies_1 = { ctr.begin(), ctr.end() };
        std::set<uint32_t> set_idicies_2;

        for (size_t i = 0; i < m.rows(); ++i)
        {
            for (size_t j = i; j < m.columns(); ++j)
            {
                set_idicies_2.insert(m.getFlattenIndexFromPosition(i, j));
            }
        }
        
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_idicies_1.begin(), set_idicies_1.end(),
                                      set_idicies_2.begin(), set_idicies_2.end(), std::back_inserter(diff));
        
        EXPECT_TRUE(diff.empty());
    }

    {
        dopt::MatrixNMD<dopt::VectorNDRaw_d> mm(13, 7);
        auto ctr = dopt::indiciesForUpperTriangularPart(mm);
        std::set<uint32_t> set_idicies_1 = { ctr.begin(), ctr.end() };
        std::set<uint32_t> set_idicies_2;
        
        for (size_t i = 0; i < mm.rows(); ++i)
        {
            for (size_t j = i; j < mm.columns(); ++j)
            {
                set_idicies_2.insert(mm.getFlattenIndexFromPosition(i, j));
            }
        }

        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_idicies_1.begin(), set_idicies_1.end(),
                                      set_idicies_2.begin(), set_idicies_2.end(), std::back_inserter(diff));

        EXPECT_TRUE(diff.empty());
    }

    {
        auto ctr = dopt::getTopKFromUpperDiagonalPart</*ignoreSign*/ false> (m, 1);
        std::set<uint32_t> set_top_1 = { ctr.begin(), ctr.end() };
        std::set<uint32_t> set_answer = { uint32_t(m.getFlattenIndexFromPosition(2, 4)) };
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_top_1.begin(), set_top_1.end(),
                                      set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }
    {
        auto ctr = dopt::getTopKFromUpperDiagonalPart</*ignoreSign*/ false> (m, 2);
        std::set<uint32_t> set_top_1 = { ctr.begin(), ctr.end() };
        std::set<uint32_t> set_answer = { uint32_t(m.getFlattenIndexFromPosition(2, 4)), 
                                          uint32_t(m.getFlattenIndexFromPosition(1, 1)) };
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_top_1.begin(), set_top_1.end(),
                                      set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }

    {
        auto mAbs = m;
        mAbs.matrixByCols = mAbs.matrixByCols.abs();

        auto ctr = dopt::getTopKFromUpperDiagonalPart</*ignoreSign*/ false>(mAbs, 2);
        std::set<uint32_t> set_top_1 = {ctr.begin(), ctr.end()};
        std::set<uint32_t> set_answer = { uint32_t(m.getFlattenIndexFromPosition(2, 4)),
                                          uint32_t(m.getFlattenIndexFromPosition(1, 3)) };
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_top_1.begin(), set_top_1.end(),
                                      set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }

    {
        auto ctr = dopt::getTopKFromUpperDiagonalPart</*ignoreSign*/ true> (m, 2);
        std::set<uint32_t> set_top_1 = {ctr.begin(), ctr.end()};
        std::set<uint32_t> set_answer = { uint32_t(m.getFlattenIndexFromPosition(2, 4)),
                                          uint32_t(m.getFlattenIndexFromPosition(1, 3)) };
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_top_1.begin(), set_top_1.end(),
            set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }

    {
        dopt::MatrixNMD<dopt::VectorNDRaw_d> mm(5, 5);

        //             0    1    2   3   4
        mm.setRow(0, {0,  5,   10,  15, 20 });
        mm.setRow(1, {1,  6,   11,  16, 21 });
        mm.setRow(2, {2,  7,   12,  17, 22 });
        mm.setRow(3, {3,  8,   13,  18, 23 });
        mm.setRow(4, {4,  9,   14,  19, 24 });

        auto ctr = dopt::getTopKFromUpperDiagonalPart</*ignoreSign*/ true>(mm, 3);
        std::set<uint32_t> set_top_1 = { ctr.begin(), ctr.end() };
        std::set<uint32_t> set_answer = { uint32_t(m.getFlattenIndexFromPosition(4, 4)),
                                          uint32_t(m.getFlattenIndexFromPosition(3, 4)),
                                          uint32_t(m.getFlattenIndexFromPosition(2, 4)) };
        std::vector<uint32_t> diff;

        std::set_symmetric_difference(set_top_1.begin(), set_top_1.end(),
            set_answer.begin(), set_answer.end(), std::back_inserter(diff));
        EXPECT_TRUE(diff.empty());
    }
}

TEST(dopt, SimpleMathRoutinesSortingGTest)
{
    std::vector<int> arr = { 10, -1, 2, 3 };
    auto cmp = [](int a, int b) {return a < b; };

    std::vector<int> arrSorted = { -1, 2, 3, 10 };

    {
        std::vector<int> arrCopy = arr;
        std::sort(arrCopy.begin(), arrCopy.end(), cmp);
        EXPECT_TRUE(arrCopy == arrSorted);
    }
    
    {
        std::vector<int> arrCopy = arr;
        dopt::insertionSort(&arrCopy[0], arrCopy.size(), cmp);
        EXPECT_TRUE(arrCopy == arrSorted);
    }

    {
        std::vector<int> draftStorage(arr.size(), -1);
        std::vector<int> arrCopy = arr;
        dopt::mergeSort<int, decltype(cmp), 0> (&arrCopy[0], arrCopy.size(), &draftStorage[0], cmp);
        EXPECT_TRUE(arrCopy == arrSorted);
    }

    std::vector<int> arrBig;
    for (int i = 0; i < 1003; ++i)
        arrBig.push_back(i % 13);

    {
        std::vector<int> arrCopy = arrBig;
        std::sort(arrCopy.begin(), arrCopy.end(), cmp);

        for (size_t i = 1; i < arrCopy.size(); ++i)
            EXPECT_TRUE(arrCopy[i] >= arrCopy[i - 1]);
    }

    {
        std::vector<int> arrCopy = arrBig;
        dopt::insertionSort(&arrCopy[0], arrCopy.size(), cmp);

        for (size_t i = 1; i < arrCopy.size(); ++i)
            EXPECT_TRUE(arrCopy[i] >= arrCopy[i - 1]);
    }

    {
        std::vector<int> draftStorage(arrBig.size(), -1);
        std::vector<int> arrCopy = arrBig;
        dopt::mergeSort<int, decltype(cmp), 0>(&arrCopy[0], arrCopy.size(), &draftStorage[0], cmp);

        for (size_t i = 1; i < arrCopy.size(); ++i)
            EXPECT_TRUE(arrCopy[i] >= arrCopy[i - 1]);
    }

    {
        std::vector<int> draftStorage(arrBig.size(), -1);
        std::vector<int> arrCopy = arrBig;
        dopt::quickSort<int, decltype(cmp), 0>(&arrCopy[0], arrCopy.size(), cmp);

        for (size_t i = 1; i < arrCopy.size(); ++i)
            EXPECT_TRUE(arrCopy[i] >= arrCopy[i - 1]);
    }

    {
        auto cmp = [](int a, int b) {return a < b; };

        std::vector<int> arrSortedArray = { 0, 1, 2, 3, 7, 5, 6, 4};
        std::vector<int> draftStorage(arrSortedArray.size(), -1);
        dopt::mergeSort<int, decltype(cmp), 0>(&arrSortedArray[0], arrSortedArray.size(), &draftStorage[0], cmp);
        for (size_t i = 1; i < arrSortedArray.size(); ++i)
            EXPECT_TRUE(arrSortedArray[i] >= arrSortedArray[i - 1]);
    }

    {
        std::vector<int> arrCopy = arrBig;
        dopt::radixSort(&arrCopy[0], arrCopy.size());

        for (size_t i = 1; i < arrCopy.size(); ++i)
            EXPECT_TRUE(arrCopy[i] >= arrCopy[i - 1]);
    }

    {
        std::vector<unsigned char> arrTest = { 1, 3, 2, 200, 100};
        dopt::radixSort(&arrTest[0], arrTest.size());

        for (size_t i = 1; i < arrTest.size(); ++i)
            EXPECT_TRUE(arrTest[i] >= arrTest[i - 1]);
    }
}

TEST(dopt, SimpleMathRoutinesStatisticsGTest)
{
    std::vector<int> arr = { 10, -1, 2, 3 };
    auto cmp = [](int a, int b) {return a < b; };

    {
        std::vector<int> arrCopy = arr;
        int iThSmallest = dopt::getIthSmallestItem(arrCopy.data(), arrCopy.size(), cmp, 0);
        EXPECT_TRUE(iThSmallest == -1);
    }

    {
        std::vector<int> arrCopy = arr;
        int iThSmallest = dopt::getIthSmallestItem(arrCopy.data(), arrCopy.size(), cmp, 1);
        EXPECT_TRUE(iThSmallest == 2);
    }

    {
        std::vector<int> arrCopy = arr;
        int iThSmallest = dopt::getIthSmallestItem(arrCopy.data(), arrCopy.size(), cmp, 2);
        EXPECT_TRUE(iThSmallest == 3);
    }

    {
        std::vector<int> arrCopy = arr;
        int iThSmallest = dopt::getIthSmallestItem(arrCopy.data(), arrCopy.size(), cmp, 3);
        EXPECT_TRUE(iThSmallest == 10);
    }
}

struct Pair2TestStable
{
    Pair2TestStable(int theX = 0, int theY = 0)
    : x(theX)
    , y(theY)
    {}

    bool operator < (const Pair2TestStable& rhs)
    {
        return x < rhs.x;
    }

    operator int() const
    {
        return x;
    }

    int x;
    int y;
};

TEST(dopt, SimpleMathRoutinesSortingStabilityGTest)
{
    uint32_t size = 1513;
    std::vector<Pair2TestStable> input;
    dopt::RandomGenIntegerLinear randX(0, size / 5);

    for (uint32_t i = 0; i < size; ++i)
        input.push_back(Pair2TestStable(randX.generateInteger(), i));
    auto cmp = [](Pair2TestStable a, Pair2TestStable b) {return (int)a < (int)b; };

    {
        std::vector<Pair2TestStable> arrCopy = input;
        dopt::insertionSort(&arrCopy[0], arrCopy.size(), cmp);

        for (size_t i = 1; i < size; ++i)
        {
            if (arrCopy[i].x == arrCopy[i - 1].x)
                EXPECT_TRUE(arrCopy[i].y > arrCopy[i - 1].y);
            else
                EXPECT_TRUE(arrCopy[i].x > arrCopy[i - 1].x);
        }
    }

    {
        std::vector<Pair2TestStable> arrCopy = input;
        std::vector<Pair2TestStable> draftStorage(input.size(), Pair2TestStable(1,1));

        dopt::mergeSort(&arrCopy[0], arrCopy.size(), &draftStorage[0], cmp);

        for (size_t i = 1; i < size; ++i)
        {
            if (arrCopy[i].x == arrCopy[i - 1].x)
                EXPECT_TRUE(arrCopy[i].y > arrCopy[i - 1].y);
            else
                EXPECT_TRUE(arrCopy[i].x > arrCopy[i - 1].x);
        }
    }
}

static_assert(std::is_integral<uint8_t>::value && std::is_integral<uint8_t>::value && std::is_integral<uint8_t>::value);
static_assert(std::is_integral<int8_t>::value && std::is_integral<int8_t>::value && std::is_integral<int8_t>::value);
static_assert(std::is_integral<size_t>::value);
static_assert(!std::is_integral<float>::value && !std::is_integral<double>::value );

TEST(dopt, SimpleMathRoutinesSortingPartialGTest)
{
    int size = 513;
    std::vector<int> input;
    for (int i = 0; i < size; ++i)
        input.push_back(i);
    dopt::RandomGenIntegerLinear rndGenerator;
    dopt::shuffle(input, size, rndGenerator);

    std::vector<int> draftStorage(input.size(), -1);
    auto cmp = [](int a, int b) {return a < b; };

    for (int k = 1; k < 30; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::mergeSortPartial<int, decltype(cmp), 0> (&arrCopy[0], arrCopy.size(), k, &draftStorage[0], cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
        EXPECT_FALSE(arrCopy[k] == k);
    }

    for (int k = 1; k < 30; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::mergeSortPartial<int, decltype(cmp), 3>(&arrCopy[0], arrCopy.size(), k, &draftStorage[0], cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
        EXPECT_FALSE(arrCopy[k] == k);
    }

    for (int k = 1; k < 30; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::mergeSortPartial<int, decltype(cmp), 13>(&arrCopy[0], arrCopy.size(), k, &draftStorage[0], cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
        EXPECT_FALSE(arrCopy[k] == k);
    }

    for (int k = 1; k < 30; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::mergeSortPartial<int, decltype(cmp)>(&arrCopy[0], arrCopy.size(), k, &draftStorage[0], cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
        EXPECT_FALSE(arrCopy[k] == k);
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::quickSortExtended<int, decltype(cmp), 0, false>(&arrCopy[0], arrCopy.size(), k, cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::quickSortExtended<int, decltype(cmp), 1, false>(&arrCopy[0], arrCopy.size(), k, cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::quickSortExtended<int, decltype(cmp), 12, false>(&arrCopy[0], arrCopy.size(), k, cmp);
        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(arrCopy[i] == i);
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::quickSortExtended<int, decltype(cmp), 12, true>(&arrCopy[0], arrCopy.size(), k, cmp);

        std::set<int> kSmallestItems;
        for (int i = 0; i < k; ++i)
            kSmallestItems.insert(i);

        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(kSmallestItems.contains(arrCopy[i]));
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::mergeSortPartial<int, decltype(cmp), 0>(&arrCopy[0], arrCopy.size(), k, &draftStorage[0], cmp);

        std::set<int> kSmallestItems;
        for (int i = 0; i < k; ++i)
            kSmallestItems.insert(i);

        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(kSmallestItems.contains(arrCopy[i]));
    }

    for (int k = 1; k < 40; ++k)
    {
        std::vector<int> arrCopy = input;
        dopt::selectionSortExtended(&arrCopy[0], arrCopy.size(), k, cmp);

        std::set<int> kSmallestItems;
        for (int i = 0; i < k; ++i)
            kSmallestItems.insert(i);

        for (int i = 0; i < k; ++i)
            EXPECT_TRUE(kSmallestItems.contains(arrCopy[i]));
    }
}

TEST(dopt, SanityCheckMinimumGTest)
{
    {
        uint32_t a = 1, b = 33;
        EXPECT_TRUE(dopt::minimum(a, b) == 1);
        EXPECT_TRUE(dopt::maximum(a, b) == 33);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        int32_t a = 1, b = -33;
        EXPECT_TRUE(dopt::minimum(a, b) == -33);
        EXPECT_TRUE(dopt::maximum(a, b) == 1);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }
    
    {
        uint16_t a = 1, b = 133;
        EXPECT_TRUE(dopt::minimum(a, b) == 1);
        EXPECT_TRUE(dopt::maximum(a, b) == 133);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        int16_t a = -100, b = 313;
        EXPECT_TRUE(dopt::minimum(a, b) == -100);
        EXPECT_TRUE(dopt::maximum(a, b) == 313);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        int64_t a = 1, b = 33;
        EXPECT_TRUE(dopt::minimum(a, b) == 1);
        EXPECT_TRUE(dopt::maximum(a, b) == 33);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        float a = 1.0f, b = -33.0f;
        EXPECT_TRUE(dopt::minimum(a, b) == -33.0f);
        EXPECT_TRUE(dopt::maximum(a, b) == 1.0f);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimum(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        double a = -1.0, b = 33.0;
        EXPECT_TRUE(dopt::minimum(a, b) == -1.0);
        EXPECT_TRUE(dopt::maximum(a, b) == 33.0);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimum(b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }
    
    {
        float a = 1.0f, b = 33.0f;
        EXPECT_TRUE(dopt::minimum(a, b) == 1.0f);
        EXPECT_TRUE(dopt::maximum(a, b) == 33.0f);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimum < /*promise that number are non-neg*/true> (b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximum < /*promise that number are non-neg*/true> (a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }

    {
        double a = 1.0, b = 33.0;
        EXPECT_TRUE(dopt::minimum(a, b) == 1.0);
        EXPECT_TRUE(dopt::maximum(a, b) == 33.0);

        EXPECT_TRUE(dopt::minimum(b, a) == dopt::minimum(a, b));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::noBracnhMinimum < /*promise that number are non-neg*/true> (b, a));
        EXPECT_TRUE(dopt::minimum(a, b) == dopt::bracnhMinimum(b, a));

        EXPECT_TRUE(dopt::maximum(b, a) == dopt::maximum(a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::noBracnhMaximum < /*promise that number are non-neg*/true> (a, b));
        EXPECT_TRUE(dopt::maximum(b, a) == dopt::bracnhMaximum(a, b));
    }
    
    {
        uint32_t a = 11, b = 3;
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(13)) == (11 + 3) % 13);
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(12)) == (11 + 3) % 12);
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(14)) == 0);
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(15)) == 14);

        a = 0, b = 0;
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(12)) == 0);
        a = 0, b = 1;
        EXPECT_TRUE(dopt::add_two_numbers_modN(a, b, uint32_t(12)) == 1);
    }
}

TEST(dopt, SimpleMathRoutinesRoundingTest)
{

    {
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(128), 7);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(120), 6);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(119), 6);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(64), 6);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(32), 5);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(16), 4);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(8),  3);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(4),  2);
        EXPECT_EQ(dopt::log2AtCompileTimeLowerBound(2),  1);


    }
    {
        size_t r1 = dopt::roundToNearestMultipleUp<4>(10);
        EXPECT_EQ(r1, 12);

        size_t r1d = dopt::roundToNearestMultipleDown<4>(10);
        EXPECT_EQ(r1d, 8);

        size_t r2 = dopt::roundToNearestMultipleUp<6>(12);
        EXPECT_EQ(r2, 12);

        size_t r2d = dopt::roundToNearestMultipleDown<6>(12);
        EXPECT_EQ(r2d, 12);

        size_t r3 = dopt::roundToNearestMultipleUp<4>(4);
        EXPECT_EQ(r3, 4);

        size_t r3d = dopt::roundToNearestMultipleDown<4>(4);
        EXPECT_EQ(r3d, 4);

        for (size_t i = 0; i < 200; ++i)
        {
            size_t rTest = dopt::roundToNearestMultipleUp<7>(i);
            EXPECT_TRUE(rTest % 7 == 0);

            if (i % 7 == 0)
            {
                EXPECT_TRUE(rTest == i);
            }

            EXPECT_TRUE(rTest >= i);
        }

        for (size_t i = 0; i < 200; ++i)
        {
            size_t rTest = dopt::roundToNearestMultipleDown<7>(i);
            EXPECT_TRUE(rTest % 7 == 0);

            if (i % 7 == 0)
            {
                EXPECT_TRUE(rTest == i);
            }

            EXPECT_TRUE(rTest <= i);
        }
    }
}

TEST(dopt, SimpleMathRoutinesSparsificationRandKGTest)
{
    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenIntegerLinear gen_index;

    gen_x.setSeed(123);
    gen_index.setSeed(456);
    
    size_t d = 20;
    size_t k = 30;
    size_t multFactor = (double(d) * double(d+1) / 2.0) / double(k);
    
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_out(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_est_mean(d, d);

    m.setAllRandomly(gen_x);
    m = (m + m.getTranspose()) / 2.0;

    m_out.setAllToDefault();
    m_est_mean.setAllToDefault();
    
    double denominator = m.frobeniusNorm();
    denominator *= denominator;

    double running_mean_for_numerator = 0.0;
    
    for (size_t trial = 0; trial < 1000; ++trial)
    {
        std::vector<uint32_t> indicies = dopt::generateRandKItemsInUpperTriangularPart(gen_index, k, m);
        m_out.setAllToDefault();

        for (size_t i = 0; i < indicies.size(); ++i)
        {
            uint32_t ind = indicies[i];
            size_t iRow = 0, iCol = 0;
            m.getPositionFromFlatternIndex(iRow, iCol, ind);

            m_out.getRaw(iRow, iCol) = multFactor * m.getRaw(iRow, iCol);
            m_out.getRaw(iCol, iRow) = multFactor * m.getRaw(iCol, iRow);
        }

        double numerator = (m_out - m).frobeniusNorm();
        numerator *= numerator;

        running_mean_for_numerator = (running_mean_for_numerator * double(trial) + numerator) / double(trial + 1);
        m_est_mean = (m_est_mean * double(trial) + m_out) / double(trial + 1);
    }
    
    double w_empirical = running_mean_for_numerator / denominator;
    double w_theoretical = dopt::computeWForRandKMatrixOperator(k, d);
    EXPECT_TRUE(fabs(w_empirical - w_theoretical) <= w_theoretical * 0.01);
    
    double descr = (m_est_mean - m).frobeniusNorm();
    double m_F = m.frobeniusNorm();
    EXPECT_TRUE(descr < m_F * 0.1);
    std::cout << " descripancy for RandK for test matrix in absolute scale: " << descr << '\n';
}

TEST(dopt, SimpleMathRoutinesSparsificationRandSeqKGTest)
{
    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenIntegerLinear gen_index;

    gen_x.setSeed(123);
    gen_index.setSeed(456);

    size_t d = 20;
    size_t k = 30;
    size_t multFactor = (double(d) * double(d + 1) / 2.0) / double(k);

    dopt::MatrixNMD<dopt::VectorNDRaw_d> m(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_out(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_est_mean(d, d);

    m.setAllRandomly(gen_x);
    m = (m + m.getTranspose()) / 2.0;

    m_out.setAllToDefault();
    m_est_mean.setAllToDefault();

    double denominator = m.frobeniusNorm();
    denominator *= denominator;

    double running_mean_for_numerator = 0.0;

    for (size_t trial = 0; trial < 1000; ++trial)
    {
        std::vector<uint32_t> indicies = dopt::generateRandSeqKItemsInUpperTriangularPart(gen_index, k, m);
        m_out.setAllToDefault();

        for (size_t i = 0; i < indicies.size(); ++i)
        {
            uint32_t ind = indicies[i];
            size_t iRow = 0, iCol = 0;
            m.getPositionFromFlatternIndex(iRow, iCol, ind);

            m_out.getRaw(iRow, iCol) = multFactor * m.getRaw(iRow, iCol);
            m_out.getRaw(iCol, iRow) = multFactor * m.getRaw(iCol, iRow);
        }

        double numerator = (m_out - m).frobeniusNorm();
        numerator *= numerator;

        running_mean_for_numerator = (running_mean_for_numerator * double(trial) + numerator) / double(trial + 1);
        m_est_mean = (m_est_mean * double(trial) + m_out) / double(trial + 1);
    }

    double w_empirical = running_mean_for_numerator / denominator;
    double w_theoretical = dopt::computeWForRandKMatrixOperator(k, d);
    EXPECT_TRUE(fabs(w_empirical - w_theoretical) <= w_theoretical * 0.01);

    double descr = (m_est_mean - m).frobeniusNorm();
    double m_F = m.frobeniusNorm();
    EXPECT_TRUE(descr < m_F * 0.1);
    std::cout << " descripancy for RandSeqK for test matrix in absolute scale: " << descr << '\n';

    gen_index.setSeed(1234);
    std::vector<uint32_t> indicies_a = dopt::generateRandSeqKItemsInUpperTriangularPart(gen_index, k, m);
    
    gen_index.setSeed(1234);
    uint32_t indicies_b = dopt::generateRandSeqKItemsInUpperTriangularPartAsIndex(gen_index, k, m);
    std::vector<uint32_t> upper_triangular_part = indiciesForUpperTriangularPart(m);
    
    EXPECT_TRUE(indicies_a.size() == k);
    
    size_t upperTriangPartSize = upper_triangular_part.size();
    
    EXPECT_TRUE(((m.rows() + 1) * m.rows()) / 2 == upperTriangPartSize);
    
    for (uint32_t i = 0; i < k; ++i)
    {
        uint32_t iA = indicies_a[i];
        uint32_t iB = upper_triangular_part[dopt::add_two_numbers_modN(indicies_b, i, uint32_t(upperTriangPartSize))];
        EXPECT_TRUE(iA == iB);
    }
}


TEST(dopt, SimpleMathRoutinesSparsificationTopKGTest)
{
    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenIntegerLinear gen_index;

    gen_x.setSeed(123);
    gen_index.setSeed(456);

    size_t d = 20;
    size_t k = 30;

    dopt::MatrixNMD<dopt::VectorNDRaw_d> m(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_out(d, d);

    double one_minus_alpha_empirical = -1.0;

    for (size_t trial = 0; trial < 1000; ++trial)
    {
        m.setAllRandomly(gen_x);
        m = (m + m.getTranspose()) / 2.0;

        double denominator = m.frobeniusNorm();
        denominator *= denominator;

        m_out.setAllToDefault();
        std::vector<uint32_t> indicies = dopt::getTopKFromUpperDiagonalPart<true>(m, k);
        m_out.setAllToDefault();

        for (size_t i = 0; i < indicies.size(); ++i)
        {
            uint32_t ind = indicies[i];
            size_t iRow = 0, iCol = 0;
            m.getPositionFromFlatternIndex(iRow, iCol, ind);
            m_out.getRaw(iRow, iCol) = m.getRaw(iRow, iCol);
            m_out.getRaw(iCol, iRow) = m.getRaw(iCol, iRow);
        }

        double numerator = (m_out - m).frobeniusNorm();
        numerator *= numerator;

        double one_mins_alha_4_trial = numerator / denominator;

        if (one_mins_alha_4_trial > one_minus_alpha_empirical)
            one_minus_alpha_empirical = one_mins_alha_4_trial;
    }

    double one_minus_alpha_th = 1.0 - dopt::computeDeltaForTopKMatrixOpeator(k, d);
    EXPECT_TRUE(one_minus_alpha_empirical <= one_minus_alpha_th);


    {
        size_t dSmall = 2;
        size_t kSmall = 2;
        
        // Special worst case test
        dopt::MatrixNMD<dopt::VectorNDRaw_d> m(dSmall, dSmall);
        dopt::MatrixNMD<dopt::VectorNDRaw_d> m_out(dSmall, dSmall);

        m.setAll(1.0);
        m.addToAllDiagonalEntries(0.00001);
        m_out.setAllToDefault();

        std::vector<uint32_t> indicies = dopt::getTopKFromUpperDiagonalPart<true>(m, kSmall);

        m_out.setAllToDefault();

        for (size_t i = 0; i < indicies.size(); ++i)
        {
            uint32_t ind = indicies[i];
            size_t iRow = 0, iCol = 0;
            m.getPositionFromFlatternIndex(iRow, iCol, ind);
            m_out.getRaw(iRow, iCol) = m.getRaw(iRow, iCol);
            m_out.getRaw(iCol, iRow) = m.getRaw(iCol, iRow);
        }

        double numerator   = (m_out - m).frobeniusNorm();
        numerator *= numerator;

        double denominator = (m).frobeniusNorm();
        denominator *= denominator;

        double one_minus_alha_4_instance = numerator / denominator;
        double one_minus_alpha_4_instance_th = 1.0 - dopt::computeDeltaForTopKMatrixOpeator(kSmall, dSmall);

#if DOPT_FIX_TOPK_CONTRACTION_FACTOR
        EXPECT_TRUE(one_minus_alha_4_instance <= one_minus_alpha_4_instance_th + 1e-5);
#endif
    }
}


TEST(dopt, SimpleMathRoutinesSparsificationTopLEKGTest)
{
    dopt::RandomGenRealLinear gen_x(-1.0, 1.0);
    dopt::RandomGenIntegerLinear gen_index;

    gen_x.setSeed(123);
    gen_index.setSeed(456);

    size_t d = 20;
    size_t k = 30;

    dopt::MatrixNMD<dopt::VectorNDRaw_d> m(d, d);
    dopt::MatrixNMD<dopt::VectorNDRaw_d> m_out(d, d);

    double one_minus_alpha_empirical = -1.0;
    std::vector<uint32_t> indicies4up_part = dopt::indiciesForUpperTriangularPart(m);

    for (size_t trial = 0; trial < 250; ++trial)
    {
        m.setAllRandomly(gen_x);
        m = (m + m.getTranspose()) / 2.0;
        double denominator = m.frobeniusNorm();
        denominator *= denominator;

        double running_mean_for_numerator = 0.0;

        for (size_t internal_trial = 0; internal_trial < 1000; ++internal_trial)
        {
            m_out.setAllToDefault();
            std::vector<uint32_t> indicies = dopt::getTopLEKFromUpperDiagonalPart<true>(m, k,
                                                                                        indicies4up_part,
                                                                                        dopt::computeDeltaForTopKMatrixOpeator(k, d));
            m_out.setAllToDefault();

            for (size_t i = 0; i < indicies.size(); ++i)
            {
                uint32_t ind = indicies[i];
                size_t iRow = 0, iCol = 0;
                m.getPositionFromFlatternIndex(iRow, iCol, ind);
                m_out.getRaw(iRow, iCol) = m.getRaw(iRow, iCol);
                m_out.getRaw(iCol, iRow) = m.getRaw(iCol, iRow);
            }
            double numerator = (m_out - m).frobeniusNorm();
            numerator *= numerator;
            running_mean_for_numerator = (running_mean_for_numerator * double(internal_trial) + numerator) / double(internal_trial + 1);
        }


        double one_mins_alha_4_trial = running_mean_for_numerator / denominator;
        if (one_mins_alha_4_trial > one_minus_alpha_empirical)
            one_minus_alpha_empirical = one_mins_alha_4_trial;
    }

    double one_minus_alpha_th = 1.0 - dopt::computeDeltaForTopKMatrixOpeator(k, d);
    EXPECT_TRUE(fabs(one_minus_alpha_empirical - one_minus_alpha_th) <= 0.1 * one_minus_alpha_th);
}


TEST(dopt, SimpleMathComparisions)
{
    EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(14, 12));
    EXPECT_FALSE(dopt::isFirstHigherThenSecondIgnoreSign(12, 14));

    EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(14, -12));
    EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(-14, 12));

    EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(-14.0f, -12.0f));
    EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(-14.0, -12.0));

    constexpr double step = 0.1;
    for (double a = -100.0; a < -10.0; a += step)
    {
        for (double b = a + 1.0; b < -10.0; b += step)
        {
            EXPECT_TRUE(dopt::isFirstHigherThenSecondIgnoreSign(a, b));
        }
    }
    
}
