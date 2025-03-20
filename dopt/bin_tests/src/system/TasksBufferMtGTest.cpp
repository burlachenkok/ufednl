#include "dopt/system/include/threads/Thread.h"
#include "dopt/system/include/threads/TasksBufferMt.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>
#include <thread>
#include <stdint.h>

template <class T>
void TasksBufferMtExecGTest()
{
    // Test wait with pop and push
    {
        dopt::TasksBufferMt<T> stack(3);
        EXPECT_TRUE(stack.isEmpty());

        T res = 0;
        EXPECT_FALSE(stack.popTask(res, false));
        EXPECT_FALSE(stack.popTask(res, false));

        EXPECT_TRUE(stack.pushTask(1, false));
        EXPECT_TRUE(stack.pushTask(2, false));
        EXPECT_TRUE(stack.pushTask(3, false));
        EXPECT_FALSE(stack.pushTask(4, false));

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 3);

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 2);

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 1);

        EXPECT_FALSE(stack.popTask(res, false));
        EXPECT_TRUE(res == 1);
    }
    
    // Single thread access
    {
        dopt::TasksBufferMt<T> stack(3);
        EXPECT_TRUE(stack.isEmpty());
        
        T res = 0;
        EXPECT_FALSE(stack.popTask(res, false));
        EXPECT_FALSE(stack.popTask(res, false));
        EXPECT_TRUE(stack.isEmpty());
        stack.clear();

        EXPECT_TRUE(stack.pushTask(1, true));
        EXPECT_TRUE(stack.pushTask(2, false));
        EXPECT_TRUE(stack.pushTask(3, true));

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 3);

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 2);

        EXPECT_TRUE(stack.popTask(res));
        EXPECT_TRUE(res == 1);

        stack.pushTask(11);
        stack.pushTask(12);
        EXPECT_FALSE(stack.isEmpty());
        stack.clear();
        EXPECT_TRUE(stack.isEmpty());
    }

    // Multiple thread access (multiple push)
    {
        dopt::TasksBufferMt<T> stack(5);

        size_t kThreads = 20;
        size_t kIterationsInsideThread = 100;       

        auto add_numer = [&stack, kIterationsInsideThread] ()
        {
            for (size_t j = 0; j < kIterationsInsideThread; ++j)
            {
                std::this_thread::yield();
                stack.pushTask(1);
                std::this_thread::yield();
                stack.pushTask(2);
                std::this_thread::yield();
                stack.pushTask(3);
                std::this_thread::yield();
                stack.pushTask(4);
                std::this_thread::yield();
                stack.pushTask(5);
                std::this_thread::yield();
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < kThreads; ++i)
            threads.emplace_back(std::thread(add_numer));
       
        std::vector<int> results;
        while (results.size() != kIterationsInsideThread * kThreads * 5)
        {
            T r = -1;
            stack.popTask(r);
            results.push_back(r);
        }

        for (size_t i = 0; i < kThreads; ++i)
            threads[i].join();

        EXPECT_TRUE(results.size() == kThreads * kIterationsInsideThread * 5);
        size_t counters[10] = {};
        
        size_t unordered_numers = 0;
        constexpr size_t orderedSequence[5] = {5, 4, 3, 2, 1};

        for (size_t i = 0; i < results.size(); ++i)
        {
            size_t item = results[i];
            ASSERT_TRUE(item >= 0 && item < 10);

            counters[item] += 1;

            if (item != orderedSequence[i % 5])
            {
                unordered_numers += 1;
            }
        }
        std::cout << "  unordered_numers: " << unordered_numers << " from totally " << results.size() << '\n';

        if (std::thread::hardware_concurrency() > 1)
        {
            EXPECT_TRUE(unordered_numers > 0) << "It's very unlikely that all numbers will be ordered correctly [possible, but unlikely]\n";
        }
        
        EXPECT_TRUE(counters[0] == 0);
        EXPECT_TRUE(counters[1] == kThreads * kIterationsInsideThread) << "Check correct insertions of 1";
        EXPECT_TRUE(counters[2] == kThreads * kIterationsInsideThread) << "Check correct insertions of 2";
        EXPECT_TRUE(counters[3] == kThreads * kIterationsInsideThread) << "Check correct insertions of 3";
        EXPECT_TRUE(counters[4] == kThreads * kIterationsInsideThread) << "Check correct insertions of 4";
        EXPECT_TRUE(counters[5] == kThreads * kIterationsInsideThread) << "Check correct insertions of 5";
        EXPECT_TRUE(counters[6] == 0);
        EXPECT_TRUE(counters[7] == 0);
        EXPECT_TRUE(counters[8] == 0);
        EXPECT_TRUE(counters[9] == 0);
    }

    // Multiple thread access (multiple pop)
    {
        size_t kThreads = 20;
        size_t kIterationsInsideThread = 10;
        size_t kIterationsTotal = kThreads * kIterationsInsideThread;

        dopt::TasksBufferMt<T> stack(kIterationsTotal);

        for (size_t i = 1; i <= kIterationsTotal; ++i)
            stack.pushTask(i);

        std::atomic<int64_t> sum = 0;
        
        auto pop_numer = [&stack, &sum, kIterationsInsideThread]()
        {
            for (size_t j = 0; j < kIterationsInsideThread; ++j)
            {
                std::this_thread::yield();
                T res = 0;
                stack.popTask(res);
                
                sum += res;
            }
        };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < kThreads; ++i)
            threads.emplace_back(std::thread(pop_numer));

        for (size_t i = 0; i < kThreads; ++i)
            threads[i].join();

        EXPECT_TRUE(stack.isEmpty());
        EXPECT_TRUE(sum == (kIterationsTotal * (kIterationsTotal + 1)) / 2);
    }
}


TEST(dopt, TasksBufferMtGTest)
{
    std::cout << "  Hint from C++ Runtime: " << std::thread::hardware_concurrency() << " concurrent threads are supported.\n";

    TasksBufferMtExecGTest<int16_t>();
    TasksBufferMtExecGTest<int32_t>();
    TasksBufferMtExecGTest<int64_t>();

    TasksBufferMtExecGTest<uint8_t>();
    TasksBufferMtExecGTest<uint16_t>();
    TasksBufferMtExecGTest<uint32_t>();
    TasksBufferMtExecGTest<uint64_t>();
}
