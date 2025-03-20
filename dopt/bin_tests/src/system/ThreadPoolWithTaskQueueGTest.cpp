#include "dopt/system/include/threads/ThreadPoolWithTaskQueue.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>
#include <thread>
#include <atomic>

#include <stdint.h>

TEST(dopt, ThreadPoolWithTaskQueueGTest)
{
    // Parallel execution of work, but once list of job is ready
    {
        std::atomic<int> res = 0;

        struct MyTask
        {
            MyTask(std::atomic<int>* theAccum, int theNumberToAdd)
                : accum(theAccum)
                , number_to_add(theNumberToAdd)
            {}

            MyTask()
                : accum(0)
                , number_to_add(0)
            {}


            std::atomic<int>* accum;
            int number_to_add;
        };

        auto add_numer = [](const MyTask& task, size_t) {
            *(task.accum) += task.number_to_add;
        };

        size_t kIterations = 1000;

        dopt::ThreadPoolWithTaskQueue<MyTask> pool(add_numer, 10, kIterations);
        EXPECT_TRUE(pool.threadPoolSize() == 10);
        pool.signalSuspendProcessing();
        
        for (size_t i = 1; i <= kIterations; ++i)
            pool.addTask(MyTask(&res, i));

        EXPECT_TRUE(pool.isWaitingTasks());
        EXPECT_TRUE(res == 0);

        pool.signalResumeProcessing();

        EXPECT_TRUE(pool.waitForCurrentJobsCompletion());
        EXPECT_FALSE(pool.isWaitingTasks());
        
        if (0)
        {
            pool.signalToTerminate();
            pool.waitForTermination();
        }

        EXPECT_TRUE(res == (kIterations * (kIterations+1) ) / 2);
    }
    
    // Parallel execution of push and a single pop thread
    {
        std::atomic<int> res = 0;

        struct MyTask
        {
            MyTask(std::atomic<int>* theAccum, int theNumberToAdd)
                : accum(theAccum)
                , number_to_add(theNumberToAdd)
            {}

            MyTask()
                : accum(0)
                , number_to_add(0)
            {}


            std::atomic<int>* accum;
            int number_to_add;
        };

        auto add_numer = [](const MyTask& task, size_t) {
            *(task.accum) += task.number_to_add;
        };

        dopt::ThreadPoolWithTaskQueue<MyTask> pool(add_numer, 1, 50);
        EXPECT_TRUE(pool.threadPoolSize() == 1);
        pool.signalResumeProcessing();

        // Just testsumm for arithmetic series
        size_t kIterations = 1000;
        for (size_t i = 1; i <= kIterations; ++i)
            pool.addTask(MyTask(&res, i));

        EXPECT_TRUE(pool.waitForCurrentJobsCompletion());
        EXPECT_FALSE(pool.isWaitingTasks());

        if (0)
        {
            pool.signalToTerminate();
            pool.waitForTermination();
        }
        
        EXPECT_TRUE(res == (kIterations * (kIterations + 1)) / 2);
    }
}

TEST(dopt, ThreadPoolWithTaskQueueAsyncGTest)
{
    // Parallel execution of push and a multiple pop threads
    {
        std::atomic<int> res = 0;
        struct MyTask
        {
            MyTask(std::atomic<int>* theAccum, int theNumberToAdd)
                : accum(theAccum)
                , number_to_add(theNumberToAdd)
            {}

            MyTask()
                : accum(0)
                , number_to_add(0)
            {}


            std::atomic<int>* accum;
            int number_to_add;
        };
                
        auto add_numer = [](const MyTask& task, size_t) {
            *(task.accum) += task.number_to_add;
        };
        
        dopt::ThreadPoolWithTaskQueue<MyTask> pool(add_numer, 10, 33);
        EXPECT_TRUE(pool.threadPoolSize() == 10);
        pool.signalResumeProcessing();
        
        // Just testsumm for arithmetic series
        size_t kIterations = 1000;
        
        for (size_t i = 1; i <= kIterations; ++i)
            pool.addTask(MyTask(&res, i));

       
        EXPECT_TRUE(pool.waitForCurrentJobsCompletion());
        ASSERT_FALSE(pool.isWaitingTasks());
        ASSERT_TRUE(res == (kIterations * (kIterations + 1)) / 2);
    }
}
