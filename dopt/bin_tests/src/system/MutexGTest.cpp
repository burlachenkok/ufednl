#include "dopt/system/include/threads/Thread.h"
#include "dopt/system/include/threads/Mutex.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>

#include <stdint.h>

namespace
{
    int32_t startRoutineSleepAndReturn1(void* arg, void*) {
        ((dopt::DefaultMutex*)arg)->lock(); // *lock*

        dopt::DefaultThread::sleepCurrentTh(50);

        ((dopt::DefaultMutex*)arg)->unlock();
        return 1;
    }
}

TEST(dopt, MutexApiGTest)
{
    {
        dopt::DefaultMutex m1;
        m1.lock();

        EXPECT_EQ(m1.tryLock(), true) << "TRY LOCK ALREADY LOCKED MUTEX";
        m1.unlock();

        m1.unlock();

        m1.lock();
        m1.lock();

        EXPECT_EQ(m1.tryLock(), true) << "TRY LOCK ALREADY LOCKED MUTEX MUTIPLE TIMES BY CURRENT THREAD";
        m1.unlock();

        m1.unlock();
        m1.unlock();
    }

    {
        dopt::DefaultMutex m2;
        m2.lock();

        dopt::DefaultThread th1(startRoutineSleepAndReturn1, &m2);
        dopt::DefaultThread::yeildCurrentThInHotLoop();

        // assume that th1 is in *lock* line of code
        // if comment m2.unlock(); then current thread will be hanged I do not know how to check it via google tests
        m2.unlock();
        th1.join();
    }
}
