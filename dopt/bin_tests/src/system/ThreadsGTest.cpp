#include "dopt/system/include/threads/Thread.h"
#include "dopt/system/include/CpuInfo.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <vector>

#include <stdint.h>
#include <math.h>

namespace
{
    int32_t startRoutineReturn123(void* arg, void*) {
        return 123;
    }

    int32_t startRoutineReturnInput(void* arg, void*) {
        dopt::DefaultThread::sleepCurrentTh(50);
        return *(static_cast<int32_t*>(arg));
    }
}

TEST(dopt, ThreadApiGtest)
{
    {
        int32_t returnCode = 0;
        dopt::DefaultThread th1(startRoutineReturn123);
        th1.join();
        EXPECT_TRUE(th1.getExitCode() == 123);

        EXPECT_EQ(th1.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(th1.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(th1.isAlive(), false);
    }

    dopt::DefaultThread::yeildCurrentTh();
    dopt::DefaultThread::yeildCurrentThInHotLoop();
    
    {
        int32_t tmpVar = 9090;

        dopt::DefaultThread th2(startRoutineReturnInput, &tmpVar);
        EXPECT_TRUE(th2.isAlive());
        th2.join(1);
        EXPECT_TRUE(th2.isAlive());
        th2.join();
        EXPECT_EQ(th2.isAlive(), false);
    }

    {
        int32_t returnCode = 0;

        dopt::DefaultThread thDeferred(startRoutineReturn123, 0, 0, 128);
        thDeferred.join();
        EXPECT_EQ(thDeferred.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(thDeferred.isAlive(&returnCode), false);
        EXPECT_EQ(returnCode, 123);
        EXPECT_EQ(thDeferred.isAlive(), false);
    }
    dopt::DefaultThread::sleepCurrentTh(0);
}

TEST(dopt, ThreadApiGtestSleepFor_435_milliseconds)
{
    dopt::HighPrecisionTimer tm;
    dopt::DefaultThread::sleepCurrentTh(435);
    EXPECT_TRUE(::fabs(435.0 - tm.getTimeMs()) < 200);
    // assume that "defaultTimer+sleep" error is less then 200ms (2/10 of second)

    dopt::DefaultThread::yeildCurrentTh();
}

TEST(dopt, ThreadsAffinityGTest)
{
#if DOPT_MACOS
    std::cout << "There are problems with support thread affinity for MacOS" << '\n';
#else
    if (dopt::logicalProcessorsInSystem() < 2)
    {
        std::cout << "In the system there is only processor. Thread Affinity can not be tested.";
    }
    
    dopt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 1);
    EXPECT_TRUE(dopt::DefaultThread::getThreadAffinityMaskForCurrentTh() == (0x1 << 1) );

    dopt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 0);
    EXPECT_TRUE(dopt::DefaultThread::getThreadAffinityMaskForCurrentTh() == 0x1);    
#endif
}

TEST(dopt, ThreadsCheckFlagSetupGTest)
{
    {
        std::atomic<bool> flag = true;
        bool reseted = dopt::checkAndResetIfSet(&flag);
        EXPECT_TRUE(reseted == true);
        EXPECT_TRUE(flag == false);
    }

    {
        std::atomic<bool> flag = false;
        bool reseted = dopt::checkAndResetIfSet(&flag);
        EXPECT_TRUE(reseted == false);
        EXPECT_TRUE(flag == false);
    }
}
