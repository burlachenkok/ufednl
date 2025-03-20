/** @file
* CRT compatible timer implementation
*/

#pragma once

#include "dopt/timers/include/BaseTimer.h"
#include  <time.h>

namespace dopt
{
    /** Timer with implementation based on CRT function clock().
    * Definition: It is processor time consumed by the program.
    * + approximate time which processor spent to this process
    * - if CPU was sharing between other processes then this timer is moving slowly
    * - if process contains many threads then this timer is moving very fast
    */
    class UsualStdTimer final: public BaseTimer
    {
    public:
        UsualStdTimer()
        : lastTime(clock())
        {
        }

        static double measureTimeResolutionInmilliseconds();

    protected:
        virtual void doSaveCurrentTickStateInPrev();
        virtual double getDelatMsFromLastTickState();

    private:
        clock_t lastTime; ///< Placeholder for ticks
    };
}
