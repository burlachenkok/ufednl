#include "UsualStdTimer.h"

namespace dopt
{
    double UsualStdTimer::measureTimeResolutionInmilliseconds()
    {
        UsualStdTimer tmp;
        return measureTimeResolutionHelper(tmp);
    }

    void UsualStdTimer::doSaveCurrentTickStateInPrev()
    {
        lastTime = clock();
    }

    double UsualStdTimer::getDelatMsFromLastTickState()
    {
        const clock_t curTime = clock();
        const double delta = ( double(curTime) - double(lastTime) ) / double(CLOCKS_PER_SEC);

        return delta * 1000.0;
    }
}
