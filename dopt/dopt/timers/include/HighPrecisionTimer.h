#pragma once

#if DOPT_LINUX || DOPT_MACOS
#include "dopt/timers/include/linux/GetTimeOfDayTimer.h"
#elif DOPT_WINDOWS
#include "dopt/timers/include/windows/HiPrecOueryPerfomanceTimer.h"
#endif

namespace dopt
{
#if DOPT_LINUX || DOPT_MACOS
    typedef posix::GetTimeOfDayTimer HighPrecisionTimer;
#elif DOPT_WINDOWS
    typedef windows::HiPrecOueryPerfomanceTimer HighPrecisionTimer;
#endif
}
