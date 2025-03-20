#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "ClockTimer.h"

#include <sstream>

#if DOPT_ARCH_X86_64BIT
    #if DOPT_WINDOWS
        #include <intrin.h>
    #elif DOPT_LINUX || DOPT_MACOS
        #include <x86intrin.h>

        // infoType => EAX
        // out[0] gets the value from eax.
        // out[1] gets the value from ebx.
        // out[2] gets the value from ecx.
        // out[3] gets the value from edx.

        #define __cpuid(out, infoType)\
            asm volatile("cpuid"\
            : "=a" (out[0]),\
            "=b" (out[1]),\
            "=c" (out[2]),\
            "=d" (out[3]): "a" (infoType));
    #endif

    namespace dopt
    {
        unsigned long long ReadTSC()
        {
            int dummy[4];          // For unused returns
            volatile int dontSkip; // Volatile to prevent optimizing
            __cpuid(dummy, 0);     // Serializing the instruction stream.  CPUID call implements a barrier to avoid out-of-order execution of the instructions above
            dontSkip = dummy[0];   // Prevent optimizing away cpuid

            const unsigned long long clock = __rdtsc();   // Read time
            return clock;
        }
    }
#elif  DOPT_ARCH_ARM
    namespace dopt
    {
        unsigned long long ReadTSC()
        {
            // Accessing the PSTATE register
            //   mrs -- Move from system register https://developer.arm.com/documentation/dui0802/a/A64-General-Instructions/MRS
            //   cntvct_el0 -- Counter-timer Virtual Count Register https://developer.arm.com/documentation/ddi0601/2024-06/AArch64-Registers/CNTVCT-EL0--Counter-timer-Virtual-Count-Register

            uint64_t cntvct;
            asm volatile("mrs %0, cntvct_el0" : "=r" (cntvct));
            return cntvct;
        }
    }
#else
    namespace dopt
    {
        unsigned long long ReadTSC()
        {
            // Dummy implementation
            return 0;
        }
    }
#endif

namespace dopt
{
    std::string PrintMessageTSC(unsigned long long tStart, unsigned long long tEnd, const char* msg)
    {
        std::stringstream str;

        if (tEnd <= tStart)
        {
            str << "some error in computing time-stamp-counters. start: " << tStart << ", end: " << tEnd;
        }

        str << "number of clocks for '" << msg << "' is: " << tEnd - tStart;
        return str.str();
    }
}
