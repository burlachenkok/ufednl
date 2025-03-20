/** @file
* Low level timer to measure latency of the code in terms of instructions.
*/

#pragma once

#include <string>

namespace dopt
{
    /** Returns processor time stamp.
    * The processor time stamp records the number of clock cycles since the last reset.
    * @return number of clocks in processor
    * @remark Convert cycles to real seconds is tricky
    * @reamrk May give different answers on different cores on the same machine.
    * @remark Sometimes timer may run backward
    * @remark x86 processors provide a time-stamp counter (TSC) in hardware. https://c9x.me/x86/html/file_module_x86_id_278.html
    * @remark aarch64 processors has Counter-timer Virtual Count Register
    */
    unsigned long long ReadTSC();

    /** Print custom message which contains information about number of clocks between tStart and tEnd in clocks
    * @param tStart TSC counter at the beginning of measured region
    * @param tEnd TSC counter at the end of measured region
    * @param msg custom message which described clock/time measurement
    * @return string in some unspecified format
    * @remark convert cycles to real seconds is tricky
    */
    std::string PrintMessageTSC(unsigned long long tStart, unsigned long long tEnd, const char* msg);
}
