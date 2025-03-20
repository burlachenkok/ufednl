/** @file
* Mutex support based on WinApi critical section
*/
#pragma once

#include "dopt/system/include/PlatformSpecificMacroses.h"

#if DOPT_WINDOWS

namespace dopt
{
    namespace internal
    {
        /** Mutex impl based on WinApi implementation. Recursive, no timeout.
        */
        class MutexWinApi
        {
        public:
            MutexWinApi();
            ~MutexWinApi();

            MutexWinApi(const MutexWinApi&) = delete;
            MutexWinApi(MutexWinApi&&) = delete;

            void lock();
            bool tryLock();
            void unlock();

        private:
            CRITICAL_SECTION cs;
        };
    }
}

#endif
