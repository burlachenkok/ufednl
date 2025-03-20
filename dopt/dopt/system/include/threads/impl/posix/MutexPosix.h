/** @file
* Mutex support based on posix pthread library
*/

#pragma once

#if DOPT_LINUX || DOPT_MACOS

#include "dopt/system/include/PlatformSpecificMacroses.h"
#include <pthread.h>

namespace dopt
{
    namespace internal
    {
        /** Mutex impl based on posix implementation. Non-Recursive.
        */
        class MutexPosix
        {
        public:
            MutexPosix();
            ~MutexPosix();

            MutexPosix(const MutexPosix&) = delete;
            MutexPosix(MutexPosix&&) = delete;

            void lock();
            bool tryLock();
            void unlock();

        private:
            pthread_mutex_t locker;
        };
    }
}

#endif
