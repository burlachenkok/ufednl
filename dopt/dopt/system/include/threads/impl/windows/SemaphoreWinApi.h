/** @file
* Semaphore, kernel-object
*/

#pragma once

#include "dopt/system/include/PlatformSpecificMacroses.h"

#if DOPT_WINDOWS

namespace dopt
{
    namespace internal
    {
        /** Semaphore impl based on WinApi implementation.
        */
        class SemaphoreWinApi
        {
        public:
            SemaphoreWinApi();
            
            SemaphoreWinApi(uint32_t theInitialCount);

            SemaphoreWinApi(const SemaphoreWinApi& rhs);

            SemaphoreWinApi(SemaphoreWinApi&& rhs) noexcept;
            
            SemaphoreWinApi& operator = (SemaphoreWinApi&& rhs) noexcept;

            ~SemaphoreWinApi();

            void acquire();

            bool tryAcquire();
            
            void release(int32_t count);

        private:
            HANDLE semaphore;
        };
    }
}

#endif
