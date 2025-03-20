/** @file
* Mutex redirection to platform-specific includes
*/

#pragma once

#if DOPT_WINDOWS
    #include "dopt/system/include/threads/impl/windows/SemaphoreWinApi.h"
#elif DOPT_LINUX || DOPT_MACOS
    #include "dopt/system/include/threads/impl/posix/SemaphorePosix.h"
#endif

namespace dopt
{
    #if DOPT_WINDOWS
        typedef dopt::internal::SemaphoreWinApi DefaultSemaphore;
    #elif DOPT_LINUX || DOPT_MACOS
        typedef dopt::internal::SemaphorePosix DefaultSemaphore;
    #endif
}
