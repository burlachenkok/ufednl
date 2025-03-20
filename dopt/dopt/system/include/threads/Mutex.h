/** @file
* Mutex redirection to platform-specific includes
*/

#pragma once

#if DOPT_WINDOWS
    #include "dopt/system/include/threads/impl/windows/MutexWinApi.h"
#elif DOPT_LINUX || DOPT_MACOS
    #include "dopt/system/include/threads/impl/posix/MutexPosix.h"
#endif

namespace dopt
{
    #if DOPT_WINDOWS
        typedef dopt::internal::MutexWinApi DefaultMutex;
    #elif DOPT_LINUX || DOPT_MACOS
        typedef dopt::internal::MutexPosix DefaultMutex;
    #endif
}
