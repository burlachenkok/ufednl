#pragma once

    #include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackClientPP)(int xDimension);

        SHARED_LIBRARY_EXPORT int doptExecuteClientPP(const char* command, ResultCallbackClientPP resultCallback);
        
    #ifdef __cplusplus
        }
    #endif

