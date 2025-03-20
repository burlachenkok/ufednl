#pragma once

    #include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackMaster)(int xDimension);

        SHARED_LIBRARY_EXPORT int doptExecuteMaster(const char* command, ResultCallbackMaster resultCallback);
        
    #ifdef __cplusplus
        }
    #endif

