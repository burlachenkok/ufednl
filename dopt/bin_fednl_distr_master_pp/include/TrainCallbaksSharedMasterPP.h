#pragma once

    #include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackMasterPP)(int xDimension);

        SHARED_LIBRARY_EXPORT int doptExecuteMasterPP(const char* command, ResultCallbackMasterPP resultCallback);
        
    #ifdef __cplusplus
        }
    #endif

