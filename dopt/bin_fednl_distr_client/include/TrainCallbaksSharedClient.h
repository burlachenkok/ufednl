#pragma once

    #include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackClient)(int xDimension);

        SHARED_LIBRARY_EXPORT int doptExecuteClient(const char* command, ResultCallbackClient resultCallback);
        
    #ifdef __cplusplus
        }
    #endif

