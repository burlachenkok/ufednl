#ifndef TRAIN_PP_CALLBACK_SHARED_H
#define TRAIN_PP_CALLBACK_SHARED_H

#include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" 
    {
    #endif

    typedef void (*ResultCallbackLocalPP)(int xDimension);

    SHARED_LIBRARY_EXPORT int doptExecuteLocalPP(const char* command, ResultCallbackLocalPP resultCallback);

    #ifdef __cplusplus
    }
    #endif

#endif