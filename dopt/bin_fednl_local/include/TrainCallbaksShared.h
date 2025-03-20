#ifndef TRAIN_CALLBACK_SHARED_H
#define TRAIN_CALLBACK_SHARED_H

#include "dopt/system/include/PlatformSpecificMacroses.h"

    #ifdef __cplusplus
    extern "C" 
    {
    #endif

    typedef void (*ResultCallbackLocal)(int xDimension,
                                        double discrWithOptSolution,
                                        double discrWithOptValue,
                                        double L2NormInLastIterate);

    SHARED_LIBRARY_EXPORT int doptExecuteLocal(const char* command, ResultCallbackLocal resultCallback);

    #ifdef __cplusplus
    }
    #endif

#endif