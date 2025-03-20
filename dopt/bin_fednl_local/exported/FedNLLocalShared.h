#ifndef FEDDNL_LOCAL_SHARED_H
#define FEDDNL_LOCAL_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackLocal)(int xDimension,
                                            double discrWithOptSolution,
                                            double discrWithOptValue,
                                            double L2NormInLastIterate);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteLocal(const char* command, ResultCallbackLocal resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteLocal(const char* command, ResultCallbackLocal resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
