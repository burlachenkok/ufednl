#ifndef FEDDNL_MASTER_SHARED_H
#define FEDDNL_MASTER_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackMaster)(int xDimension);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteMaster(const char* command, ResultCallbackMaster resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteMaster(const char* command, ResultCallbackMaster resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
