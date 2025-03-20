#ifndef FEDDNL_MASTER_SHARED_H
#define FEDDNL_MASTER_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackMasterPP)(int xDimension);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteMasterPP(const char* command, ResultCallbackMasterPP resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteMasterPP(const char* command, ResultCallbackMasterPP resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
