#ifndef FEDDNL_CLIENT_SHARED_H
#define FEDDNL_CLIENT_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackClientPP)(int xDimension);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteClientPP(const char* command, ResultCallbackClientPP resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteClientPP(const char* command, ResultCallbackClientPP resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
