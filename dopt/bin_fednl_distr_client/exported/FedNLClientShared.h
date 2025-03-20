#ifndef FEDDNL_CLIENT_SHARED_H
#define FEDDNL_CLIENT_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackClient)(int xDimension);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteClient(const char* command, ResultCallbackClient resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteClient(const char* command, ResultCallbackClient resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
