#ifndef FEDDNL_PP_LOCAL_SHARED_H
#define FEDDNL_PP_LOCAL_SHARED_H

    #ifdef __cplusplus
    extern "C" {
    #endif

        typedef void (*ResultCallbackLocalPP)(int xDimension);

        #ifdef _WIN32
        __declspec(dllimport) int doptExecuteLocalPP(const char* command, ResultCallbackLocalPP resultCallback);
        #else
        /* empty import prefix for GCC/Clang*/ 
                              int doptExecuteLocalPP(const char* command, ResultCallbackLocalPP resultCallback);
        #endif

    #ifdef __cplusplus
        }
    #endif

#endif
