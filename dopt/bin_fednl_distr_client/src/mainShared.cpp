#include "TrainLoop.h"

#include "TrainCallbaksSharedClient.h"

#include "dopt/cmdline/include/CmdLineParser.h"
#include "dopt/fs/include/StringUtils.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <vector>
#include <string>
#include <string_view>

#if BUILD_SHARED_LIBRARY

int doptExecuteClient(const char* command, ResultCallbackClient resultCallback)
{
    std::string commandStr = std::string("train ") + std::string(command);
    std::vector<std::string_view> commandStrViews = dopt::string_utils::splitToSubstrings(commandStr, ' ');

    size_t argc = commandStrViews.size();
    std::vector<char*> argv(argc);

    for (size_t i = 0; i < argc; ++i)
    {
        argv[i] = new char[commandStrViews[i].size() + 1];
        memcpy(argv[i], commandStrViews[i].data(), commandStrViews[i].size());
        argv[i][commandStrViews[i].size()] = '\0';
    }
    
    dopt::CmdLine cmdline(argc, argv.data());
    
    int result = train(cmdline, resultCallback);

    for (size_t i = 0; i < argc; ++i)
        delete[] argv[i];        
    argv.clear();

    return result;
}
#endif
