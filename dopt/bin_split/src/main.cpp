/** Command line:
SPLIT DATAL
--splitdata --train_dataset C:\projects\new\fednl_impl\dopt\datasets\a1a  --add_intercept --reshuffle_train --clients 15
*/

#include "SplitLoop.h"

#include "dopt/timers/include/HighPrecisionTimer.h"
#include "dopt/system/include/ProcessInfo.h"
#include "dopt/cmdline/include/CmdLineParser.h"

#include <string>

int main(int argc, char** argv)
{
    dopt::HighPrecisionTimer timer_main;
    dopt::CmdLine cmdline(argc, argv);

    if (cmdline.isFlagSetuped("version"))
        dopt::printInformationAboutBuild(argv[0]);

    int result = 0;
    
    if (cmdline.isFlagSetuped("splitdata"))
    {
        result = splitData(cmdline);
    }
    else
    {
        if (!cmdline.isFlagSetuped("version"))
        {
            std::cout << R"(Example Command line: bin_split --splitdata --train_dataset C:\projects\new\fednl_impl\dopt\datasets\a1a  --add_intercept --reshuffle_train --clients 15 --show-stats --version --debug crc32fordata,meminfo)" << '\n';
            result = -1;
            return result;
        }
        else
        {
            // Application has been launched to check version
            result = 0;
        }
    }

    double deltaMs = timer_main.getTimeMs();
    std::cout << '\n';
    std::cout << "Working Directory: " << dopt::FileSystemHelpers::getCwd() << '\n';
    std::cout << "Time spent to execution is " << deltaMs << " milliseconds\n" << '\n';

    return result;
}
