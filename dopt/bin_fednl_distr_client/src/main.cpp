/** Command line:
FEDNL:
--algorithm fednl1 --k_compressor_as_d_mult 35.5 --compressor randk --train --train_dataset C:\projects\new\fednl_impl\dopt\datasets\a1a --add_intercept --reshuffle_train --clients 2 --lambda 0.001 --rounds 20 --silent --tracking --global_lr 0.01 --theoretical_alpha --use_theoretical_alpha_option_1 --fednl-option-b --out result.bin --master tcpv4:localhost:1212 --iam server
--algorithm fednl1 --k_compressor_as_d_mult 35.5 --compressor randk --train --client_train_dataset C:\projects\new\fednl_impl\dopt\datasets\a1a --add_intercept --reshuffle_train --clients 10 --lambda 0.001 --rounds 20 --silent --tracking --global_lr 0.01 --theoretical_alpha --use_theoretical_alpha_option_1 --fednl-option-b --out result.bin --master tcpv4:localhost:1212 --iam client:1

GD:
--algorithm gd --k_compressor_as_d_mult 35.5 --compressor randk --train --train_dataset C:\projects\new\fednl_impl\dopt\datasets\phishing --add_intercept --reshuffle_train --clients 10 --lambda 0.001 --rounds 20 --silent --tracking --global_lr 0.01 --theoretical_alpha --use_theoretical_alpha_option_1 --fednl-option-b --out result.bin

SPLIT DATAL
--splitdata --train_dataset C:\projects\new\fednl_impl\dopt\datasets\a1a  --add_intercept --reshuffle_train --clients 15
*/

#include "TrainLoop.h"

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
    
    if (cmdline.isFlagSetuped("train"))
    {
        result = train(cmdline, nullptr);
    }
    else
    {
        if (!cmdline.isFlagSetuped("version"))
        {
            std::cout << "Please specify one of the action [train]\n";
            result = -1;
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
