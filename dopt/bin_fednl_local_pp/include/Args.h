#pragma once

// Numerical math includes
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"

// Command Line parser
#include "dopt/cmdline/include/CmdLineParser.h"

// ML relative
#include "dopt/optimization_problems/include/ml/dataset_loader_fs.h"

// Standart header files
#include <iostream>
#include <stddef.h>
#include <limits.h>
#include <stdint.h>

//===========================================================================================================
// Used index type for data set

typedef int32_t IndexType; /// Used index type for data set
typedef int32_t LabelType; /// Used label type for data set

// DEBUG FLAGS START
static constexpr bool kTrackHessianDifference = DOPT_EXTRA_DEBUG;     ///< For Debug Purposes only to track hessian differences
static constexpr bool kDebugOutputForSolve = DOPT_EXTRA_DEBUG;        ///< DEBUG OUTPUT FOR SOLVE OF SYSTEM OF LINEAR EQUATIONS STEP
// DEBUG FLAGS END

#if 0
    template <class T>
    using VectorND = dopt::VectorNDStd<T>;           // Used implementation for Vec

    typedef dopt::VectorNDStd_d VectorND_d;          // Used implementation for Vec fp64
    typedef dopt::VectorNDStd_f VectorND_f;          // Used implementation for Vec fp32
    typedef dopt::VectorNDStd_i VectorND_i;          // Used implementation for Vec int_32
    typedef dopt::VectorNDStd_b VectorND_b;          // Used implementation for Vec int_32
#else
    template <class T>
    using VectorND = dopt::VectorNDRaw<T>;           // Used implementation for Vec

    typedef dopt::VectorNDRaw_d VectorND_d;          // Used implementation for Vec fp64
    typedef dopt::VectorNDRaw_f VectorND_f;          // Used implementation for Vec fp32
    typedef dopt::VectorNDRaw_i VectorND_i;          // Used implementation for Vec int_32
    typedef dopt::VectorNDRaw_b VectorND_b;          // Used implementation for Vec int_32
#endif

// Used type for design matrix
typedef dopt::MatrixNMD<VectorND_d> MatrixNMD_d;

// Used type for dataset
typedef dopt::DataSet<MatrixNMD_d, VectorND_d, VectorND_b, VectorND_d> DatasetType;

/** General return codes from training process
*/
enum TrainReturnCodes : std::int8_t
{
    eOk = 0,
    eMissingArgument = -1,
    eWrongArgument = -2,
    eTrainDatasetIsNotAvailable = -3,
    eTrainDatasetErrorInParsing = -4,
    eDataSplitError = -5,
    eResultSavingHasBeenFailed = -6,
    eInternalError = -7,
    eUnknownDebugMode = -8
};
//===========================================================================================================

/** Debug Purposes Flags.
* This flag should be passed in format : --debug "meminfo,forcesequpdate,crc32fordata,printtrain"
*/
struct DebugCfg
{
    bool debugMemInfo = false;                    ///< Print information about memory. Should be specified via --debug "meminfo".
    bool debugForceSequentialUpdate = false;      ///< Server obtainining updates from clients in order 1,...,n. Should be specified via --debug "forcesequpdate".
    bool debugCrc32ForDatasets = false;           ///< Print information about CRC32 for datasets loaded for clients. Should be specified via --debug "crc32fordata".
    bool debugPrintTrainDataset = false;          ///< Print train dataset for debug purposes.
    bool debugSilentPrintingInTraining = false;   ///< Do not print information during training
};

inline TrainReturnCodes parseFlagsForDebug(DebugCfg& debugFlags, 
                                           const dopt::CmdLine& cmdline)
{
    std::string_view debug_flags_str;

    if (cmdline.getStringViewArgByName(debug_flags_str, "debug"))
    {
        std::vector<std::string_view> modes;

        auto theFieldsDelimiter = [](int c) {
            return c == ',' || c == ';';
        };

        dopt::string_utils::splitToSubstrings(modes, debug_flags_str, theFieldsDelimiter);

        for (const std::string_view& mode : modes)
        {
            if (mode == "notracingprint")
            {
                debugFlags.debugSilentPrintingInTraining = true;
            }
            else if (mode == "meminfo")
            {
                debugFlags.debugMemInfo = true;
            }
            else if (mode == "forcesequpdate")
            {
                debugFlags.debugForceSequentialUpdate = true;
            }
            else if (mode == "crc32fordata")
            {
                debugFlags.debugCrc32ForDatasets = true;
            }
            else if (mode == "printtrain")
            {
                debugFlags.debugPrintTrainDataset = true;
            }
            else
            {
                std::cerr << "Unknown debug mode: " << mode << '\n';
                return TrainReturnCodes::eUnknownDebugMode;
            }
        }
    }

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Dataset charactersitics
*/
struct DatasetCfg
{
    int nClients = 1;                ///< Number of clients in data set. Clients correpond to really independent devices.
    bool reshuffle = false;          ///< Reshuffle data in uniformly at random.
    uint32_t reshuffle_seed = 123;   ///< Reshuffle train data seed.
    bool add_intercept = false;      ///< Flag which specify adding intercept term into a last column of a design matrix in runtime.

    std::string path;                          ///< Path to data set in Tab Separated Format in format: [label] ([feature_index:value][ ])*
};

inline TrainReturnCodes parseFlagsForTrainDataset(DatasetCfg& datasetCfg, const dopt::CmdLine& cmdline)
{
    cmdline.getIntArgByName(datasetCfg.nClients, "clients");
    datasetCfg.reshuffle = cmdline.isFlagSetuped("reshuffle_train");
    cmdline.getUnsignedArgByName(datasetCfg.reshuffle_seed, "reshuffle_train_seed");
    datasetCfg.add_intercept = cmdline.isFlagSetuped("add_intercept");

    if (cmdline.getStringArgByName(datasetCfg.path, "train_dataset"))
    {
        if (dopt::FileSystemHelpers::isFileExist(datasetCfg.path))
        {
            // std::cout << "Used train dataset: '" << datasetCfg.path << "'. File exist: [OK]\n";
        }
        else
        {
            std::cerr << "Used train dataset: '" << datasetCfg.path << "'. File exist: [NOT]\n";
            return TrainReturnCodes::eTrainDatasetIsNotAvailable;
        }

    }
    else
    {
        std::cerr << "Please specify argument \"" << "train_dataset" << "\" with path to train dataset\n";
        return TrainReturnCodes::eMissingArgument;
    }

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Tracking information
*/
struct TrackingCfg
{
    bool tracking_is_on = false;              ///< Tracking is ON. This information is needed for plots in paper, but it's not needed in principle for launching Algorithm itself.
    std::string_view output_binary;           ///< Path to output file with tracking information and to final results
};

inline TrainReturnCodes parseFlagsForTracking(TrackingCfg& tracking, 
                                              const dopt::CmdLine& cmdline)
{
    tracking.tracking_is_on = cmdline.isFlagSetuped("tracking");

    if (tracking.tracking_is_on)
    {
        // std::cout << "Tracking information [OK]\n";
    }
    else
    {
        std::cout << "Tracking information [IGNORE]\n";
    }

    if (!cmdline.getStringViewArgByName(tracking.output_binary, "out"))
    {
        std::cerr << "You did not specify argument \"" << "out" << " with path to results which optionally has tracking information\n";
        return TrainReturnCodes::eMissingArgument;
    }

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Optimization Problem characterstics
*/
struct OptProblemCfg
{
    MatrixNMD_d::TElementType lambda;         ///< Lambda for L2 regularized Logistic Regression
};

inline TrainReturnCodes parseFlagsForOptProblem(OptProblemCfg& optProblemCfg, const dopt::CmdLine& cmdline)
{
    double lambda_ = 0.0;
    if (!cmdline.getDoubleArgByName(lambda_, "lambda"))
    {
        std::cerr << "Please specify argument \"" << "lambda" << "\" with value of L2 regulirization for logistic regression\n";
        return TrainReturnCodes::eMissingArgument;
    }
    optProblemCfg.lambda = MatrixNMD_d::TElementType(lambda_);

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Runtime charactersitics
*/
struct RuntimeCfg
{
    dopt::RandomGenRealLinear x0_rnd_generator = dopt::RandomGenRealLinear(0.0, 0.0); ///< Pseudo Random generator(PRG) for initial point "x0"

    uint32_t x0_seed = 1234;                                                          ///< Seed for setting x0
    int nWorkers = 1;                                                                 ///< Number of workers to process data
    int rounds = 0;                                                                   ///< Number of optimization rounds
    int nServerGradWorkers = 0;                                                       ///< Number of workers to process data on server side [for gradient updates]
    int nServerHessianWorkers = 0;                                                    ///< Number of workers to process data on server side [for hessian updates]

    int nClientsPerRound = -1;                                                        ///< Number of clients per round for FedNL-PP
    uint32_t clientSelectionSeed = 1234;                                              ///< Seed for client selection
};

inline TrainReturnCodes parseFlagsForRuntime(RuntimeCfg& runtimeFlags, const dopt::CmdLine& cmdline)
{
    cmdline.getIntArgByName(runtimeFlags.nServerGradWorkers, "server_grad_workers");

    cmdline.getIntArgByName(runtimeFlags.nServerHessianWorkers, "server_hessian_workers");

    cmdline.getIntArgByName(runtimeFlags.nWorkers, "workers");

    if (!cmdline.getIntArgByName(runtimeFlags.rounds, "rounds"))
    {
        std::cerr << "Please specify argument \"" << "rounds" << "\" with number of rounds\n";
        return TrainReturnCodes::eMissingArgument;
    }

    cmdline.getUnsignedArgByName(runtimeFlags.x0_seed, "x0_seed");

    cmdline.getIntArgByName(runtimeFlags.nClientsPerRound, "clients-per-round");

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Optimization Algorithm characterstics
*/
struct OptAlgorithmCfg
{
    std::string algorithm;                    ///< Name of Optimization Algorithm

    // Compressor
    //============================================================================================
    double k_compressor_as_d_mult = 1;         ///< k for FedNL compressor as multiplier of D    

    std::string compressor;                    ///< Type of compressor
    //============================================================================================

    // Step Sizes for GD (Glboal step size is not used currently in FedNL)
    //============================================================================================
    double global_step_size = 0.0;               ///< Computed Global step size.

    bool has_theoretical_global_lr_flag = false; ///< Flag for using global step size

    double global_step_size_multiplier = 1.0;    ///< Global step size multiplier. This flag is used for impoving convergence speed not according to Theory.
    //============================================================================================

    // Alpha Step Size for FedNL (Local step size currently is not used in GD)
    //============================================================================================
    double alpha_step_size = 0.0;                  ///< Alpha step size.

    bool has_theoretical_alpha_flag = false;       ///< Flag to use theoretical alpha.

    bool use_theoretical_alpha_option_1 = false;   ///< Flag to use theoretical alpha option 1 (for contractive compressors)

    bool use_theoretical_alpha_option_2 = false;   ///< Flag to use theoretical alpha option 2 (for contractive compressors)
    //============================================================================================
};

inline TrainReturnCodes parseFlagsForOptAlgortihm(OptAlgorithmCfg& optAlgoCfg, const dopt::CmdLine& cmdline)
{
    if (!cmdline.getStringArgByName(optAlgoCfg.algorithm, "algorithm"))
    {
        std::cerr << "Please specify argument \"" << "algorithm" << "\" with name of algorithm\n";
        return TrainReturnCodes::eMissingArgument;
    }

    if (optAlgoCfg.algorithm != "gd")
    {
        // FedNL relative settings
        if (!cmdline.getDoubleArgByName(optAlgoCfg.k_compressor_as_d_mult, "k_compressor_as_d_mult"))
        {
            std::cerr << "Please specify argument \"" << "k_compressor_as_d_mult" << "\" with value of multiplier for K=<mult>D for Rand-K\n";
            return TrainReturnCodes::eMissingArgument;
        }

        if (!cmdline.getStringArgByName(optAlgoCfg.compressor, "compressor"))
        {
            std::cerr << "Please specify argument \"" << "compressor" << "\" with value [identical,randk,topk,seqk]\n";
            return TrainReturnCodes::eMissingArgument;
        }

        optAlgoCfg.has_theoretical_alpha_flag = cmdline.isFlagSetuped("theoretical_alpha");

        optAlgoCfg.use_theoretical_alpha_option_1 = cmdline.isFlagSetuped("use_theoretical_alpha_option_1");

        optAlgoCfg.use_theoretical_alpha_option_2 = cmdline.isFlagSetuped("use_theoretical_alpha_option_2");

        if (!optAlgoCfg.has_theoretical_alpha_flag)
        {
            if (!cmdline.getDoubleArgByName(optAlgoCfg.alpha_step_size, "alpha_step_size"))
            {
                std::cerr << "Please specify argument \"" << "alpha_step_size" << "\" with value of step size for FedNL\n";
                return TrainReturnCodes::eMissingArgument;
            }
        }
    }
    else
    {
        // GD relative settings
        optAlgoCfg.has_theoretical_global_lr_flag = cmdline.isFlagSetuped("theoretical_global_lr");

        if (!cmdline.getDoubleArgByName(optAlgoCfg.global_step_size, "global_lr") && !optAlgoCfg.has_theoretical_global_lr_flag)
        {
            std::cerr << "Please specify argument \"" << "global_lr" << "\" with value of global_lr\n";
            return TrainReturnCodes::eMissingArgument;
        }

        cmdline.getDoubleArgByName(optAlgoCfg.global_step_size_multiplier, "global_step_size_multiplier");
    }

    return TrainReturnCodes::eOk;
}

//===========================================================================================================

/** Training Arguments
*/
struct TrainingProcessCfg
{    
    TrackingCfg tracking;                        ///< Tracking optimization process information
    DatasetCfg train_dataset;                    ///< Train dataset
    OptProblemCfg optPrb;                        ///< Optimization Problem
    OptAlgorithmCfg optAlgo;                     ///< Optimization Algorithms relative    
    RuntimeCfg runtime;                          ///< Runtime configuration
};

inline TrainReturnCodes parseFlagsForTrain(TrainingProcessCfg& args, 
                                           const dopt::CmdLine& cmdline, 
                                           bool skipTrainDataSetParsing)
{
    if (TrainReturnCodes retCode = parseFlagsForTracking(args.tracking, cmdline); retCode != TrainReturnCodes::eOk) {
        std::cerr << "Problems during parsing 'tracking' information\n";
        return retCode;
    }

    if (!skipTrainDataSetParsing)
    {
        if (TrainReturnCodes retCode = parseFlagsForTrainDataset(args.train_dataset, cmdline); retCode != TrainReturnCodes::eOk) {
            std::cerr << "Problems during parsing 'train dataset' information\n";
            return retCode;
        }
    }

    if (TrainReturnCodes retCode = parseFlagsForOptProblem(args.optPrb, cmdline); retCode != TrainReturnCodes::eOk) {
        std::cerr << "Problems during parsing 'optimization problem' information\n";
        return retCode;
    }

    if (TrainReturnCodes retCode = parseFlagsForOptAlgortihm(args.optAlgo, cmdline); retCode != TrainReturnCodes::eOk) {
        std::cerr << "Problems during parsing 'optimization algorithm' information\n";
        return retCode;
    }

    if (TrainReturnCodes retCode = parseFlagsForRuntime(args.runtime, cmdline); retCode != TrainReturnCodes::eOk) {
        std::cerr << "Problems during parsing 'runtime' information\n";
        return retCode;
    }
    
    return TrainReturnCodes::eOk;
}
//===========================================================================================================
