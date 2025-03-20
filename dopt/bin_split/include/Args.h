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
#include <stdint.h>

//===========================================================================================================
// Used index type for data set

typedef int32_t IndexType; /// Used index type for data set
typedef int32_t LabelType; /// Used label type for data set

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
};

inline TrainReturnCodes parseFlagsForDebug(DebugCfg& debugFlags, const dopt::CmdLine& cmdline)
{
    std::string debug_flags_str;

    if (cmdline.getStringArgByName(debug_flags_str, "debug"))
    {
        auto theFieldsDelimiter = [](int c) {
            return c == ',' || c == ';';
        };

        std::vector<std::string_view> modes = dopt::string_utils::splitToSubstrings(debug_flags_str, theFieldsDelimiter);

        for (const std::string_view& mode : modes)
        {
            if (mode == "meminfo")
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
    int nClients = 1;                ///< Number of clients in data set.
    bool reshuffle = false;          ///< Reshuffle data in uniformaly at random
    uint32_t reshuffle_seed = 123;   ///< Reshuffle train data seed.
    bool add_intercept = false;      ///< Flag which specify adding intercept term into a last column of a design matrix in runtime.
    std::string path;                ///< Path to data set in Tab Separated Format in format: [label] ([feature_index:value][ ])*
};

inline TrainReturnCodes parseFlagsForTrainDataset(DatasetCfg& datasetCfg, const dopt::CmdLine& cmdline)
{
    cmdline.getIntArgByName(datasetCfg.nClients, "clients");
    datasetCfg.reshuffle = cmdline.isFlagSetuped("reshuffle_train");
    cmdline.getUnsignedArgByName(datasetCfg.reshuffle_seed, "reshuffle_train_seed");
    datasetCfg.add_intercept = cmdline.isFlagSetuped("add_intercept");

    if (!cmdline.getStringArgByName(datasetCfg.path, "train_dataset"))
    {
        std::cerr << "Please specify argument \"" << "train_dataset" << " with path to train dataset\n";
        return TrainReturnCodes::eMissingArgument;
    }
    else
    {
        if (dopt::FileSystemHelpers::isFileExist(datasetCfg.path))
        {
            std::cout << "Used train dataset: '" << datasetCfg.path << "'. File exist: [OK]\n";
        }
        else
        {
            std::cout << "Used train dataset: '" << datasetCfg.path << "'. File exist: [NOT]\n";
            return TrainReturnCodes::eTrainDatasetIsNotAvailable;
        }
    }

    return TrainReturnCodes::eOk;
}

