#pragma once

//                Platform specific macroses
#include "dopt/system/include/PlatformSpecificMacroses.h"

//                      General utilities
#include "dopt/cmdline/include/CmdLineParser.h"
#include "dopt/timers/include/HighPrecisionTimer.h"
#include "dopt/fs/include/StringUtils.h"
#include "dopt/fs/include/FileSystemHelpers.h"
#include "dopt/fs/include/FileNameHelpers.h"

//             Optimization/Training process arguments
#include "Args.h"

//               Random number generators, reshfullers
#include "dopt/random/include/Shuffle.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"

//                     Print and report utilities
#include "Reports.h"

//                            C++/C
#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>

inline TrainReturnCodes splitData(const dopt::CmdLine& cmdline)
{
    dopt::HighPrecisionTimer timer;

    // Parse Debug Flags and Information about dataset
    DebugCfg debugFlags;
    TrainReturnCodes resultParseFlagsForDebug = parseFlagsForDebug(debugFlags, cmdline);
    if (resultParseFlagsForDebug != TrainReturnCodes::eOk)
        return resultParseFlagsForDebug;

    DatasetCfg datasetInfo;
    TrainReturnCodes resultParseDataset = parseFlagsForTrainDataset(datasetInfo, cmdline);
    if (resultParseDataset != TrainReturnCodes::eOk)
        return resultParseDataset;

    //========================================================================================
    // Parse input data: Different separators
    auto theLineSeparator = [](char c) constexpr const_func_ext
    {return c == '\n' || c == '\r'; };

    auto theFieldSeparatorInLine = [](char c) constexpr const_func_ext
    {return c == '\t' || c == ' '; };

    auto theAttributeValueSeparator = [](char c) constexpr const_func_ext
    {return c == ':'; };

    auto theSymbolSkipping = [](char c) constexpr const_func_ext
    {return false; };

    // There is nothing to skip specifically for dataset during parsing
    constexpr bool kOmitSymbolSkipCheck = true;
    //========================================================================================
    // Parse input data: Actual Parsing
    dopt::FileSystemHelpers::FileMappingResult trainDs = dopt::FileSystemHelpers::mapFileToMemory(datasetInfo.path.c_str(), true);

    if (!trainDs.isOk)
    {
        std::cout << "Train dataset: '" << datasetInfo.path << "'. [Error during loading] " << timer.timeStamp() << '\n';
        std::cout << "Error message: '" << trainDs.errorMsg << '\n';
        return TrainReturnCodes::eTrainDatasetErrorInParsing;
    }

    dopt::Data trainDatasetRaw(trainDs.memory, trainDs.memorySizeInBytes, dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
    dopt::TextFileDataSetLoader dataset_loader(trainDatasetRaw, theLineSeparator, theFieldSeparatorInLine, theAttributeValueSeparator, theSymbolSkipping);
    dopt::DataSetLoaderStats<IndexType, LabelType> stats = dataset_loader.computeStatisticsWithOneDataPass<false /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/, IndexType, LabelType>();

    if (!stats.ok)
    {
        std::cout << "Train dataset: '" << datasetInfo.path << "'. [Error during parsing] " << timer.timeStamp() << '\n';
        return TrainReturnCodes::eTrainDatasetErrorInParsing;
    }
    else
    {
        std::cout << "Train dataset: '" << datasetInfo.path << "' is parsed." << timer.timeStamp() << '\n';
    }

    DatasetType trainset;
    int dataset_loading_flags = (datasetInfo.add_intercept == false ? dopt::CreateDatasetFlags::eNone : dopt::CreateDatasetFlags::eAddInterceptTerm);
    size_t processed_train_examples = dataset_loader.createDataset<true /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/ >(trainset, stats, dataset_loading_flags | dopt::CreateDatasetFlags::eMakeRemappingForBinaryLogisticLoss);
    
    trainDatasetRaw.clear();                                                         // Clear the memory pointer
    bool unmapInputFile = dopt::FileSystemHelpers::unmapFileFromMemory(trainDs);     // Perform unmapping
    assert(unmapInputFile == true);

    std::cout << "Train dataset: '" << datasetInfo.path << "' was instantiated [OK]. " << timer.timeStamp() << '\n';
    printInformationAboutDataset("train dataset", trainset, debugFlags.debugCrc32ForDatasets, debugFlags.debugPrintTrainDataset);
    int nClients = datasetInfo.nClients;

    size_t totalSamples = trainset.totalSamples();
    size_t totalSamplesPerClient = totalSamples / nClients;
    int skipped_samples = 0;

    // Index for all datapoints
    VectorND<IndexType> examples = VectorND<IndexType>::sequence<(IndexType)0>(totalSamples);

    // Indicies per client
    std::vector<std::vector<IndexType>> examplesPerClient = std::vector<std::vector<IndexType>>(nClients);
    std::vector<size_t> posExamplesPerClient = std::vector<size_t>(nClients);
    std::vector<size_t> negExamplesPerClient = std::vector<size_t>(nClients);

    // Indicies for skipped samples
    std::vector<IndexType> skippedExamples;
    skippedExamples.reserve(totalSamples - totalSamplesPerClient * nClients);

    {
        if (datasetInfo.reshuffle)
        {
            dopt::RandomGenIntegerLinear gen4shuffler;
            gen4shuffler.setSeed(datasetInfo.reshuffle_seed);
            dopt::shuffle(examples, gen4shuffler);
        }

        // strategy: drop residual samples
        int current_sample = 0;
        for (int c = 0; c < nClients; ++c)
        {
            examplesPerClient[c].reserve(totalSamplesPerClient);

            for (int s = 0; s < totalSamplesPerClient; ++s, ++current_sample)
            {
                examplesPerClient[c].push_back(examples[current_sample]);
            }
        }

        for (; current_sample < totalSamples; ++current_sample)
            skippedExamples.push_back(examples[current_sample]);
    }

    std::vector<DatasetType> datasets_per_clients = trainset.splitDataset(examplesPerClient);
    
    for (int c = 0; c < nClients; ++c)
    {
        size_t posSamples = 0;
        size_t negSamples = 0;

        for (size_t i = 0; i < datasets_per_clients[c].train_outputs.size(); ++i)
        {
            if (datasets_per_clients[c].train_outputs.get(i) > 0)
            {
                posSamples++;
            }
            else
            {
                negSamples++;
            }
        }

        posExamplesPerClient[c] = posSamples;
        negExamplesPerClient[c] = negSamples;
    }

    std::cout << "  Train samples has been reshuffled: " << (datasetInfo.reshuffle ? "[YES]" : "[NO]") << '\n';
    std::cout << "  Train samples reshuffled seed: " << (datasetInfo.reshuffle_seed) << '\n';
    std::cout << "  Number of clients: " << datasetInfo.nClients << '\n';
    std::cout << "  Dataset path: " << datasetInfo.path << '\n';
    std::cout << "  Total samples in dataset: " << totalSamples << '\n';
    std::cout << "  Number of samples per client: " << totalSamplesPerClient << '\n';
    std::cout << "  Number of skipped samples during samples distribution: " << skippedExamples.size() << '\n';
    std::cout << "  Train data distribution across clients has been completed. [OK] " << timer.timeStamp() << '\n';
    std::cout << "=========================================================================\n";

    if (cmdline.isFlagSetuped("check_split"))
    {
        if (!trainset.checkSplitConsistency(datasets_per_clients, examplesPerClient, skippedExamples, trainset))
        {
            std::cout << "  Checking data split [FAILED] " << timer.timeStamp() << '\n';
            return TrainReturnCodes::eDataSplitError;
        }
        else
        {
            std::cout << "  Checking data split [OK] " << timer.timeStamp() << '\n';
        }
    }
    else
    {
        std::cout << "  Checking data split [SKIP] \n";
    }
    
    timer.reset();

    std::cout << "=========================================================================\n";

    //
    bool showStats = cmdline.isFlagSetuped("show-stats");
    
    // Serialize all datasets into filesystem
    for (size_t i = 0; i < datasets_per_clients.size(); ++i)
    {
        std::string dataset_path = datasetInfo.path + "_client_" + dopt::string_utils::toString(i) + ".csv";
        std::ofstream myfile;
        
        myfile.open(dataset_path);
        if (!myfile.good()) { 
            std::cout << "  Serializing: '" << dataset_path << "' for client " << std::setw(2) << i  << " has problems with this file [ERROR]\n";
        } else { 
            datasets_per_clients[i].printInTextFormat(myfile);
            myfile.close();
            std::cout << "  Serializing: '" << dataset_path << "' for client " << std::setw(2) << i << " has been done [OK] ";
            
            size_t totalSamplesPerClients = posExamplesPerClient[i] + negExamplesPerClient[i];
            assert(totalSamplesPerClients == datasets_per_clients[i].train_outputs.size());
            if (showStats)
            {
                uint64_t fileSizeInBytes = dopt::FileSystemHelpers::getFileSize(dataset_path);
                
                std::cout << "[POS.SAMPLES:"
                    << posExamplesPerClient[i] << "(" << dopt::roundToNearestInt(100.0 * posExamplesPerClient[i] / (double)totalSamplesPerClients) << "%)"
                    << ","
                    << "NEG.SAMPLES:" << negExamplesPerClient[i] << "(" << dopt::roundToNearestInt(100.0 * negExamplesPerClient[i] / (double)totalSamplesPerClients) << "%)"
                    << ",ALL.SAMPLES:" << totalSamplesPerClients << ",FILE SIZE: " << dopt::roundToNearestInt(fileSizeInBytes / 1024.0) << " KBytes"
                    << "]";
            }
            
            std::cout << '\n';
            
        }
    }
    std::cout << "=========================================================================\n";
    std::cout << '\n';

    if (debugFlags.debugMemInfo)
        printMemoryInformation();

    std::cout << "  Splitted datasets was serialized into filesystem [OK] " << timer.timeStamp() << '\n';

    return TrainReturnCodes::eOk;
}
