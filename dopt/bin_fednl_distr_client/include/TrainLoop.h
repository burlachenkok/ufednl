#pragma once

// Platform specific macroses
#include "dopt/system/include/PlatformSpecificMacroses.h"

// General utilities
#include "dopt/cmdline/include/CmdLineParser.h"
#include "dopt/timers/include/HighPrecisionTimer.h"
#include "dopt/fs/include/FileSystemHelpers.h"
#include "dopt/fs/include/FileNameHelpers.h"
#include "dopt/fs/include/StringUtils.h"

// Data manipulation
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/Data.h"

// Optimization/Training process arguments
#include "UsedTypes.h"
#include "dopt/optimization_problems/include/ml/logistic_regression.h"

// Workers
#include "LocalWorkerForFedNL_Identical.h"
#include "LocalWorkerForFedNL_RandK.h"
#include "LocalWorkerForFedNL_RandSeqK.h"
#include "LocalWorkerForFedNL_TopK.h"
#include "LocalWorkerForFedNL_TopLEK.h"
#include "LocalWorkerForFedNL_Natural.h"

#include "LocalWorkerForGD.h"

// Numerical math
#include "dopt/linalg_matrices/include/factorization/CholeskyFactorization.h"
#include "dopt/linalg_linsolvers/include/GaussEliminationSolvers.h"
#include "dopt/linalg_linsolvers/include/ElementarySolvers.h"

// Parallel work organization
#include "dopt/system/include/threads/Thread.h"

// Digests and control sums
#include "dopt/system/include/digest/Crc.h"

// Random number generators, reshfullers
#include "dopt/random/include/Shuffle.h"
#include "dopt/random/include/RandomGenCrt.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"

#include "dopt/linalg_vectors/include/LightVectorND.h"

// Print and report utilities
#include "Reports.h"

// System
#include "dopt/system/include/threads/Thread.h"
#include "dopt/system/include/threads/ThreadPoolWithTaskQueue.h"
#include "dopt/system/include/CpuInfo.h"

// Numerics
#include "dopt/math_routines/include/SimpleMathRoutines.h"

// Callbacks
#include "TrainCallbaksSharedClient.h"

#include <unordered_map>
#include <map>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

constexpr size_t kTotalConnectionsPerClient = 1;

inline int32_t connectionToMaster(void* arg1, void* arg2)
{
    const NetworkCfg& networkCfg = *(reinterpret_cast<NetworkCfg*>(arg1));
    typedef dopt::Socket* SocketPtr;

    SocketPtr* serverConnectionSockets = reinterpret_cast<SocketPtr*>(arg2);
    
    for (size_t i = 0; i < kTotalConnectionsPerClient; ++i)
    {
        // Try connect
        bool connectionEstablished = false;
        
        for (size_t connectAttempts = 0; connectAttempts < networkCfg.maxConnectAttempts; ++connectAttempts)
        {
            if (serverConnectionSockets[i]->connect(networkCfg.serverHostName.c_str(), networkCfg.serverPort + i))
            {
                connectionEstablished = true;
                break;
            }
            else
            {
                // Recreate socket for try connect one more time
                *(serverConnectionSockets[i]) = dopt::Socket(networkCfg.protocol);
                dopt::DefaultThread::sleepCurrentTh(networkCfg.maxConnectTimeoutMilliseconds / networkCfg.maxConnectAttempts);
            }
        }
        
        if (!connectionEstablished)
        {
            return TrainReturnCodes::eInternalError;
        }
    }

    for (size_t i = 0; i < kTotalConnectionsPerClient; ++i)
    {
        serverConnectionSockets[i]->setNoDelay(true /* no delay*/);
    }
    
    return TrainReturnCodes::eOk;
}

inline TrainReturnCodes train(const dopt::CmdLine& cmdline, ResultCallbackClient resultCallback)
{
    // Timer for measurements
    dopt::HighPrecisionTimer timer;

    // Parse Flags for command line interface

    DebugCfg debugFlags;
    TrainReturnCodes resultParseFlagsForDebug = parseFlagsForDebug(debugFlags, cmdline);
    if (resultParseFlagsForDebug != TrainReturnCodes::eOk)
        return resultParseFlagsForDebug;

    // Parse flags for network configutation
    NetworkCfg networkCfg;
    TrainReturnCodes resultParseFlagsForNetworkCfg = parseFlagsForNetworkConfiguration(networkCfg, cmdline);
    if (resultParseFlagsForNetworkCfg != TrainReturnCodes::eOk)
        return resultParseFlagsForNetworkCfg;
    
    // Explicitly check that we are running client
    assert(networkCfg.iAmServer == false);
    if (networkCfg.iAmServer)
    {
        std::cerr << "This binary application represent a server, not a client. Please specify '--iam client:number' during launching it\n";
        return TrainReturnCodes::eWrongArgument;
    }

    // Parse Flags for command line interface
    TrainingProcessCfg args;
    TrainReturnCodes resultParseFlagsForTrain = parseFlagsForTrain(args, cmdline, networkCfg.iAmServer);
    if (resultParseFlagsForTrain != TrainReturnCodes::eOk)
        return resultParseFlagsForTrain;


    bool printAtTraining = !(debugFlags.debugSilentPrintingInTraining);

    // Initialize Network Subsystem
    if (!dopt::Socket::initNetworkSubSystem())
    {
        if (printAtTraining)
            std::cerr << "There are problems with initializing Network Subsystem\n";
        return TrainReturnCodes::eInternalError;
    }

    timer.reset();
    
    // Initialize Runtime
    int nClients = args.runtime.nClients;                   ///< Obtained from command line
    int nWorkers = args.runtime.nWorkers;                   ///< Will be obtained
    assert(nWorkers == 1);
    std::vector<DatasetType> datasets_per_clients;          // Will be obtained (if needed)

    //dopt::Socket messageToSendGradientsSocket;        ///< Uncompressed information about gradients
    //dopt::Socket messageToSendLkSocket;               ///< Lk information to send (one scalar Lk)
    //dopt::Socket messageToSendFiSocket;               ///< Function value in current iterate
    //dopt::Socket messageToSendHessiansIndiciesSocket; ///< Compressed information about Hessian items indicies that are going to be send to master
    //dopt::Socket messageToSendHessiansItemsSocket;    ///< Compressed information about Hessian items values that are going to be send to master
    //dopt::Socket learningHessianSocket;               ///< Debug socket to send Hessian from clients
    //dopt::Socket auxInformationSocket;                ///< Aux information socket
    dopt::Socket controlSocket;                       ///< Debug socket to send Hessian from clients

    // Indicies to fetch items from arrays
    //constexpr size_t kGrad          = 0;
    //constexpr size_t kLk            = 1;
    //constexpr size_t kFi            = 2;
    //constexpr size_t kHessianInd    = 3;
    //constexpr size_t kHessianItems  = 4;
    //constexpr size_t kLearnHessians = 5;
    //constexpr size_t kAux           = 6;

    constexpr size_t kControl       = 0;

    // Connection Sockets
    dopt::Socket* serverConnectionSockets[] = { /*&messageToSendGradientsSocket,
                                                &messageToSendLkSocket, 
                                                &messageToSendFiSocket,
                                                &messageToSendHessiansIndiciesSocket, 
                                                &messageToSendHessiansItemsSocket, 
                                                &learningHessianSocket,
                                                &auxInformationSocket, */
                                                &controlSocket };

    static_assert(kTotalConnectionsPerClient == sizeof(serverConnectionSockets) / sizeof(serverConnectionSockets[0]));
    static_assert(kTotalConnectionsPerClient == kControl + 1);

    std::cout << "Create need socket " << timer.timeStamp() << '\n';

    // Phase 0: Create need sockets
    {
        for (size_t i = 0; i < kTotalConnectionsPerClient; ++i)
        {
            *(serverConnectionSockets[i]) = dopt::Socket(networkCfg.protocol);
        }
    }

    //=======================================================================================================//
    std::cout << "Connecting to master: " << networkCfg.serverHostName << ':' << networkCfg.serverPort << '\n';
    dopt::DefaultThread connectionThread(connectionToMaster, &networkCfg, serverConnectionSockets);
    //=======================================================================================================//

    // Parse input data
    auto theLineSeparator = [](char c) constexpr const_func_ext
    { return c == '\n' || c == '\r'; };

    auto theFieldSeparatorInLine = [](char c) constexpr const_func_ext
    { return c == '\t' || c == ' '; };

    auto theAttributeValueSeparator = [](char c) constexpr const_func_ext
    { return c == ':'; };

    auto theSymbolSkipping = [](char c) constexpr const_func_ext
    { return false; };

    // There is nothing to skip specifically for dataset during parsing
    constexpr bool kOmitSymbolSkipCheck = true;

    // Phase I: Load Dataset for client
    std::cout << "Load Dataset for client " << timer.timeStamp() << '\n';
    {
        dopt::FileSystemHelpers::FileMappingResult trainDs = dopt::FileSystemHelpers::mapFileToMemory(args.train_dataset.path.c_str(), true);

        if (!trainDs.isOk)
        {
            std::cout << "Train dataset: '" << args.train_dataset.path << "'. [Error during loading] " << timer.timeStamp() << '\n';
            std::cout << "Error message: '" << trainDs.errorMsg << '\n';
            return TrainReturnCodes::eTrainDatasetErrorInParsing;
        }

        dopt::Data trainDatasetRaw(trainDs.memory, trainDs.memorySizeInBytes, dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::TextFileDataSetLoader dataset_loader(trainDatasetRaw, theLineSeparator, theFieldSeparatorInLine, theAttributeValueSeparator, theSymbolSkipping);
        dopt::DataSetLoaderStats<IndexType, LabelType> stats = dataset_loader.computeStatisticsWithOneDataPass<false /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/, IndexType, LabelType>();

        if (!stats.ok)
        {
            std::cerr << "Train dataset: '" << args.train_dataset.path << "'. [Error during parsing] " << timer.timeStamp() << '\n';
            return TrainReturnCodes::eTrainDatasetErrorInParsing;
        }
        else
        {
            std::cerr << "Train dataset: '" << args.train_dataset.path << "' is parsed." << timer.timeStamp() << '\n';
        }

        DatasetType trainset;
        int dataset_loading_flags = (args.train_dataset.add_intercept == false ? dopt::CreateDatasetFlags::eNone : dopt::CreateDatasetFlags::eAddInterceptTerm);

        size_t processed_train_examples = dataset_loader.createDataset<true /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/>(trainset, stats, dataset_loading_flags | dopt::CreateDatasetFlags::eMakeRemappingForBinaryLogisticLoss);

        trainDatasetRaw.clear();

        bool unmapInputFile = dopt::FileSystemHelpers::unmapFileFromMemory(trainDs);
        assert(unmapInputFile == true);

        if (printAtTraining)
            std::cout << "Train dataset: '" << args.train_dataset.path << "' was instantiated [OK]. " << timer.timeStamp() << '\n';
        printInformationAboutDataset("train dataset", trainset,
                                     debugFlags.debugCrc32ForDatasets,
                                     debugFlags.debugPrintTrainDataset);

        if (args.train_dataset.path_is_client_specific_data)
        {
            // Specified dataset is client specific - we can prepate array of datasets without any problem
            // TODO: In future just remove datasets_per_clients
            datasets_per_clients.resize(nClients);
            datasets_per_clients[networkCfg.clientNumber] = trainset;
        }
        else
        {
            // Specified dataset is for all clients - to get a specific subset of data which will be used for specific client, reshuffling should be carried
            size_t totalSamples = trainset.totalSamples();
            size_t totalSamplesPerClient = totalSamples / nClients;
            int skipped_samples = 0;

            // Index for all datapoints
            VectorND<IndexType> examples = VectorND<IndexType>::sequence<(IndexType)0>(totalSamples);

            // Indicies per client
            std::vector<std::vector<IndexType>> examplesPerClient = std::vector<std::vector<IndexType>>(nClients);

            // Indicies for skipped samples
            std::vector<IndexType> skippedExamples;
            skippedExamples.reserve(totalSamples - totalSamplesPerClient);

            {
                if (args.train_dataset.reshuffle)
                {
                    dopt::RandomGenIntegerLinear gen4shuffler;
                    gen4shuffler.setSeed(args.train_dataset.reshuffle_seed);
                    dopt::shuffle(examples, gen4shuffler);
                }

                // Strategy: drop residual samples (and ignore allocation of samples for not current client)
                int current_sample = 0;
                for (int c = 0; c < nClients; ++c)
                {
                    if (c == networkCfg.clientNumber)
                    {
                        examplesPerClient[c].reserve(totalSamplesPerClient);

                        for (int s = 0; s < totalSamplesPerClient; ++s, ++current_sample)
                        {
                            examplesPerClient[c].push_back(examples[current_sample]);
                        }
                    }
                    else
                    {
                        for (int s = 0; s < totalSamplesPerClient; ++s, ++current_sample)
                        {
                            skippedExamples.push_back(examples[current_sample]);
                        }
                    }
                }

                for (; current_sample < totalSamples; ++current_sample)
                    skippedExamples.push_back(examples[current_sample]);


                datasets_per_clients = trainset.splitDataset(examplesPerClient);

                if (printAtTraining)
                {
                    std::cout << "  Train samples has been reshuffled: " << (args.train_dataset.reshuffle ? "[YES]" : "[NO]") << '\n';
                    std::cout << "  Train samples reshuffled seed: " << (args.train_dataset.reshuffle_seed) << '\n';
                    std::cout << "  Number of clients: " << nClients << '\n';
                    std::cout << "  Number of samples per client: " << totalSamplesPerClient << '\n';
                    std::cout << "  Number of skipped samples during samples distribution: " << skippedExamples.size() << '\n';
                    std::cout << "  Train data distribution across clients has been completed. [OK] " << timer.timeStamp() << '\n';
                    std::cout << "=========================================================================\n";
                }

                if (cmdline.isFlagSetuped("check_split"))
                {
                    if (!trainset.checkSplitConsistency(datasets_per_clients, examplesPerClient, skippedExamples, trainset))
                    {
                        std::cerr << "  Checking data split [FAILED] " << timer.timeStamp() << '\n';
                        return TrainReturnCodes::eDataSplitError;
                    }
                    else
                    {
                        if (printAtTraining)
                            std::cout << "  Checking data split [OK] " << timer.timeStamp() << '\n';
                    }
                }
                else
                {
                    if (printAtTraining)
                        std::cout << "  Checking data split [SKIP] \n";
                }
            }
        }

        // Data is loaded. 
        std::cout << "Data is loaded " << timer.timeStamp() << '\n';
    }

    timer.reset();
    if (printAtTraining)
        std::cout << "=========================================================================\n";

    // algorithm selector: gd || fednl1
    bool is_algorithm_gd = (args.optAlgo.algorithm == "gd");
    bool is_algorithm_fednl1 = (args.optAlgo.algorithm == "fednl1");

    // options for fednl
    bool is_algorithm_fednl_has_option_a = cmdline.isFlagSetuped("fednl-option-a");
    bool is_algorithm_fednl_has_option_b = cmdline.isFlagSetuped("fednl-option-b");

    bool compute_L_smooth = cmdline.isFlagSetuped("compute-L-smooth") || args.optAlgo.has_theoretical_global_lr_flag;

    if (is_algorithm_fednl1)
    {
        if (!is_algorithm_fednl_has_option_a && !is_algorithm_fednl_has_option_b)
        {
            std::cerr << "For FedNL you should specify fednl-option-a or fednl-option-b\n";
            return TrainReturnCodes::eMissingArgument;
        }
    }
    //=======================================================================================//
    connectionThread.join();

    if (connectionThread.getExitCode() == TrainReturnCodes::eOk)
    {
        std::cout << "  Connection to Master from client: " << networkCfg.clientNumber 
                  << " has been established successfully [OK]" 
                  << '\n';
    }
    else
    {
        std::cout << "  Can not connect to master: " << networkCfg.serverHostName 
                  << ':' << networkCfg.serverPort 
                  << '\n';
        return (TrainReturnCodes)connectionThread.getExitCode();
    }
    //=======================================================================================//

    std::cout << "=========================================================================\n";
    std::cout << "Main Loop preparation for: " << args.optAlgo.algorithm << '\n';
    std::cout << "=========================================================================\n";
    
    // Dimension of Optimization Problem
    size_t d = 0;
    size_t iterations = 0;

    // Qunatities will be obtained from Master
    double mathMuf = -1;
    double mathLf = -1.0;

    {
        MatrixNMD_d train_samples;
        const MatrixNMD_d& train_samples_tr = datasets_per_clients[networkCfg.clientNumber].train_samples_tr;

        for (;;)
        {
            WorkerContext::ControlSignals nextCommand;
            bool nextCommandIsReceived = WorkerContext::receiveCommand</*block*/true> (nextCommand, *(serverConnectionSockets[kControl]));
            
            if (nextCommandIsReceived == false)
            {
                dopt::DefaultThread::yeildCurrentThInHotLoop();
                continue;
            }
            //================================================================================================================//
            
            if (nextCommand == WorkerContext::ControlSignals::sig_terminate_loop)
            {
                WorkerContext::ignoreResidualFromCommand(nextCommand, *(serverConnectionSockets[kControl]));
                break;
            }
            else if (nextCommand == WorkerContext::ControlSignals::sig_halt)
            {
                std::cout << "Request Halt by server";
                WorkerContext::ignoreResidualFromCommand(nextCommand, *(serverConnectionSockets[kControl]));
                return TrainReturnCodes::eInternalError;
            }
            else if (nextCommand == WorkerContext::ControlSignals::sig_request_matrix_vector_multiplication_with_samples)
            {
                if (train_samples.sizeInBytesNoPadding() == 0)
                {
                    train_samples = train_samples_tr.getTranspose();
                }

                VectorND_d nextCommandVectorArgument = WorkerContext::extractVectorArgFromCommand(nextCommand, *(serverConnectionSockets[kControl]));
                VectorND_d tmp = train_samples * nextCommandVectorArgument;

                dopt::MutableData buffer;
                WorkerContext::sendMatrixVectorMultiplyWithSamples(buffer, tmp);                
                serverConnectionSockets[kControl]->sendData(buffer.getPtr(), buffer.getFilledSize());
            }
            else if (nextCommand == WorkerContext::ControlSignals::sig_request_matrix_vector_multiplication_with_samples_tranpose)
            {
                VectorND_d nextCommandVectorArgument = WorkerContext::extractVectorArgFromCommand(nextCommand, *(serverConnectionSockets[kControl]));
                VectorND_d tmp = train_samples_tr * nextCommandVectorArgument;

                dopt::MutableData buffer;
                WorkerContext::sendMatrixVectorMultiplyWithSamplesTranpose(buffer, tmp);
                serverConnectionSockets[kControl]->sendData(buffer.getPtr(), buffer.getFilledSize());
            }
            else if (nextCommand == WorkerContext::ControlSignals::sig_get_worker_desciption)
            {
                WorkerContext::ignoreResidualFromCommand(nextCommand, *(serverConnectionSockets[kControl]));

                dopt::Socket& socket = *(serverConnectionSockets[kControl]);

                ClientDescription description;
                description.clientId = networkCfg.clientNumber;
                description.dimension = datasets_per_clients[networkCfg.clientNumber].numberOfAttributesForSamples();
                description.samplesInClient = datasets_per_clients[networkCfg.clientNumber].totalSamples();
                description.rounds = args.runtime.rounds;
                description.hasInterceptTerm = datasets_per_clients[networkCfg.clientNumber].hasInterceptTerm();

                dopt::MutableData buffer;
                if (WorkerContext::sendWorkerDecription(buffer, description) == false)
                {
                    std::cout << "Can not send task description from client: " << networkCfg.clientNumber << '\n';
                    return eInternalError;
                }
                if (serverConnectionSockets[kControl]->sendData(buffer.getPtr(), buffer.getFilledSize()) == false)
                {
                    std::cout << "Can not send task description from client: " << networkCfg.clientNumber << '\n';
                    return eInternalError;
                }
            }
            else if (nextCommand == WorkerContext::ControlSignals::sig_info_about_opt_problem)
            {
                OptProblemDescription descr = WorkerContext::extractProblemDescrFromCommand(nextCommand, *(serverConnectionSockets[kControl]));
                
                if (descr.flags & OptProblemDescriptionFlags::eDimension)
                    d = descr.d;                    

                if (descr.flags & OptProblemDescriptionFlags::eMu)
                    mathMuf = descr.mu_f;

                if (descr.flags & OptProblemDescriptionFlags::eL)
                    mathLf = descr.L_f;
            }
            else
            {
                std::cout << "Unexpected command at this stage: " << nextCommand << '\n';
                WorkerContext::ignoreResidualFromCommand(nextCommand, *(serverConnectionSockets[kControl]));

                return TrainReturnCodes::eInternalError;
            }
        }
    }
    //================================================================================================================//
    // Derived Quantitity from Specification

    // Number of Rounds
    int rounds = args.runtime.rounds;

    // LS parameters
    bool has_line_search = args.optAlgo.has_line_search;

    double c_line_search = args.optAlgo.c_line_search;
    double gamma_line_search = args.optAlgo.gamma_line_search;
    constexpr double t_start_line_search = 1.0;

    assert(c_line_search > 0.0);
    assert(c_line_search <= 0.5 + 1e-9);
    assert(t_start_line_search > 0.0);

    assert(gamma_line_search > 0.0);
    assert(gamma_line_search < 1.0);

    // Use global and local step size
    double used_global_step_size = args.optAlgo.global_step_size;
    double used_alpha_step_size = args.optAlgo.alpha_step_size;

    // K for compressor
    int32_t kForCompressor = args.optAlgo.k_compressor_as_d_mult * d;

    // Type of used compressor
    Compressor used_compressor = Compressor::eIdentical;

    if (args.optAlgo.algorithm != "gd")
    {
        if (args.optAlgo.compressor == "identical")
        {
            used_compressor = Compressor::eIdentical;
            kForCompressor = ((d + 1) * d) / 2; // Send information is only upper triangular part of hessian: all pairs {i,j} i \ne j AND d (i,i) tuples.
        }
        else if (args.optAlgo.compressor == "natural")
        {
            used_compressor = Compressor::eNatural;
            kForCompressor = ((d + 1) * d) / 2; // Send information is only upper triangular part of hessian: all pairs {i,j} i \ne j AND d (i,i) tuples.
        }
        else if (args.optAlgo.compressor == "randk")
        {
            used_compressor = Compressor::eRandK;
        }
        else if (args.optAlgo.compressor == "topk")
        {
            used_compressor = Compressor::eTopK;
        }
        else if (args.optAlgo.compressor == "toplek")
        {
            used_compressor = Compressor::eTopLEK;
        }
        else if (args.optAlgo.compressor == "seqk")
        {
            used_compressor = Compressor::eRandSeqK;
        }
        else
        {
            if (printAtTraining)
                std::cerr << "You have specified unknown type of compressor: " << args.optAlgo.compressor << '\n';
            return TrainReturnCodes::eWrongArgument;
        }
    }

    // We use theoretical global lr for GD. TODO: rename that this flag is GD relative
    if (args.optAlgo.has_theoretical_global_lr_flag)
    {
        used_global_step_size = 2.0 / (mathMuf + mathLf);
    }

    if (args.optAlgo.has_theoretical_alpha_flag)
    {
        // Theretical step sizes for strongly convex case
        if (is_algorithm_fednl1)
        {
            if (used_compressor == Compressor::eIdentical)
            {
                double w = 0.0;
                used_alpha_step_size = 1.0 / (w + 1.0);
            }
            else if (used_compressor == Compressor::eNatural)
            {
                double w = 1.0/8.0;
                used_alpha_step_size = 1.0 / (w + 1.0);
            }
            else if (used_compressor == Compressor::eRandK || used_compressor == Compressor::eRandSeqK)
            {
                // W quanity for unbiased compressors
                double w = dopt::computeWForRandKMatrixOperator(kForCompressor, d);

                // Number of iterms in upper triangular part of square marix with shape [d,d] with diagonal.
                size_t totalPossibleItems2Send = ((d + 1) * d) / 2;

                // Alpha step size from theore
                used_alpha_step_size = 1.0 / (w + 1.0);

                if (kForCompressor > totalPossibleItems2Send)
                {
                    if (printAtTraining)
                        std::cerr << " Multiplier for D is specifed as "
                                  << args.optAlgo.k_compressor_as_d_mult
                                  << ". For this problem with D=" << d
                                  << " maximum allowable muliplier is "
                                  << double(totalPossibleItems2Send) / double(d) << " !\n";

                    return TrainReturnCodes::eWrongArgument;
                }
            }
            else if (used_compressor == Compressor::eTopK)
            {
                // Number of iterms in upper triangular part of square marix with shape [d,d] with diagonal.
                double delta = dopt::computeDeltaForTopKMatrixOpeator(kForCompressor, d);
                size_t totalPossibleItems2Send = ((d + 1) * d) / 2;

                if (kForCompressor > totalPossibleItems2Send)
                {
                    if (printAtTraining)
                        std::cerr << " Multiplier for D is specifed as "
                                  << args.optAlgo.k_compressor_as_d_mult
                                  << ". For this problem with D=" << d
                                  << " maximum allowable muliplier is "
                                  << double(totalPossibleItems2Send) / double(d) << "!\n";

                    return TrainReturnCodes::eWrongArgument;
                }

                if (args.optAlgo.use_theoretical_alpha_option_1)
                {
                    // FedNL Option-1
                    used_alpha_step_size = 1.0 - sqrt(1.0 - delta);
                }
                else if (args.optAlgo.use_theoretical_alpha_option_2)
                {
                    // FedNL Option-2
                    used_alpha_step_size = 1.0;
                }
                else
                {
                    if (printAtTraining)
                        std::cerr << "Please specify --use_theoretical_alpha_option_2 or --use_theoretical_alpha_option_1 in command line\n";
                    return TrainReturnCodes::eMissingArgument;
                }
            }
            else if (used_compressor == Compressor::eTopLEK)
            {
                // Number of iterms in upper triangular part of square marix with shape [d,d] with diagonal.
                double delta = dopt::computeDeltaForTopKMatrixOpeator(kForCompressor, d);
                size_t totalPossibleItems2Send = ((d + 1) * d) / 2;

                if (kForCompressor > totalPossibleItems2Send)
                {
                    if (printAtTraining)
                        std::cerr << " Multiplier for D is specifed as "
                            << args.optAlgo.k_compressor_as_d_mult
                            << ". For this problem with D=" << d
                            << " maximum allowable muliplier is "
                            << double(totalPossibleItems2Send) / double(d) << "!\n";

                    return TrainReturnCodes::eWrongArgument;
                }

                if (args.optAlgo.use_theoretical_alpha_option_1)
                {
                    // FedNL Option-1
                    used_alpha_step_size = 1.0 - sqrt(1.0 - delta);
                }
                else if (args.optAlgo.use_theoretical_alpha_option_2)
                {
                    // FedNL Option-2
                    used_alpha_step_size = 1.0;
                }
                else
                {
                    if (printAtTraining)
                        std::cerr << "Please specify --use_theoretical_alpha_option_2 or --use_theoretical_alpha_option_1 in command line\n";

                    return TrainReturnCodes::eMissingArgument;
                }
            }
        }
    }

    // Multiply global step size
    used_global_step_size *= args.optAlgo.global_step_size_multiplier;

    VectorND_d x0(d);
    args.runtime.x0_rnd_generator.setSeed(args.runtime.x0_seed);
    x0.setAllRandomly(args.runtime.x0_rnd_generator);

    std::vector<dopt::DefaultThread*> localWorkers;
    std::vector<WorkerContext*>       localWorkersContext;
    std::vector<dopt::RandomGenIntegerLinear> localWorkersRandomNumberGenerators;

    localWorkers.reserve(nWorkers);
    localWorkersContext.reserve(nClients);
    localWorkersRandomNumberGenerators.reserve(nClients);

    // Silent Run
    bool silentRun = cmdline.isFlagSetuped("silent");

    // Derived quantity to reduce number of division
    double inv_NClients = 1.0 / double(nClients);

    double updateScalingInMaster = inv_NClients * used_alpha_step_size;

    // id for a computation job
    int run_id_int = int(dopt::RandomGenCrt::global().generateReal() * 10000);
    std::string runId = dopt::string_utils::toString(run_id_int);

    // special strategies to save communication
    bool transfer_indicies_for_randk = cmdline.isFlagSetuped("transfer_indicies_for_randk");

    for (int c = 0; c < nClients; ++c)
    {        
        if (c == networkCfg.clientNumber)
        {
            auto * clientOptProblem = new dopt::L2RegulirizeLogisticRegression(datasets_per_clients[c].train_samples_tr,
                                                                               datasets_per_clients[c].train_outputs,
                                                                               args.optPrb.lambda);

            if (is_algorithm_gd)
            {
                WorkerContextForGD* ctx = new WorkerContextForGD();
                ctx->workerIndex = c;
                ctx->send_fi_from_worker = args.tracking.tracking_is_on;

                ctx->optProblem = clientOptProblem;
                ctx->ctrBlock.gradIsReady = false;
                ctx->ctrBlock.fiIsReady = false;

                ctx->control = serverConnectionSockets[kControl];

                localWorkersContext.push_back(ctx);

                //ctx->messageToSendGradientsSocket = serverConnectionSockets[kGrad];
                //ctx->messageToSendFiSocket = serverConnectionSockets[kFi];
                //ctx->control = serverConnectionSockets[kControl];
                //ctx->auxInformation = serverConnectionSockets[kAux];
            }
            else if (is_algorithm_fednl1)
            {
                WorkerContextForFedNL* ctx = new WorkerContextForFedNL();

                ctx->workerIndex = c;
                ctx->transfer_indicies_for_randk = transfer_indicies_for_randk;
                ctx->send_fi_from_worker = (args.tracking.tracking_is_on || args.optAlgo.has_line_search);

                ctx->optProblem = clientOptProblem;

                ctx->ctrBlock.messageToSendGradientsIsReady = false;

                ctx->ctrBlock.messageToSendHessiansIndiciesIsReady = false;
                ctx->ctrBlock.messageToSendHessiansItemsIsReady = false;

                ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk = false;
                ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk = false;

                ctx->ctrBlock.messageToSendLearningHessiansIsReady = false;
                ctx->ctrBlock.messageToSendFiIsReady = false;
                ctx->ctrBlock.messageToSendLkIsReady = false;
                ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = false;

                ctx->compressorType = used_compressor;
                ctx->kForCompressor = kForCompressor;
                ctx->seedForRandomizedCompressor = size_t(123 + c); // hard coded seed for randomized compressor. TODO: If needed to make it more explicitly if needed.
                ctx->alpha = used_alpha_step_size;
                ctx->preScaleBeforeSend2Master = updateScalingInMaster;
                ctx->send_Lk_from_worker = (is_algorithm_fednl_has_option_b ? true : false);
                ctx->raise_finish_flag_for_worker = kTrackHessianDifference; // this flag is only needed in case of tracking difference

                if (clientOptProblem)
                {
                    ctx->learningHessian = clientOptProblem->evaluateHessian(x0);
                }

                ctx->control = serverConnectionSockets[kControl];

                //ctx->messageToSendGradientsSocket = serverConnectionSockets[kGrad];
                //ctx->messageToSendLkSocket = serverConnectionSockets[kLk];
                //ctx->messageToSendFiSocket = serverConnectionSockets[kFi];
                //ctx->messageToSendHessiansIndiciesSocket = serverConnectionSockets[kHessianInd];
                //ctx->messageToSendHessiansItemsSocket = serverConnectionSockets[kHessianItems];
                //ctx->learningHessianSocket = serverConnectionSockets[kLearnHessians];
                //ctx->auxInformation = serverConnectionSockets[kAux];
                //ctx->control = serverConnectionSockets[kControl];

                if (ctx->compressorType == Compressor::eRandK || ctx->compressorType == Compressor::eRandSeqK)
                {
                    dopt::RandomGenIntegerLinear g;
                    g.setSeed(ctx->seedForRandomizedCompressor);
                    localWorkersRandomNumberGenerators.push_back(g);
                }

                localWorkersContext.push_back(ctx);
            }
            else
            {
                if (printAtTraining)
                    std::cerr << " Please specify valid algorithm name.\n";
                return TrainReturnCodes::eInternalError;
            }
        }
        else
        {
            localWorkersContext.push_back(nullptr);
        }
    }

    std::vector<std::vector<WorkerContext*>> localWorkersContextPerWorker(nWorkers);
    assert(nWorkers == 1);
    
    {
        for (uint32_t curClient = networkCfg.clientNumber; curClient < networkCfg.clientNumber + 1; curClient++)
        {
            for (int w = 0; w < nWorkers; ++w)
            {
                localWorkersContextPerWorker[w].push_back(localWorkersContext[curClient]);
            }
        }
    }
    
    //===================================Create server context ===========================================================
    VectorND_d xCur = x0;
    ServerContext serverContext;
    serverContext.rounds = rounds;
    serverContext.currenIterate = &xCur;
    serverContext.roundToStart = std::numeric_limits<size_t>::max();
    serverContext.lineSearchRound = has_line_search ? (0) : (std::numeric_limits<size_t>::max());
    serverContext.lineSearchIteration = std::numeric_limits<size_t>::max();

    //====================================================================================================================
    // Create worker threads start
    if (is_algorithm_gd)
    {
        for (int w = 0; w < nWorkers; ++w)
        {
            std::vector<WorkerContext*>* wctx = &(localWorkersContextPerWorker[w]);
            dopt::DefaultThread* th = new dopt::DefaultThread(workerThreadTrainLoopForGD, wctx, &serverContext);
            assert(th);
            localWorkers.push_back(th);
        }
    }
    else if (is_algorithm_fednl1)
    {
        for (int w = 0; w < nWorkers; ++w)
        {
            std::vector<WorkerContext*>* wctx = &(localWorkersContextPerWorker[w]);
            dopt::DefaultThread* th = nullptr;

            if (used_compressor == Compressor::eTopK)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_TopKCompressor, wctx, &serverContext);
            else if (used_compressor == Compressor::eTopLEK)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_TopLEKCompressor, wctx, &serverContext);
            else if (used_compressor == Compressor::eRandK)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_RandKCompressor, wctx, &serverContext);
            else if (used_compressor == Compressor::eRandSeqK)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_RandSeqKCompressor, wctx, &serverContext);
            else if (used_compressor == Compressor::eIdentical)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_IdenticalCompressor, wctx, &serverContext);
            else if (used_compressor == Compressor::eNatural)
                th = new dopt::DefaultThread(workerThreadTrainLoopForFedNL1_NaturalCompressor, wctx, &serverContext);

            assert(th);
            localWorkers.push_back(th);
        }
    }
    else
    {
        std::cerr << " Please specify valid algorithm name.\n";
        return TrainReturnCodes::eInternalError;
    }
    // Create worker threads end
    //====================================================================================================================

    // Start worker threads
    for (int c = 0; c < nWorkers; ++c)
    {
        localWorkers[c]->join();
    }

    // Finish
    //====================================================================================================================
    if (printAtTraining)
    {
        std::cout << "=========================================================================\n";
        std::cout << "Statistics\n\n";
        std::cout << "  Total number of rounds: " << rounds << '\n';
        std::cout << "  Total number of clients: " << nClients << '\n';
        std::cout << '\n';
        std::cout << "  Algorithm: " << args.optAlgo.algorithm << '\n';
        std::cout << "  Data set: " << args.train_dataset.path << '\n';
        std::cout << "  d: " << d << '\n';
        std::cout << '\n';
        std::cout << "  Global step size (with multiplier): " << used_global_step_size << '\n';
        std::cout << "  Global step size multiplier: " << args.optAlgo.global_step_size_multiplier << '\n';
        std::cout << "  Is global step size theoretical: " << (args.optAlgo.has_theoretical_global_lr_flag ? "[YES]" : "[NO]") << '\n';
        std::cout << "  Alpha step size: " << used_alpha_step_size << '\n';
        std::cout << "  Is alpha step size theoretical: " << (args.optAlgo.has_theoretical_alpha_flag ? "[YES]" : "[NO]") << '\n';
        std::cout << "  Lambda for regularization: " << args.optPrb.lambda << '\n';
        std::cout << "  Clients compressor: " << args.optAlgo.compressor << '\n';
        std::cout << "  Clients compressor K: " << kForCompressor << " / Hessian shape is [d,d] where d: " << d
              << " / Maximum K ( {d(d+1)}/2 ): " << (d * (d + 1)) / 2 << '\n';
        std::cout << '\n';
        std::cout << "  Last iterate CRC-32: " << dopt::crc32(xCur.rawData(), xCur.sizeInBytes(), dopt::crc32Seed()) << '\n';
        std::cout << "  Norm of last iterate: " << xCur.vectorL2Norm() << '\n';
        std::cout << '\n';
        std::cout << "  Line Search: " << (args.optAlgo.has_line_search ? "[YES]" : "[NO]") << '\n';
        std::cout << "  Line Search parameters: " << "c: " << args.optAlgo.c_line_search << ", gamma: " << args.optAlgo.gamma_line_search << '\n';
        std::cout << "=========================================================================\n";
        std::cout << '\n';
        std::cout << "Debug information\n\n";
        std::cout << "  Generated run id: " << runId;
        std::cout << "  Transfer indicies for RandK: " << (transfer_indicies_for_randk ? "[YES]" : "[NO]") << '\n';
        std::cout << "  mu(f): " << mathMuf << '\n';

        if (compute_L_smooth)
        {
            std::cout << "  L(f): " << mathLf << '\n';
            std::cout << "  cond(f) = (L(f)/mu(f)): " << mathLf / mathMuf << '\n';
        }
        else
        {
            std::cout << "  L(f): " << "unavailable" << '\n';
            std::cout << "  cond(f) = (L(f)/mu(f)): " << "unavailable" << '\n';
        }

        std::cout << "  Maximum allowable k multiplier for RandK and TopK: " << double(((d + 1) * d) / 2) / d << '\n';
        std::cout << "  Maximum sendable components for RandK and TopK from one client: " << double(((d + 1) * d) / 2) << '\n';

        std::cout << '\n';
        std::cout << "  Print information about CRC32 for datasets: " << debugFlags.debugCrc32ForDatasets << '\n';
        std::cout << "  Print information about used memory: " << debugFlags.debugMemInfo << '\n';
        std::cout << "  Server obtainining updates from clients in order: " << debugFlags.debugForceSequentialUpdate << '\n';
        std::cout << '\n';
        std::cout << "=========================================================================\n";

        std::cout << "  Training has been finished successfully [OK]\n";
    }

    assert(localWorkers.size() == nWorkers);
    
    for (int c = 0; c < nWorkers; ++c)
    {
        delete localWorkers[c];
    }

    assert(localWorkersContext.size() == nClients);

    for (int c = 0; c < nClients; ++c)
    {
        delete localWorkersContext[c];
    }

    // Remove all items from worker contexts
    localWorkersContext.clear();
    localWorkers.clear();

    if (printAtTraining)
    {
        std::cout << "=========================================================================\n";
    }

    // Deinitialize Network Subsystem
    dopt::Socket::deinitNetworkSubSystem();
    
    if (resultCallback) {
        resultCallback(xCur.size());
    }

    // Return that all is ok
    return TrainReturnCodes::eOk;
}
