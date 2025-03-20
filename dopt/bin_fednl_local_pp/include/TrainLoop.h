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
#include "LocalWorkerForFedNL_Natural.h"
#include "LocalWorkerForFedNL_RandK.h"
#include "LocalWorkerForFedNL_RandSeqK.h"
#include "LocalWorkerForFedNL_TopK.h"
#include "LocalWorkerForFedNL_TopLEK.h"
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

// Threadpool
#include "dopt/system/include/threads/ThreadPoolWithTaskQueue.h"

// Callbacks
#include "TrainPPCallbaksShared.h"

#include <fstream>
#include <stdlib.h>
#include <assert.h>

inline double computeLkAverage(const std::vector<WorkerContext*>& localWorkersContext, const ServerContext* ctx)
{
    double result = 0.0;
    int nClients = localWorkersContext.size();

    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

        if (!clientHasBeenSelected(c, ctx))
        {
            result += wctx->localLk;
            continue;
        }

        for (; wctx->ctrBlock.messageToSendLkIsReady != true; )
            dopt::DefaultThread::yeildCurrentThInHotLoop();
        
        result += wctx->localLk;
    }

    return result / double(nClients);
}

inline VectorND_d computeGdirectionAverage(const std::vector<WorkerContext*>& localWorkersContext, const ServerContext* ctx)
{
    VectorND_d result = VectorND_d(ctx->currenIterate->size() );
    int nClients = localWorkersContext.size();

    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

        if (!clientHasBeenSelected(c, ctx))
        {
            result += wctx->localGradient;
            continue;
        }

        for (; wctx->ctrBlock.messageToSendLkIsReady != true; )
            dopt::DefaultThread::yeildCurrentThInHotLoop();

        result += wctx->localGradient;
    }

    return result / double(nClients);
}

inline VectorND_d computeFullGradient(const std::vector<WorkerContext*>& localWorkersContext, const VectorND_d& x)
{
    VectorND_d result = VectorND_d(x.size());
    int nClients = localWorkersContext.size();

    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];
        result += wctx->optProblem->evaluateGradient(x);
    }

    return result / double(nClients);
}


inline void waitForInformationFromClientsForFedNL(const std::vector<WorkerContext*>& localWorkersContext,  const ServerContext* ctx, bool is_algorithm_fednl_has_option_b)
{
    int nClients = localWorkersContext.size();
    
    // Wait for grads
    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

        if (!clientHasBeenSelected(c, ctx))
        {
            assert(wctx->ctrBlock.messageToSendGradientsIsReady == false);
            continue;
        }
        
        for (; wctx->ctrBlock.messageToSendGradientsIsReady != true; )
        {
            dopt::DefaultThread::yeildCurrentThInHotLoop();
        }
    }

    // Wait for hessians
    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

        if (!clientHasBeenSelected(c, ctx))
        {
            assert(wctx->ctrBlock.messageToSendHessiansIsReady == false);
            continue;
        }

        for (; wctx->ctrBlock.messageToSendHessiansIsReady != true;) {
            dopt::DefaultThread::yeildCurrentThInHotLoop();
        }
    }

    // Wait for Lk
    if (is_algorithm_fednl_has_option_b)
    {
        for (int c = 0; c < nClients; ++c)
        {
            WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

            if (!clientHasBeenSelected(c, ctx))
            {
                assert(wctx->ctrBlock.messageToSendLkIsReady == false);
                continue;
            }

            for (; wctx->ctrBlock.messageToSendLkIsReady != true;) {
                dopt::DefaultThread::yeildCurrentThInHotLoop();
            }
        }
    }
}

inline TrainReturnCodes train(const dopt::CmdLine& cmdline, ResultCallbackLocalPP resultCallback)
{
    // Timer for measurements
    dopt::HighPrecisionTimer timer;

    // Parse Flags for command line interface

    DebugCfg debugFlags;
    TrainReturnCodes resultParseFlagsForDebug = parseFlagsForDebug(debugFlags, cmdline);
    if (resultParseFlagsForDebug != TrainReturnCodes::eOk)
        return resultParseFlagsForDebug;

    TrainingProcessCfg args;
    TrainReturnCodes resultParseFlagsForTrain = parseFlagsForTrain(args, cmdline, false);
    if (resultParseFlagsForTrain != TrainReturnCodes::eOk)
        return resultParseFlagsForTrain;


    bool printAtTraining = !(debugFlags.debugSilentPrintingInTraining);

    // Parse input data
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
    
    timer.reset();
    
    dopt::FileSystemHelpers::FileMappingResult trainDs = dopt::FileSystemHelpers::mapFileToMemory(args.train_dataset.path.c_str(), true);
    
    if (!trainDs.isOk)
    {
        if (printAtTraining) {
            std::cerr << "Train dataset: '" << args.train_dataset.path << "'. [Error during loading] " << timer.timeStamp() << '\n';
            std::cerr << "Error message: '" << trainDs.errorMsg << '\n';
        }

        return TrainReturnCodes::eTrainDatasetErrorInParsing;
    }
    
    dopt::Data trainDatasetRaw(trainDs.memory, trainDs.memorySizeInBytes, dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
    dopt::TextFileDataSetLoader dataset_loader(trainDatasetRaw, theLineSeparator, theFieldSeparatorInLine, theAttributeValueSeparator, theSymbolSkipping);
    dopt::DataSetLoaderStats<IndexType, LabelType> stats = dataset_loader.computeStatisticsWithOneDataPass<false /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/, IndexType, LabelType>();
    
    if (!stats.ok)
    {
        if (printAtTraining)
            std::cerr << "Train dataset: '" << args.train_dataset.path << "'. [Error during parsing] " << timer.timeStamp() << '\n';
        return TrainReturnCodes::eTrainDatasetErrorInParsing;
    }
    else
    {
        if (printAtTraining)
            std::cout << "Train dataset: '" << args.train_dataset.path << "' is parsed." << timer.timeStamp() << '\n';
    }

    DatasetType trainset;
    int dataset_loading_flags = (args.train_dataset.add_intercept == false ? dopt::CreateDatasetFlags::eNone : dopt::CreateDatasetFlags::eAddInterceptTerm);

    size_t processed_train_examples = dataset_loader.createDataset<true /*restart datasource*/, kOmitSymbolSkipCheck /*skip symbol checks*/> (trainset, stats, dataset_loading_flags | dopt::CreateDatasetFlags::eMakeRemappingForBinaryLogisticLoss);

    trainDatasetRaw.clear();

    bool unmapInputFile = dopt::FileSystemHelpers::unmapFileFromMemory(trainDs);
    assert(unmapInputFile == true);

    if (printAtTraining) {
        std::cout << "Train dataset: '" << args.train_dataset.path << "' was instantiated [OK]. " << timer.timeStamp() << '\n';
        printInformationAboutDataset("train dataset", trainset,
                                                      debugFlags.debugCrc32ForDatasets,
                                                      debugFlags.debugPrintTrainDataset);
    }
    // Train Dataset has been parsed
    
    // Initialize Runtime
    timer.reset();

    int nClients = args.train_dataset.nClients;  ///< Number of (local) clients
    int nWorkers = args.runtime.nWorkers;        ///< Workers for process

    dopt::RandomGenIntegerLinear clientSampler;                 ///< Sampler for client selection
    int nSelectedClients = args.runtime.nClientsPerRound;       ///< Number of (local) clients
    
    assert(nSelectedClients > 0 && nSelectedClients <= nClients);
    
    VectorND<IndexType> allClients = VectorND<IndexType>::sequence<(IndexType)0>(nClients);
    
    VectorND<IndexType> selectedClients;
    clientSampler.setSeed(args.runtime.clientSelectionSeed);
    
    int workItemsPerClientLowerBound = 0;
    int workItemsPerClientUpperBound = 0;

    if (nWorkers >= nClients)
    {
        nWorkers = nClients;
        workItemsPerClientLowerBound = 1;
        workItemsPerClientUpperBound = 1;
    }
    else
    {
        workItemsPerClientLowerBound = nClients / nWorkers;
        workItemsPerClientUpperBound = (nClients + nWorkers - 1) / nWorkers;
    }

    int nFreeWorkers = nWorkers - nClients;
    if (nFreeWorkers < 0)
        nFreeWorkers = 0;

    size_t totalSamples = trainset.totalSamples();
    size_t totalSamplesPerClient = totalSamples / nClients;
    int skipped_samples = 0;

    // Index for all datapoints
    VectorND<IndexType> examples = VectorND<IndexType>::sequence<(IndexType)0> (totalSamples);
    
    // Indicies per client
    std::vector<std::vector<IndexType>> examplesPerClient = std::vector<std::vector<IndexType>>(nClients);

    // Indicies for skipped samples
    std::vector<IndexType> skippedExamples;
    skippedExamples.reserve(totalSamples - totalSamplesPerClient * nClients);

    {
        if (args.train_dataset.reshuffle)
        {
            dopt::RandomGenIntegerLinear gen4shuffler;
            gen4shuffler.setSeed(args.train_dataset.reshuffle_seed);
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

    if (printAtTraining) {
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
            if (printAtTraining)
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

    timer.reset();

    if (printAtTraining)
        std::cout << "=========================================================================\n";
    
    DatasetType clientsDataset = DatasetType::concatDatasets(datasets_per_clients);
    
    if (debugFlags.debugCrc32ForDatasets && printAtTraining) {
        printInformationAboutDataset("concatenated dataset from clients", clientsDataset, 
                                                                          debugFlags.debugCrc32ForDatasets,
                                                                          debugFlags.debugPrintTrainDataset);
    }

    dopt::L2RegulirizeLogisticRegression singleObjective(clientsDataset.train_samples_tr, clientsDataset.train_outputs, args.optPrb.lambda);

    // algorithm selector: gd || fednl1
    bool is_algorithm_gd = (args.optAlgo.algorithm == "gd");
    bool is_algorithm_fednl1 = (args.optAlgo.algorithm == "fednl1");

    // options for fednl
    bool is_algorithm_fednl_has_option_a = cmdline.isFlagSetuped("fednl-option-a");
    bool is_algorithm_fednl_has_option_b = cmdline.isFlagSetuped("fednl-option-b");

    size_t iterations = 0;
    double mathMuf = singleObjective.computeMuStrongConvexity();
    double mathLf = -1.0;

    bool compute_L_smooth = cmdline.isFlagSetuped("compute-L-smooth") || args.optAlgo.has_theoretical_global_lr_flag;

    if (compute_L_smooth)
    {
        // L smooth constant need to be computed for varioation of GD with optimal step size
        mathLf = singleObjective.computeLSmoothness(0.001 * mathMuf, &iterations);
        if (printAtTraining)
            std::cout << "Compute L constants for objective (power iterations: " << iterations << ")" << timer.timeStamp() << '\n';
    }

    if (is_algorithm_fednl1)
    {
        if (!is_algorithm_fednl_has_option_a && !is_algorithm_fednl_has_option_b)
        {
            if (printAtTraining)
                std::cerr << "For FedNL you should specify fednl-option-a or fednl-option-b\n";
            return TrainReturnCodes::eMissingArgument;
        }
    }

    if (printAtTraining) {
        std::cout << "=========================================================================\n";
        std::cout << "Main Loop preparation for: " << args.optAlgo.algorithm << '\n';
        std::cout << "=========================================================================\n";
    }

    // Derived Quantitity from Specification

    // Dimension of Optimization Problem
    size_t d = trainset.numberOfAttributesForSamples();

    // Number of Rounds
    int rounds = args.runtime.rounds;

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
                
                // Alpha step size from theory for compressor 1/(w+1) * C(x)
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

    //===========================================================================================//
    // Our vector semantics is actually copying.
    VectorND_d xCur = x0;

    // For tracking algorithm behaviour
    VectorND_d::TElementType function_aggregation = VectorND_d::TElementType();

    // For actual algorithm behabiour
    VectorND_d prev_global_gradient_estimation(d);

    VectorND_d global_gradient_estimation(d);
    dopt::LightVectorND<VectorND_d> global_gradient_estimation_view(global_gradient_estimation.data(), d);

    VectorND_d true_gradient(d);
    //MatrixNMD_d S_in_place;

    // Need matrices for Newton System Solve. Not initialize for now.
    MatrixNMD_d H;
    MatrixNMD_d chol_l_factor_tr;

    double global_Li_fednl = 0.0;
    double prev_global_Li_fednl = 0.0;

    VectorND_d obtained_local_gradient(d);

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

    // Initialize need matrices for FedNL
    if (is_algorithm_fednl1)
    {
        H = MatrixNMD_d::getZeroSquareMatrix(d);
    }

    // Compute indicies once (possibly reconstruct)
    std::vector<uint32_t> indiciesOfUpperTriangularPart;
    
    if (!transfer_indicies_for_randk && (used_compressor == Compressor::eRandSeqK || used_compressor == Compressor::eRandK))
    {
        indiciesOfUpperTriangularPart = dopt::indiciesForUpperTriangularPart(H);
    }

    ServerContext serverContext;

    serverContext.rounds = rounds;
    serverContext.currenIterate = &xCur;
    serverContext.roundToStart = std::numeric_limits<size_t>::max();
    serverContext.clientPerRound = nSelectedClients;   
    serverContext.selectedClients = &selectedClients;

#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
    serverContext.startSempahore = std::move(dopt::DefaultSemaphore(0));
#endif

    for (int c = 0; c < nClients; ++c)
    {
        auto* clientOptProblem = new dopt::L2RegulirizeLogisticRegression(datasets_per_clients[c].train_samples_tr,
                                                                          datasets_per_clients[c].train_outputs,
                                                                          args.optPrb.lambda);

        if (is_algorithm_gd)
        {
            WorkerContextForGD* ctx = new WorkerContextForGD();
            ctx->workerIndex = c;
            ctx->optProblem = clientOptProblem;
            ctx->ctrBlock.resultIsReady = false;
            localWorkersContext.push_back(ctx);
        }
        else if (is_algorithm_fednl1)
        {
            WorkerContextForFedNL* ctx = new WorkerContextForFedNL();
            ctx->workerIndex = c;
            ctx->transfer_indicies_for_randk = transfer_indicies_for_randk;
            ctx->send_fi_from_worker = args.tracking.tracking_is_on;

            ctx->optProblem = clientOptProblem;

            ctx->ctrBlock.messageToSendGradientsIsReady = false;
            ctx->ctrBlock.messageToSendHessiansIsReady = false;
            ctx->ctrBlock.messageToSendLkIsReady = false;
            ctx->ctrBlock.messageToSendFiIsReady = false;
            ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = false;
            ctx->ctrBlock.messageToSendClientHasBeenInitialized = false;

            ctx->compressorType = used_compressor;
            ctx->kForCompressor = kForCompressor;
            ctx->seedForRandomizedCompressor = (123 + c); // hard coded seed for randomized compressor. TODO: If needed to make it more explicitly if needed.
            ctx->alpha = used_alpha_step_size;
            ctx->preScaleBeforeSend2Master = updateScalingInMaster;
            
            ctx->send_Lk_from_worker = (is_algorithm_fednl_has_option_b ? true : false);
            
            localWorkersContext.push_back(ctx);

            if (ctx->compressorType == Compressor::eRandK || ctx->compressorType == Compressor::eRandSeqK)
            {
                dopt::RandomGenIntegerLinear g;
                g.setSeed(ctx->seedForRandomizedCompressor);
                localWorkersRandomNumberGenerators.push_back(g);
            }
        }
        else
        {
            if (printAtTraining)
                std::cerr << " Please specify valid algorithm name.\n";

            return TrainReturnCodes::eInternalError;
        }
    }

    //====================================================================================================================
    // Static Distribution of work across workers
    std::vector<std::vector<WorkerContext*>> localWorkersContextPerWorker(nWorkers);
    {
        for (int w = 0; w < nWorkers; ++w)
            localWorkersContextPerWorker[w].reserve(workItemsPerClientUpperBound);

        for (int curClient = 0;;)
        {
            for (int w = 0; w < nWorkers; ++w)
            {
                localWorkersContextPerWorker[w].push_back(localWorkersContext[curClient]);
                assert(localWorkersContextPerWorker[w].size() <= workItemsPerClientUpperBound);

                curClient += 1;

                if (curClient == nClients)
                    break;
            }

            if (curClient == nClients)
                break;

        }
    }
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
        if (printAtTraining)
            std::cerr << " Please specify valid algorithm name.\n";
        return TrainReturnCodes::eInternalError;
    }
    // Create worker threads end
    //====================================================================================================================

    if (printAtTraining)
    {
        //std::cout << "!!" << H.frobeniusNormForSymmetricMatrixFromUpPart() << '\n';
        std::cout << "Main Loop preparation has been finished [OK]. " << timer.timeStamp() << '\n';
        std::cout << "=========================================================================\n";
        std::cout << "Main loop has been started: r:" << rounds << ",clients:" << nClients << ",d:" << d << ",ni:" << totalSamplesPerClient << '\n';
    }

    // placeholder for tracking results
    dopt::MutableData trackingData;
    
    double receivedBytesFromClientsForScalarsAccum  = 0.0;
    double receivedBytesFromClientsForIndiciesAccum = 0.0;

    double receivedBytesFromClientsForScalarsLast  = 0.0;
    double receivedBytesFromClientsForIndiciesLast = 0.0;

    if (args.tracking.tracking_is_on)
    {
        trackingData.putString(dopt::FileNameHelpers::extractBaseName(args.train_dataset.path), dopt::MutableData::PutStringFlags::ePutZeroTerminator);

        if (is_algorithm_fednl1)
        {
            trackingData.putString("FedNL-PP", dopt::MutableData::PutStringFlags::ePutNoTerminator);
            if (is_algorithm_fednl_has_option_a)
                trackingData.putString(" (a)", dopt::MutableData::PutStringFlags::ePutNoTerminator);
            if (is_algorithm_fednl_has_option_b)
                trackingData.putString(" (b)", dopt::MutableData::PutStringFlags::ePutNoTerminator);
        }
        else if (is_algorithm_gd)
        {
            trackingData.putString("GD", dopt::MutableData::PutStringFlags::ePutNoTerminator);
        }
        else
        {
            trackingData.putString("Unknown", dopt::MutableData::PutStringFlags::ePutNoTerminator);
        }
        trackingData.putByte(0);

        // trackingData.putString(args.optAlgo.algorithm, dopt::MutableData::PutStringFlags::ePutZeroTerminator);

        trackingData.putString(args.optAlgo.compressor, dopt::MutableData::PutStringFlags::ePutZeroTerminator);
        
        trackingData.putInt32(totalSamplesPerClient);
        trackingData.putInt32(totalSamples);
        trackingData.putInt32(rounds);
        trackingData.putInt32(nClients);
        trackingData.putInt32(d);
        trackingData.putInt32(kForCompressor);

        trackingData.putDouble(args.optAlgo.k_compressor_as_d_mult);
        trackingData.putDouble(used_global_step_size);
        trackingData.putDouble(used_alpha_step_size);
        trackingData.putDouble(args.optPrb.lambda);
    }
    timer.reset();

    // POOLS INITIALIZATION IN MASTER STRUCTS
    //===========================================================================================================
    struct GradientUpdateTask
    {
        WorkerContextForFedNL* wCtx = nullptr;
        dopt::LightVectorND<VectorND_d>* destinationsForGradient = nullptr;
    };
    auto processingRoutineForThPoolGradUpdate = [](const GradientUpdateTask& task, size_t thIndex) -> void
    {
        dopt::LightVectorND<VectorND_d> obtained_local_gradient_light((VectorND_d::TElementType*)task.wCtx->messageToSendGradients.getPtr(), task.wCtx->messageToSendGradients.getFilledSize() / sizeof(VectorND_d::TElementType));
        task.destinationsForGradient[thIndex] += obtained_local_gradient_light;
    };
    //===========================================================================================================
    struct HessianUpdateTask
    {
        WorkerContextForFedNL* wCtx = nullptr;                                        ///< Worker context
        const std::vector<uint32_t>* indiciesOfUpperTriangularPart = nullptr;         ///< Read only array
        
        dopt::RandomGenIntegerLinear* localWorkersRandomNumberGenerator = nullptr;    ///< Update only from this thread
        //MatrixNMD_d* S_in_place = nullptr;                                          ///< Parallel updates
        MatrixNMD_d* H_in_place = nullptr;                                            ///< Parallel updates
    };
    
    auto processingRoutineForThPoolHessianUpdate = [](const HessianUpdateTask& task, size_t thIndex) -> void
    {
        WorkerContextForFedNL* ctx = task.wCtx;
        Compressor used_compressor = ctx->compressorType;
        size_t d = ctx->optProblem->getInputVariableDimension();
        size_t kForCompressor = ctx->kForCompressor;
        bool transfer_indicies_for_randk = ctx->transfer_indicies_for_randk;
        
        // WARNING: memory in H in place is updated from several places
        MatrixNMD_d& H_in_place = *(task.H_in_place);
        
        //======================================================================================================
        dopt::Data messageHessianItems = dopt::Data(ctx->messageToSendHessiansItems.getPtr(), ctx->messageToSendHessiansItems.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::Data messageHessianIndicies = dopt::Data(ctx->messageToSendHessiansIndicies.getPtr(), ctx->messageToSendHessiansIndicies.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        //======================================================================================================
        
        if (used_compressor == Compressor::eIdentical)
        {
            for (size_t j = 0; j < d; ++j)
            {
                // messageHessianIndicies are not used in this mode
                for (size_t i = 0; i < j; ++i)
                {
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.getRaw(i, j), value);
                }

                // Diagonal element
                {
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.getRaw(j, j), value);
                }
            }
        }
        else if (used_compressor == Compressor::eNatural)
        {
            typedef dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, VectorND_d::TElementType> FPCompPack;
            FPCompPack packsBuffer[2];
            size_t componentsInBuffer = 0;

            uint32_t buffer = 0;

            for (size_t j = 0; j < d; ++j)
            {
                // messageHessianIndicies are not used in this mode
                for (size_t i = 0; i <= j; ++i)
                {
                    if (componentsInBuffer == 0)
                    {
                        messageHessianItems.getBytes(&buffer, 3);
                        dopt::unpack2FP64NoMantissa(packsBuffer, buffer);
                        componentsInBuffer = 2;
                    }

                    double value = packsBuffer[0].real_value_repr; // get 0-component
                    double updateValue = ctx->preScaleBeforeSend2Master * value;
                    dopt::appendMT(H_in_place.getRaw(i, j), updateValue);

                    packsBuffer[0] = packsBuffer[1];               // copy 1-component to 0-component
                    componentsInBuffer--;                          // decrease number of components
                }
            }
        }
        else if (used_compressor == Compressor::eTopK)
        {
            for (size_t i = 0; i < kForCompressor; ++i)
            {
                uint32_t index = messageHessianIndicies.getUint32();
                double value = messageHessianItems.getDouble();
                dopt::appendMT(H_in_place.matrixByCols[index], value);                
            }
        }
        else if (used_compressor == Compressor::eTopLEK)
        {
            int32_t kForCompressorInThisRound = messageHessianIndicies.getUint32();

            for (size_t i = 0; i < kForCompressorInThisRound; ++i)
            {
                uint32_t index = messageHessianIndicies.getUint32();
                double value = messageHessianItems.getDouble();
                dopt::appendMT(H_in_place.matrixByCols[index], value);
            }
        }
        else if (used_compressor == Compressor::eRandK)
        {
            std::vector<uint32_t> kcoordinatesReconstruct;

            if (!transfer_indicies_for_randk)
            {
                const std::vector<uint32_t>& indiciesOfUpperTriangularPart = *(task.indiciesOfUpperTriangularPart);
                dopt::RandomGenIntegerLinear& localWorkersRandomNumberGenerator = *(task.localWorkersRandomNumberGenerator);
                
                // No index transferring from client. We reconstruct indicies on the server.
                kcoordinatesReconstruct = dopt::generateRandKItemsInUpperTriangularPart(localWorkersRandomNumberGenerator,
                                                                                        kForCompressor,
                                                                                        H_in_place,
                                                                                        indiciesOfUpperTriangularPart);

                for (size_t i = 0; i < kForCompressor; ++i)
                {
                    uint32_t index = kcoordinatesReconstruct[i];
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);                    
                }
            }
            else
            {
                for (size_t i = 0; i < kForCompressor; ++i)
                {
                    uint32_t index = messageHessianIndicies.getUint32();
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
        }
        else if (used_compressor == Compressor::eRandSeqK)
        {
            std::vector<uint32_t> kcoordinatesReconstruct;

            if (!transfer_indicies_for_randk)
            {
                const std::vector<uint32_t>& indiciesOfUpperTriangularPart = *(task.indiciesOfUpperTriangularPart);
                dopt::RandomGenIntegerLinear& localWorkersRandomNumberGenerator = *(task.localWorkersRandomNumberGenerator);
                
                // No index transferring from client. We reconstruct indicies on the server.
                kcoordinatesReconstruct = dopt::generateRandSeqKItemsInUpperTriangularPart(localWorkersRandomNumberGenerator,
                                                                                           kForCompressor,
                                                                                           H_in_place,
                                                                                           indiciesOfUpperTriangularPart);

                for (size_t i = 0; i < kForCompressor; ++i)
                {
                    uint32_t index = kcoordinatesReconstruct[i];
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
            else
            {
                for (size_t i = 0; i < kForCompressor; ++i)
                {
                    uint32_t index = messageHessianIndicies.getUint32();
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
        }        
    };
    
    // POOLS INITIALIZATION IN MASTER START
    //===========================================================================================================
    const size_t kUseThreadPoolSize4Grad = args.runtime.nServerGradWorkers;
    const size_t kUseThreadPoolSize4Hessians = args.runtime.nServerHessianWorkers;

    dopt::ThreadPoolWithTaskQueue<GradientUpdateTask> threadPoolInMaster4GradUp(processingRoutineForThPoolGradUpdate, kUseThreadPoolSize4Grad, nClients);
    dopt::ThreadPoolWithTaskQueue<HessianUpdateTask> threadPoolInMaster4HessiansUp(processingRoutineForThPoolHessianUpdate, kUseThreadPoolSize4Hessians, nClients);

    std::vector<VectorND_d> gradientInMasterInPool;
    std::vector<dopt::LightVectorND<VectorND_d>> gradientInMasterInPoolView;
    
    gradientInMasterInPool.reserve(kUseThreadPoolSize4Grad);
    gradientInMasterInPoolView.reserve(kUseThreadPoolSize4Grad);
    for (size_t k = 0; k < kUseThreadPoolSize4Grad; ++k)
    {
        gradientInMasterInPool.emplace_back(VectorND_d(d));
        gradientInMasterInPoolView.emplace_back(dopt::LightVectorND<VectorND_d>(gradientInMasterInPool.back(), 0));
    }
    
    threadPoolInMaster4GradUp.signalResumeProcessing();
    threadPoolInMaster4HessiansUp.signalResumeProcessing();
    
    //===========================================================================================================
    // POOLS INITIALIZATION IN MASTER END
    
    {
        if (is_algorithm_fednl1)
        {
            for (int c = 0; c < nClients; ++c)
            {
                WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];
                while (!ctx->ctrBlock.messageToSendClientHasBeenInitialized)
                    dopt::DefaultThread::yeildCurrentThInHotLoop();
                
                H += ctx->learningHessian;
                prev_global_gradient_estimation += ctx->localGradient;
                prev_global_Li_fednl += ctx->localLk;
            }
            
            H *= inv_NClients;
            prev_global_gradient_estimation *= inv_NClients;
            prev_global_Li_fednl *= inv_NClients;            
        }

        double xCurL2Norm = 0.0;
        
        for (int r = 0; r < rounds; ++r)
        {
            if (is_algorithm_gd)
            {
                // GD

                // Start new round
#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
                serverContext.startSempahore.release(nWorkers);
#endif

                serverContext.roundToStart = r;

                xCurL2Norm = xCur.vectorL2Norm();
                
                true_gradient = computeFullGradient( localWorkersContext, *(serverContext.currenIterate) );

                // Reset information about gradient and about function
                global_gradient_estimation.setAllToDefault();

                // Reset function aggregation
                function_aggregation = VectorND_d::TElementType();

                int waitlist_with_grad_info = nClients;
                int waitlist_with_function_info = args.tracking.tracking_is_on ? nClients : 0;

                while (waitlist_with_grad_info >= 0 || waitlist_with_function_info >= 0)
                {
                    for (int c = 0; c < nClients; ++c)
                    {
                        WorkerContextForGD* ctx = (WorkerContextForGD*)localWorkersContext[c];

                        if (dopt::checkAndResetIfSet(&ctx->ctrBlock.resultIsReady))
                        {
                            waitlist_with_grad_info -= 1;
                            global_gradient_estimation += ctx->localGradient;
                            receivedBytesFromClientsForScalarsLast += global_gradient_estimation.sizeInBytes();

                            if (args.tracking.tracking_is_on)
                            {
                                // Make function aggregation (if needed)
                                function_aggregation += ctx->optProblem->evaluateFunction(xCur);
                                waitlist_with_function_info -= 1;
                                function_aggregation += sizeof(function_aggregation);
                            }
                        }
                    }

                    if (waitlist_with_grad_info == 0)
                    {
                        global_gradient_estimation *= inv_NClients;
                        waitlist_with_grad_info--;
                    }

                    if (waitlist_with_function_info == 0)
                    {
                        function_aggregation *= inv_NClients;
                        waitlist_with_function_info--;
                    }
                }

                //xCur -= used_global_step_size * global_gradient_estimation;
                xCur.subInPlaceVectorWithMultiple(used_global_step_size, global_gradient_estimation);
            }

            if (is_algorithm_fednl1)
            {
                // FedNL

                // Start new round

                // Select clients for FedNL-PP
                *(serverContext.selectedClients) = allClients;
                dopt::shuffle(*serverContext.selectedClients, serverContext.clientPerRound, clientSampler);
                
                serverContext.roundToStart = r;

#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
                serverContext.startSempahore.release(nWorkers);
#endif
                xCurL2Norm = xCur.vectorL2Norm();

                true_gradient = computeFullGradient(localWorkersContext, *(serverContext.currenIterate));

                ///< Reset information about gradient and about function
                global_gradient_estimation.setAllToDefault();
                for (size_t k = 0; k < kUseThreadPoolSize4Grad; ++k) {
                    gradientInMasterInPool[k].setAllToDefault();
                }

                ///< Reset global Li
                global_Li_fednl = 0.0;

                // Reset function aggregation
                function_aggregation = VectorND_d::TElementType();

                int waitlist_with_grad_info = nSelectedClients;
                int waitlist_with_shifted_hessian_info = nSelectedClients;
                int waitlist_with_Lk_info = is_algorithm_fednl_has_option_b ? nSelectedClients : -1;

                //===============================================================================
                // Debug mode: wait for information from all clients: grad, hessians, LK
                //===============================================================================
                if (debugFlags.debugForceSequentialUpdate)
                    waitForInformationFromClientsForFedNL(localWorkersContext, &serverContext, is_algorithm_fednl_has_option_b);

                //===============================================================================
                
                while (waitlist_with_grad_info >= 0 || waitlist_with_shifted_hessian_info >= 0 || waitlist_with_Lk_info >= 0)
                {
                    for (int c = 0; c < nClients; ++c)
                    {
                        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];

                        if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendGradientsIsReady))
                        {
                            waitlist_with_grad_info -= 1;

                            receivedBytesFromClientsForScalarsLast += ctx->messageToSendGradients.getFilledSize();

                            assert(ctx->messageToSendGradients.getFilledSize() == d * sizeof(double));
                            assert(obtained_local_gradient.sizeInBytes() == d * sizeof(double));
                            
                            if (kUseThreadPoolSize4Grad > 0)
                            {
                                GradientUpdateTask task;                                
                                task.wCtx = ctx;
                                task.destinationsForGradient = gradientInMasterInPoolView.data();
                                threadPoolInMaster4GradUp.addTask(task);
                            }
                            else
                            {
                                dopt::LightVectorND<VectorND_d> obtained_local_gradient_light((VectorND_d::TElementType*)ctx->messageToSendGradients.getPtr(),
                                                                                              ctx->messageToSendGradients.getFilledSize() / sizeof(VectorND_d::TElementType)
                                                                                              );

                                global_gradient_estimation_view += obtained_local_gradient_light;
                            }
                        }

                        if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendHessiansIsReady))
                        {
                            waitlist_with_shifted_hessian_info -= 1;

                            // Real number of bytes that need to be send
                            receivedBytesFromClientsForScalarsLast  += ctx->messageToSendHessiansItems.getFilledSize();
                            receivedBytesFromClientsForIndiciesLast += ctx->messageToSendHessiansIndicies.getFilledSize();

                            if (kUseThreadPoolSize4Hessians > 0)
                            {
                                HessianUpdateTask task;
                                task.wCtx = ctx;
                                task.indiciesOfUpperTriangularPart = &indiciesOfUpperTriangularPart;

                                task.localWorkersRandomNumberGenerator = localWorkersRandomNumberGenerators.empty() ? nullptr : &localWorkersRandomNumberGenerators[c];
                                task.H_in_place = &H;
                                threadPoolInMaster4HessiansUp.addTask(task);
                            }
                            else
                            {
                                // Unpack the results: start
                                dopt::Data messageHessianItems = dopt::Data(ctx->messageToSendHessiansItems.getPtr(), ctx->messageToSendHessiansItems.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
                                dopt::Data messageHessianIndicies = dopt::Data(ctx->messageToSendHessiansIndicies.getPtr(), ctx->messageToSendHessiansIndicies.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

                                if (used_compressor == Compressor::eIdentical)
                                {
                                    for (size_t j = 0; j < d; ++j)
                                    {
                                        // messageHessianIndicies are not used in this mode
                                        for (size_t i = 0; i < j; ++i)
                                        {
                                            double value = messageHessianItems.getDouble();
                                            H.getRaw(i, j) += value;                                            
                                        }

                                        // Diagonal element
                                        //for (size_t i = j; i <= j; ++i)
                                        {
                                            double value = messageHessianItems.getDouble();
                                            H.getRaw(j, j) += value;
                                        }
                                    }
                                }
                                else if (used_compressor == Compressor::eNatural)
                                {
                                    uint32_t buffer = 0;
                                    dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, VectorND_d::TElementType> packsBuffer[2];
                                    size_t componentsInBuffer = 0;

                                    for (size_t j = 0; j < d; ++j)
                                    {
                                        // messageHessianIndicies are not used in this mode
                                        for (size_t i = 0; i <= j; ++i)
                                        {
                                            if (componentsInBuffer == 0)
                                            {
                                                messageHessianItems.getBytes(&buffer, 3);
                                                dopt::unpack2FP64NoMantissa(packsBuffer, buffer);
                                                componentsInBuffer = 2;
                                            }

                                            double value = packsBuffer[0].real_value_repr; // get 0-component
                                            double updateValue = ctx->preScaleBeforeSend2Master * value;

                                            H.getRaw(i, j) += updateValue;

                                            packsBuffer[0] = packsBuffer[1];               // copy 1-component to 0-component
                                            componentsInBuffer--;                          // decrease number of components
                                        }
                                    }
                                }
                                else if (used_compressor == Compressor::eTopK)
                                {
                                    for (size_t i = 0; i < kForCompressor; ++i)
                                    {
                                        uint32_t index = messageHessianIndicies.getUint32();
                                        double value = messageHessianItems.getDouble();
                                        H.matrixByCols[index] += value;
                                    }
                                }
                                else if (used_compressor == Compressor::eTopLEK)
                                {
                                    int32_t kForCompressorInThisRound = messageHessianIndicies.getUint32();

                                    for (size_t i = 0; i < kForCompressorInThisRound; ++i)
                                    {
                                        uint32_t index = messageHessianIndicies.getUint32();
                                        double value = messageHessianItems.getDouble();
                                        H.matrixByCols[index] += value;
                                    }
                                }
                                else if (used_compressor == Compressor::eRandK)
                                {
                                    std::vector<uint32_t> kcoordinatesReconstruct;

                                    if (!transfer_indicies_for_randk)
                                    {
                                        // No index transferring from client. We reconstruct indicies on the server.
                                        kcoordinatesReconstruct = dopt::generateRandKItemsInUpperTriangularPart(localWorkersRandomNumberGenerators[c],
                                                                                                                kForCompressor,
                                                                                                                H,
                                                                                                                indiciesOfUpperTriangularPart);

                                        for (size_t i = 0; i < kForCompressor; ++i)
                                        {
                                            uint32_t index = kcoordinatesReconstruct[i];
                                            double value = messageHessianItems.getDouble();
                                            H.matrixByCols[index] += value;
                                        }
                                    }
                                    else
                                    {
                                        for (size_t i = 0; i < kForCompressor; ++i)
                                        {
                                            uint32_t index = messageHessianIndicies.getUint32();
                                            double value = messageHessianItems.getDouble();
                                            H.matrixByCols[index] += value;
                                        }
                                    }
                                }
                                else if (used_compressor == Compressor::eRandSeqK)
                                {
                                    std::vector<uint32_t> kcoordinatesReconstruct;

                                    if (!transfer_indicies_for_randk)
                                    {
                                        // No index transferring from client. We reconstruct indicies on the server.
                                        kcoordinatesReconstruct = dopt::generateRandSeqKItemsInUpperTriangularPart(localWorkersRandomNumberGenerators[c],
                                                                                                                   kForCompressor,
                                                                                                                   H,
                                                                                                                   indiciesOfUpperTriangularPart);

                                        for (size_t i = 0; i < kForCompressor; ++i)
                                        {
                                            uint32_t index = kcoordinatesReconstruct[i];
                                            double value = messageHessianItems.getDouble();
                                            H.matrixByCols[index] += value;                                            
                                        }
                                    }
                                    else
                                    {
                                        for (size_t i = 0; i < kForCompressor; ++i)
                                        {
                                            uint32_t index = messageHessianIndicies.getUint32();
                                            double value = messageHessianItems.getDouble();                                            
                                            H.matrixByCols[index] += value;
                                        }
                                    }
                                }
                            }
                        }

                        if (is_algorithm_fednl_has_option_b)
                        {
                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendLkIsReady))
                            {
                                waitlist_with_Lk_info -= 1;
                                dopt::Data message = dopt::Data(ctx->messageToSendLk.getPtr(), ctx->messageToSendLk.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
                                double Li_fednl = message.getDouble();
                                global_Li_fednl += Li_fednl;

                                receivedBytesFromClientsForScalarsLast += ctx->messageToSendLk.getFilledSize();
                            }
                        }
                    }

                    if (waitlist_with_grad_info == 0)
                    {
                        // All workers report their gradient information. Make waitlist negative to indicate that we can leave the loop.
                        
                        if (kUseThreadPoolSize4Grad > 0)
                        {
                            if (threadPoolInMaster4GradUp.isWaitingTasks() == false)
                            {
                                threadPoolInMaster4GradUp.waitForCurrentJobsCompletion();

                                for (size_t k = 0; k < kUseThreadPoolSize4Grad; ++k)
                                {
                                    global_gradient_estimation += gradientInMasterInPool[k];
                                }
                                
                                global_gradient_estimation *= inv_NClients;

                                // Correctness for FEDNL-PP mode [START]
                                global_gradient_estimation += prev_global_gradient_estimation;
                                prev_global_gradient_estimation = global_gradient_estimation;
                                // Correctness for FEDNL-PP mode [END]

                                waitlist_with_grad_info -= 1;
                            }
                            else
                            {
                                // Do nothing. Try next time.
                            }
                        }
                        else
                        {
                            global_gradient_estimation *= inv_NClients;
                            
                            // Correctness for FEDNL-PP mode [START]
                            global_gradient_estimation += prev_global_gradient_estimation;
                            prev_global_gradient_estimation = global_gradient_estimation;
                            // Correctness for FEDNL-PP mode [END]

                            waitlist_with_grad_info -= 1;
                        }
                    }

                    if (waitlist_with_shifted_hessian_info == 0)
                    {                        
                        waitlist_with_shifted_hessian_info -= 1;
                    }

                    if (waitlist_with_Lk_info == 0)
                    {
                        // Average in master (Line 20 in FedNL-PP)
                        global_Li_fednl *= inv_NClients;                       
                        // Add shift
                        global_Li_fednl += prev_global_Li_fednl;
                        // Update
                        prev_global_Li_fednl = global_Li_fednl;

                        waitlist_with_Lk_info -= 1;
                    }
                }
                
                //=========================================================================//
#if 0
                // [DEBUG]
                if (r < rounds - 1)
                {
                    double Lk_avg = computeLkAverage(localWorkersContext, &serverContext);
                    VectorND_d local_g = computeGdirectionAverage(localWorkersContext, &serverContext);
                    double dicr_Li = Lk_avg - global_Li_fednl;
                    double dicr_gr_1 = (local_g - global_gradient_estimation).vectorL2Norm();
                    assert(dicr_Li < 1e-9);
                    assert(dicr_gr_1 < 1e-9);
                }
#endif
                //=========================================================================//
                if (kTrackHessianDifference)
                {
                    MatrixNMD_d LL = MatrixNMD_d::getZeroSquareMatrix(d);
                    size_t finished = 0;

                    do
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];

                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished))
                            {
                                LL += ctx->learningHessian;
                                finished++;
                            }
                        }
                    } while (finished != nClients);

                    LL *= inv_NClients;

                    if (printAtTraining)
                        std::cout << "CONSISTENCY FOR LEARNED HESSIAN DEBUG [SHOULD BE NEAR ZERO/DECAY]: " << (H - LL).frobeniusNormForSymmetricMatrixFromUpPart() << '\n';
                }                
                else
                {
                    //=========================================================================//
                    // WAIT FOR ALL WORKERS TO FINISH THIS ROUND
                    //=========================================================================//
                    size_t finished = 0;
                    do
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];

                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished))
                            {
                                finished++;
                            }
                        }
                    } while (finished != nClients);
                }

                //=========================================================================//
                if (is_algorithm_fednl_has_option_a)
                {
                    constexpr int algorithmNumberForSystem = 2;

                    if (algorithmNumberForSystem == 1)
                    {
                        // Pretty lazy projection into mathMuf conde of PSD matrices
                        MatrixNMD_d matrixOfSlau = H;
                        matrixOfSlau.addToAllDiagonalEntries(mathMuf);
                        matrixOfSlau.symmetrizeLowerTriangInPlace();

                        xCur = dopt::gausEleminationSolver(matrixOfSlau, global_gradient_estimation);

                        if constexpr (kDebugOutputForSolve)
                        {
                            if (printAtTraining)
                                std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                        }
                    }
                    else if (algorithmNumberForSystem == 2)
                    {
                        //std::cout << "global_Li_fednl " << global_Li_fednl << '\n';
                        //std::cout << "debug H frob.norm: " << H.frobeniusNormForSymmetricMatrixFromUpPart() << '\n';

                        bool sstFactorization = dopt::cholFactorization::choleskyFactorization<decltype(chol_l_factor_tr),
                                                                                               true /*fill_Lfactor*/,
                                                                                               true /*in_input_symmetric*/>
                                                                                                (chol_l_factor_tr, H, mathMuf);

                        assert(sstFactorization == true);

                        if (sstFactorization)
                        {                            
                            // (L * transpose(L))dx = matrixOfSlau*dx = b"
                            // => transpose(L) dx = (L^-1) b
                            // => transpose(L) dx = ( (L_tr)^-1) b, but with flag that we provide L_tr such L_tr = transpose(L)
                            VectorND_d Ltr_dx = dopt::forwardSubstitutionWithATranspose(chol_l_factor_tr, global_gradient_estimation);

                            // Second step transpose(L) dx = [(L^-1) b]
                            // => dx = (transpose(L)) ^ {-1} * [(L^-1) b]
                            // => dx = (L) ^ {-1} * [(L^-1) b], with flag that we provide L such L = transpose(L_tr)                            
                            xCur = dopt::backwardSubstitutionWithATranspose(chol_l_factor_tr, Ltr_dx);

                            if constexpr (kDebugOutputForSolve)
                            {
                                MatrixNMD_d matrixOfSlau = H;
                                matrixOfSlau.addToAllDiagonalEntries(mathMuf);

                                if (printAtTraining)
                                    std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                            }
                        }
                        else
                        {
                            if (printAtTraining)
                                std::cout << "SST FACTORIZATION FAILED\n";

                            MatrixNMD_d matrixOfSlau = H;
                            matrixOfSlau.addToAllDiagonalEntries(mathMuf);

                            xCur = dopt::gausEleminationSolver(matrixOfSlau, global_gradient_estimation);
                            
                            if constexpr (kDebugOutputForSolve)
                            {
                                if (printAtTraining)
                                    std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                            }
                        }
                    }
                }
                else if (is_algorithm_fednl_has_option_b)
                {
                    constexpr int algorithmNumberForSystem = 2;

                    if (algorithmNumberForSystem == 1)
                    {
                        MatrixNMD_d matrixOfSlau = H;
                        matrixOfSlau.addToAllDiagonalEntries(global_Li_fednl);
                        xCur = dopt::gausEleminationSolver(matrixOfSlau, global_gradient_estimation);

                        if constexpr (kDebugOutputForSolve)
                        {
                            if (printAtTraining)
                                std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                        }
                    }
                    else if (algorithmNumberForSystem == 2)
                    {
                        //std::cout << "global_Li_fednl " << global_Li_fednl << '\n';
                        //std::cout << "debug H frob.norm: " << H.frobeniusNormForSymmetricMatrixFromUpPart() << '\n';

                        bool sstFactorization = dopt::cholFactorization::choleskyFactorization<decltype(chol_l_factor_tr),
                                                                                                true /*fill_Lfactor*/,
                                                                                                true /*in_input_symmetric*/>
                                                                                                (chol_l_factor_tr, H, global_Li_fednl);

                        assert(sstFactorization == true);

                        if (sstFactorization)
                        {
                            // (L * transpose(L))dx = matrixOfSlau*dx = b"
                            // => transpose(L) dx = (L^-1) b
                            // => transpose(L) dx = ( (L_tr)^-1) b, but with flag that we provide L_tr such L_tr = transpose(L)
                            VectorND_d Ltr_dx = dopt::forwardSubstitutionWithATranspose(chol_l_factor_tr, global_gradient_estimation);

                            // Second step transpose(L) dx = [(L^-1) b]
                            // => dx = (transpose(L)) ^ {-1} * [(L^-1) b]
                            // => dx = (L') ^ {-1} * [(L^-1) b], with flag that we provide L' such L' = transpose(L_tr)
                            xCur = dopt::backwardSubstitutionWithATranspose(chol_l_factor_tr, Ltr_dx);

                            if constexpr (kDebugOutputForSolve)
                            {
                                MatrixNMD_d matrixOfSlau = H;
                                matrixOfSlau.addToAllDiagonalEntries(global_Li_fednl);

                                if (printAtTraining)
                                    std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                            }
                        }
                        else
                        {
                            if (printAtTraining)
                                std::cout << "SST FACTORIZATION FAILED\n";

                            MatrixNMD_d matrixOfSlau = H;
                            matrixOfSlau.addToAllDiagonalEntries(global_Li_fednl);
                            xCur = dopt::gausEleminationSolver(matrixOfSlau, global_gradient_estimation);

                            if constexpr (kDebugOutputForSolve)
                            {
                                if (printAtTraining)
                                    std::cout << "SOLVE DISCREPANCY:" << (matrixOfSlau * xCur - global_gradient_estimation).vectorL2Norm() << "!!\n";
                            }
                        }
                    }
                }
                else
                {
                    if (printAtTraining)
                        std::cout << " GD step for FEDNL\n";
                    xCur = used_global_step_size * global_gradient_estimation;
                }
            }

            // Function aggregation if needed
            if (args.tracking.tracking_is_on)
            {
                int waitlist_with_function_info = nSelectedClients;

                while (waitlist_with_function_info >= 0)
                {
                    for (int c = 0; c < nClients; ++c)
                    {
                        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];

                        if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendFiIsReady))
                        {
                            waitlist_with_function_info -= 1;

                            // Unpack the results: start
                            dopt::Data message = dopt::Data(ctx->messageToSendFi.getPtr(),
                                                            ctx->messageToSendFi.getFilledSize(),
                                                            dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

                            double fi = message.getDouble();
                            function_aggregation += fi;
                            receivedBytesFromClientsForScalarsLast += ctx->messageToSendLk.getFilledSize();
                        }
                    }

                    if (waitlist_with_function_info == 0)
                    {
                        // All workers report their gradient information
                        function_aggregation *= inv_NClients;
                        waitlist_with_function_info -= 1;
                    }
                }
            }
            
            // Report norm of full gradient
            if (!silentRun && printAtTraining)
            {
                std::cout << "  Round #" << r
                          << ": norm of full gradient: " << true_gradient.vectorL2Norm()
                          << ": f(x): " << function_aggregation
                          << ": avg. time per round: " << timer.getTimeMs() / double(r + 1) << " milliseconds"
                    << '\n';
            }

            if (debugFlags.debugMemInfo && printAtTraining)
                printMemoryInformation();

            if (args.tracking.tracking_is_on)
            {
                trackingData.putInt32(r);

                trackingData.putDouble(true_gradient.vectorL2Norm());
                trackingData.putDouble(function_aggregation);
                trackingData.putDouble(xCurL2Norm);

                trackingData.putDouble(receivedBytesFromClientsForScalarsAccum);
                trackingData.putDouble(receivedBytesFromClientsForIndiciesAccum);

                trackingData.putDouble(timer.getTimeSec());
            }

            receivedBytesFromClientsForScalarsAccum += receivedBytesFromClientsForScalarsLast;
            receivedBytesFromClientsForIndiciesAccum += receivedBytesFromClientsForIndiciesLast;

            receivedBytesFromClientsForScalarsLast = 0.0;
            receivedBytesFromClientsForIndiciesLast = 0.0;            
        }
    }

    if (printAtTraining) {
        std::cout << "=========================================================================\n";
        std::cout << "Main loop has been finished: r:" << rounds <<
                     ",clients:" << nClients <<
                     ",d:" << d <<
                     ",ni:" << totalSamplesPerClient << '\n';
    }

    double elapsedTime = timer.getTimeMs();

    if (printAtTraining) {
        std::cout << timer.timeStamp() << '\n';
        std::cout << "=========================================================================\n";
        std::cout << "Statistics\n\n";
        std::cout << "  Average time per round: " << elapsedTime / rounds << " milliseconds\n";
        std::cout << "  Total number of rounds: " << rounds << '\n';
        std::cout << "  Total number of clients: " << nClients << '\n';
        std::cout << "  Number of clients per round selected u.a.r: " << nSelectedClients << '\n';
        std::cout << '\n';
        std::cout << "  Total number of workers for clients: " << args.runtime.nWorkers << '\n';
        std::cout << "  Total number of workers for master [for gradient updates]: " << args.runtime.nServerGradWorkers << '\n';
        std::cout << "  Total number of workers for master [for hessian updates]: " << args.runtime.nServerHessianWorkers << '\n';
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
        std::cout << "  Received MBytes from the clients: " << (receivedBytesFromClientsForScalarsAccum + receivedBytesFromClientsForIndiciesAccum) / (1024.0 * 1024.0) << '\n';
        std::cout << "  Clients compressor: " << args.optAlgo.compressor << '\n';
        std::cout << "  Clients compressor K: " << kForCompressor << " / Hessian shape is [d,d] where d: " << d
                  << " / Maximum K ( {d(d+1)}/2 ): " << (d * (d + 1)) / 2 << '\n';
        std::cout << '\n';
        std::cout << "  Last iterate CRC-32: " << dopt::crc32(xCur.rawData(), xCur.sizeInBytes(), dopt::crc32Seed()) << '\n';

        std::cout << "  Gradient in last iterate CRC-32: " << dopt::crc32(true_gradient.rawData(), true_gradient.sizeInBytes(), dopt::crc32Seed()) << '\n';
    std::cout << "  Norm of last iterate: " << xCur.vectorL2Norm() << '\n';

    // std::cout << "  Estimate of norm of full gradient in last iterate: " << global_gradient_estimation.vectorL2Norm() << '\n';
        std::cout << "  True of norm of full gradient in last iterate: " << true_gradient.vectorL2Norm() << '\n';

    
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

    threadPoolInMaster4GradUp.terminate();
    threadPoolInMaster4HessiansUp.terminate();

    assert(localWorkers.size() == nWorkers);

    for (int c = 0; c < nWorkers; ++c)
    {
        localWorkers[c]->join();
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

    // replace run id, algorithm, dataset
    std::string tracking_output_path = std::string(args.tracking.output_binary);
    {
        constexpr int kStringToReplace = 4;
        
        std::string strs_to_replace[kStringToReplace] = { "{run_id}", "{algorithm}", "{dataset}", "{kmultiplier}" };
        
        std::string content_to_replace[kStringToReplace] = { dopt::string_utils::toString(runId), 
                                                             args.optAlgo.algorithm, 
                                                             dopt::FileNameHelpers::extractBaseName(args.train_dataset.path), 
                                                             dopt::string_utils::toString(args.optAlgo.k_compressor_as_d_mult) 
                                                           };

        for (size_t i = 0; i < kStringToReplace; ++i)
        {
            auto pos = tracking_output_path.find(strs_to_replace[i]);
            if (pos != std::string::npos)
                tracking_output_path.replace(pos, strs_to_replace[i].size(), content_to_replace[i]);
        }
    }

    if (printAtTraining)
        std::cout << "Results\n\n";
    
    if (!args.tracking.output_binary.empty())
    {
        bool tracking_saved = dopt::FileSystemHelpers::saveFile(tracking_output_path, trackingData.getPtr(), trackingData.getFilledSize());

        if (printAtTraining)
        {
            std::cout << "  Tracking results has been saved to: "
                      << tracking_output_path
                      << (tracking_saved ? " [OK]" : " [FAILED]")
                      << '\n';
        }
    }

    if (printAtTraining)
    {
        std::cout << "=========================================================================\n";
    }

    if (resultCallback) {
        resultCallback(xCur.size());
    }

    return TrainReturnCodes::eOk;
}
