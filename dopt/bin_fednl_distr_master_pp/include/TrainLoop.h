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

// Numerical math
#include "dopt/linalg_matrices/include/factorization/CholeskyFactorization.h"
#include "dopt/linalg_linsolvers/include/GaussEliminationSolvers.h"
#include "dopt/linalg_linsolvers/include/ElementarySolvers.h"

// Special methods for generate and get TopK indicies
#include "dopt/math_routines/include/SpecialMathRoutinesForMatrix.h"

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
#include "TrainCallbaksSharedMasterPP.h"

#include <unordered_map>
#include <map>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

inline void logComSpeed(double serverRequestTimeStampMs, double serverResponseTimeStampMs, int client, const char* message, size_t transferedBytes)
{
    const double communicatedBits = 8.0 * double(transferedBytes);
    const double eclapsedMsec = serverResponseTimeStampMs - serverRequestTimeStampMs;
    
    // [MBits]/[Sec] = ([Bits]/[Sec]) / 1e6 = ([Bits]/[MSec/1000]) / 1e6 = ([Bits]/[mSec]) * 1e3 / 1e6 = ([Bits]/[mSec]) / 1e3
    // [KBits]/[Sec] = ([Bits]/[Sec]) / 1e3 = ([Bits]/[MSec/1000]) / 1e3 = ([Bits]/[mSec]) * 1e3 / 1e3 = ([Bits]/[mSec])
    double comSpeedKBitsSec = (communicatedBits/ eclapsedMsec);
    double comSpeedMBitsSec = (communicatedBits / eclapsedMsec) / 1e3;

    if (comSpeedKBitsSec > 1000.0)
    {
        std::cout << "  Client " << client << " " << message 
                  << ": " << comSpeedMBitsSec << " Mbit/s (comp|comm)" 
                  << ": elapsed " << (serverResponseTimeStampMs - serverRequestTimeStampMs) << " msec" << '\n';
    }
    else
    {
        std::cout << "  Client " << client << " " << message
                  << ": " << comSpeedKBitsSec << " kbit/s (comp|comm)"
                  << ": elapsed " << (serverResponseTimeStampMs - serverRequestTimeStampMs) << " msec" << '\n';

    }
}

inline VectorND_d computeFullGradient(const std::vector<WorkerContext*>& localWorkersContext, 
                                      ServerContext& serverContext, 
                                      const VectorND_d& x)
{
    VectorND_d result = VectorND_d(x.size());
    int nClients = localWorkersContext.size();


    // Request initial g^0
    {
        dopt::MutableData buffer;
        serverContext.prepareUpdate(buffer, ServerContext::sig_request_full_gradient_in_client);

        for (size_t c = 0; c < nClients; ++c)
        {
            WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];
            dopt::Socket& socketAux = *(wctx->control);

            socketAux.sendData(buffer.getPtr(), buffer.getFilledSize());
        }

        for (size_t c = 0; c < nClients; ++c)
        {
            WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];
            dopt::Socket& socketAux = *(wctx->control);

            uint8_t ctrl = 0;
            socketAux.recvData(&ctrl, sizeof(ctrl));
            assert(ctrl == ServerContext::ControlSignals::sig_response_full_gradient_in_client);

            VectorND_d g_local = VectorND_d(x.size());
            size_t szInBytes = socketAux.getUnsignedVaryingInteger();
            assert(szInBytes == g_local.sizeInBytes());
            socketAux.recvData(g_local.data(), g_local.sizeInBytes());
            result += g_local;
        }
    }

    return result / double(nClients);
}

inline void waitForInformationFromClientsForFedNL(const std::vector<WorkerContext*>& localWorkersContext, 
                                                  const ServerContext* ctx,
                                                  bool is_algorithm_fednl_has_option_b)
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

        for (;;)
        {
            if (wctx->ctrBlock.messageToSendGradientsIsReady)
            {
                break;
            }
            else
            {
                wctx->receiveUpdates</*block*/true> (wctx->control);
            }
        }
    }

    // Wait for hessians
    for (size_t c = 0; c < nClients; ++c)
    {
        WorkerContextForFedNL* wctx = (WorkerContextForFedNL*)localWorkersContext[c];

        if (!clientHasBeenSelected(c, ctx))
        {
            assert(wctx->ctrBlock.messageToSendHessiansIndiciesIsReady == false && wctx->ctrBlock.messageToSendHessiansItemsIsReady);
            continue;
        }

        for (;;)
        {
            if (wctx->ctrBlock.messageToSendHessiansIndiciesIsReady && wctx->ctrBlock.messageToSendHessiansItemsIsReady)
            {
                break;
            }
            else
            {
                wctx->receiveUpdates</*block*/true>(wctx->control);
            }
            
            break;            
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

            for (;;)
            {
                if (wctx->ctrBlock.messageToSendLkIsReady)
                {
                    break;
                }
                else
                {
                    wctx->receiveUpdates</*block*/true>(wctx->control);
                }                
            }
        }
    }
}

inline TrainReturnCodes train(const dopt::CmdLine& cmdline, ResultCallbackMasterPP resultCallback)
{
    // Timer for measurements
    dopt::HighPrecisionTimer timer;

    // Parse debug flags
    DebugCfg debugFlags;
    TrainReturnCodes resultParseFlagsForDebug = parseFlagsForDebug(debugFlags, cmdline);
    if (resultParseFlagsForDebug != TrainReturnCodes::eOk)
        return resultParseFlagsForDebug;

    // Parse flags for network configutation
    NetworkCfg networkCfg;
    TrainReturnCodes resultParseFlagsForNetworkCfg = parseFlagsForNetworkConfiguration(networkCfg, cmdline);
    if (resultParseFlagsForNetworkCfg != TrainReturnCodes::eOk)
        return resultParseFlagsForNetworkCfg;
    
    // Explicitly check that we are running master
    assert(networkCfg.iAmServer == true);
    if (!networkCfg.iAmServer)
    {
        std::cerr << "This binary application represent a server, not a client. Please specify '--iam server' during launching it\n";
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
    int nClients = args.runtime.nClients;                        ///< Obtained from command line

    dopt::RandomGenIntegerLinear clientSampler;                  ///< Sampler for client selection
    int nSelectedClients = args.runtime.nClientsPerRound;        ///< Number of (local) clients
    assert(nSelectedClients > 0 && nSelectedClients <= nClients);

    VectorND<IndexType> allClients = VectorND<IndexType>::sequence<(IndexType)0>(nClients);
    VectorND<IndexType> selectedClients;
    clientSampler.setSeed(args.runtime.clientSelectionSeed);

    //dopt::Socket messageToSendGradientsSocketListen;        ///< Uncompressed information about gradients
    //dopt::Socket messageToSendLkSocketListen;               ///< Lk information to send (one scalar Lk)
    //dopt::Socket messageToSendFiSocketListen;               ///< Function value in current iterate
    //dopt::Socket messageToSendHessiansIndiciesSocketListen; ///< Compressed information about Hessian items indicies that are going to be send to master
    //dopt::Socket messageToSendHessiansItemsSocketListen;    ///< Compressed information about Hessian items values that are going to be send to master
    //dopt::Socket learningHessianSocketListen;               ///< Debug socket to send Hessian from clients

    //dopt::Socket auxInformationSocketListen;                ///< Aux information socket
    dopt::Socket controlSocketListen;                       ///< Debug socket to send Hessian from clients

    // Indicies to fetch items from arrays
    //constexpr size_t kGrad          = 0;
    //constexpr size_t kLk            = 1;
    //constexpr size_t kFi            = 2;
    //constexpr size_t kHessianInd    = 3;
    //constexpr size_t kHessianItems  = 4;
    //constexpr size_t kLearnHessians = 5;
    //constexpr size_t kAux           = 6;

    constexpr size_t kControl       = 0;

    // Listen Sockets for accept Connections
    dopt::Socket* serverListenSockets[] = { /*&messageToSendGradientsSocketListen,
                                            &messageToSendLkSocketListen, 
                                            &messageToSendFiSocketListen,
                                            &messageToSendHessiansIndiciesSocketListen, 
                                            &messageToSendHessiansItemsSocketListen, 
                                            &learningHessianSocketListen,
                                            &auxInformationSocketListen, */
                                            &controlSocketListen };

    // Total number of connections which each client should establish with master
    constexpr size_t kTotalConnectionsPerClient = sizeof(serverListenSockets) / sizeof(serverListenSockets[0]);

    static_assert(kTotalConnectionsPerClient == kControl + 1);
    
    
    // Connections with clients
    std::map<uint32_t/*client*/, std::unique_ptr<dopt::Socket>/*socket for this client*/> workersConnections[kTotalConnectionsPerClient];

    std::vector<std::unique_ptr<dopt::Socket>> workersConnectionsToAssign[kTotalConnectionsPerClient];
    
    {
        if (printAtTraining)
        {
            // Initialize Server: Create Listening sockets
            std::cout << "Master. Waiting for clients at base port " << networkCfg.serverHostName << ':' << networkCfg.serverPort << '\n';
        }

        for (size_t i = 0; i < kTotalConnectionsPerClient; ++i)
        {
            dopt::Socket& s = *(serverListenSockets[i]);
            s = dopt::Socket(networkCfg.protocol);

            if (!s.bind(networkCfg.serverPort + i, true))
            {
                std::cerr << "Master can not be bind to port " << networkCfg.serverPort + i << ". Please try another port.\n";
                return TrainReturnCodes::eInternalError;
            }

            assert(networkCfg.protocol == dopt::Socket::Protocol::TCPv4 || networkCfg.protocol == dopt::Socket::Protocol::TCPv6);
                
            if (!s.listen())
            {
                std::cerr << "Master can not start listen to port " << networkCfg.serverPort + i << ". Please try another port.\n";
                return TrainReturnCodes::eInternalError;
            }
        }
        std::cout << "Waiting for " << nClients << " workers to connect to training process at " << networkCfg.serverHostName << ':' << networkCfg.serverPort << '\n';
        
        // Initialize Server: Setup connections with all clients
        {
            dopt::MutableData bufferReq;
            WorkerContext::requestWorkerDescriptionBySever(bufferReq);

            for (size_t cNum = 0; cNum < nClients; ++cNum)
            {
                // Accept connection for all ports
                for (size_t i = 0; i < kTotalConnectionsPerClient; ++i)
                {
                    dopt::Socket& s = *(serverListenSockets[i]);
                    std::unique_ptr<dopt::Socket> incoming = s.serverAcceptConnection();
                    bool result = incoming->sendData(bufferReq.getPtr(), bufferReq.getFilledSize());
                    assert(result == true);
                    workersConnectionsToAssign[i].push_back(std::move(incoming));
    #if 0
                    uint32_t clientId = 0;
                    incoming->recvData(&clientId, sizeof(clientId));
                    auto result = workersConnections[i].emplace(clientId, std::move(incoming));
                
                    if (result.second == false)
                    {
                        std::cout << "Master ignores client: " << clientId << ". Such client has already been registered." << '\n';
                        return TrainReturnCodes::eInternalError;
                    }
    #endif
                }

                std::cout << cNum + 1 << "/" << nClients << " has been estblished connection [OK]" << '\n';
                std::cout << std::flush;
            }
        }    
    }

    timer.reset();
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

    std::cout << "=========================================================================\n";
    std::cout << "Main Loop preparation for: " << args.optAlgo.algorithm << '\n';
    std::cout << "=========================================================================\n";

    std::map<uint32_t, ClientDescription> clientsInformation;

    // Collect information about clients
    //for (size_t i = 0; i < workersConnectionsToAssign[kControl].size(); ++i)
    //{
    //    WorkerContext::requestWorkerDescriptionBySever(*workersConnectionsToAssign[kControl][i]);
    //}
    
    // Fomulate information about the clients
    while (!workersConnectionsToAssign[kControl].empty())
    {
        std::unique_ptr<dopt::Socket> socket = std::move(workersConnectionsToAssign[kControl].back());
        workersConnectionsToAssign[kControl].pop_back();
        
        ClientDescription description;
        
        if (WorkerContext::recvWorkerDecription(*socket, description) == false)
        {
            std::cout << "Can not obtain task description from client: " << description.clientId << '\n';
            return eInternalError;
        }
        
        if (clientsInformation.find(description.clientId) != clientsInformation.end())
        {
            std::cout << "Obtained information that two clients has the same identifier: " << description.clientId << '\n';
            
            //============================================================================================================//
            // Send terminate signal to all workers
            // TODO: check
            dopt::MutableData buffer_a;
            WorkerContext::requestTerminateLoopBySever(buffer_a);

            ServerContext serverContext;
            dopt::MutableData buffer_b;
            serverContext.prepareUpdate(buffer_b, ServerContext::sig_terminate);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                i->second->sendData(buffer_a.getPtr(), buffer_a.getFilledSize());
                i->second->sendData(buffer_b.getPtr(), buffer_b.getFilledSize());
            }
            socket->sendData(buffer_a.getPtr(), buffer_a.getFilledSize());
            socket->sendData(buffer_b.getPtr(), buffer_b.getFilledSize());
            //============================================================================================================//           
            return eInternalError;
        }
        else
        {
            if (!socket->setNoDelay(true
                                    //false
                                    ))
            {
                std::cerr << "Can not setup options for TCP connection for client: " << description.clientId << '\n';
                return eInternalError;                
            }
            
            clientsInformation.try_emplace(description.clientId, description);
            workersConnections[kControl][description.clientId] = std::move(socket);
        }
    }

    // Compute Dimension of Optimization Problem
    size_t d = 0;
    size_t totalSamples = 0;
    
    for (auto i = clientsInformation.begin(); i != clientsInformation.end(); ++i)
    {
        if (i->second.dimension > d)
            d = i->second.dimension;
        totalSamples += i->second.samplesInClient;
    }

    double mathMuf = args.optPrb.lambda + 0.0; // Mu strong convexity from Logistic Loss with L2 regulizers in form \lambda |x|^2/2
    double mathLf = -1.0;


    if (!compute_L_smooth)
    {
        // Report Dimension/Muf of Optimization Problem and Leave the Loop
        {
            OptProblemDescription descr = {};

            descr.d = d;
            descr.L_f = mathLf;
            descr.mu_f = mathMuf;
            descr.flags = (OptProblemDescriptionFlags::eDimension) | (OptProblemDescriptionFlags::eMu);

            dopt::MutableData buffer;
            bool res_report = WorkerContext::reportInformationAboutOptProblemDescrBySever(buffer, descr);
            assert(res_report == true);

            bool res_terminate = WorkerContext::requestTerminateLoopBySever(buffer);
            assert(res_terminate == true);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                // uint32_t clientId = i->first;
                dopt::Socket& socket = *(i->second);
                bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                assert(result == true);
            }
        }        
    }
    else
    {
        // Report Dimension of Optimization Problem
        {
            OptProblemDescription descr = {};

            descr.d = d;
            descr.flags = (OptProblemDescriptionFlags::eDimension);

            dopt::MutableData buffer;
            bool res = WorkerContext::reportInformationAboutOptProblemDescrBySever(buffer, descr);
            assert(res == true);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                // uint32_t clientId = i->first;
                dopt::Socket& socket = *(i->second);
                bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                assert(result == true);
            }
        }

        // Start iterate to compute L constant
        const size_t kMaxIteration = 1000;

        // L smooth constant need to be computed for varioation of GD with optimal step size

        // mathLf = singleObjective.computeLSmoothness(0.001 * mathMuf, &iterations);
        
        if (networkCfg.iAmServer)
        {
            double maxEigenValue = 0.0;
            double epsTolerance = 0.001 * mathMuf;
            double invM = 1.0 / double(totalSamples);
            
            // TVec v(hessianBound.rows());
            
            VectorND_d v(d);
            VectorND_d tmp(totalSamples);

            dopt::RandomGenRealLinear rnd_generator = dopt::RandomGenRealLinear(0.0, 1.0);
            v.setAllRandomly(rnd_generator);

            //TVec vPrev = v;
            VectorND_d vPrev = v;
            
            vPrev /= vPrev.vectorLinfNorm();
            
            size_t numIteration = 0;
            dopt::MutableData buffer;
            
            for (; numIteration < kMaxIteration; ++numIteration)
            {
                //Mat hessianBound = Atr * A * invM / 4.0;
                // v = hessianBound * vPrev;
                // ======================================================================
                // A * vPrev
                
                buffer.seekStart(0);
                bool res = WorkerContext::requestMatrixVectorMultiplyWithSamples(buffer, vPrev);
                assert(res == true);

                for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                {
                    // uint32_t clientId = i->first;
                    dopt::Socket& socket = *(i->second);
                    bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                    assert(result == true);
                }
                
                {
                    size_t offset = 0;

                    // Traverse clients in order
                    for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                    {
                        uint32_t clientId = i->first;
                        VectorND_d result;
                        bool res = WorkerContext::recieveVector(*(i->second), result);
                        assert(res == true);
                        dopt::CopyHelpers::copy(tmp.data() + offset, result.dataConst(), result.size());
                        offset += result.size();
                    }
                    assert(offset == totalSamples);
                }

                // A' * (A * vPrev)
                {
                    // Traverse clients in order
                    size_t offset = 0;
                    for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                    {
                        uint32_t clientId = i->first;
                        VectorND_d tmp_request = tmp.get(offset, offset + clientsInformation[clientId].samplesInClient);
                        buffer.seekStart(0);
                        bool res = WorkerContext::requestMatrixVectorMultiplyWithSamplesTranpose(buffer, tmp_request);
                        assert(res == true);
                        dopt::Socket& socket = *(i->second);
                        bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                    }
                }

                {
                    v.setAllToDefault();
                    for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                    {
                        VectorND_d result;
                        bool res = WorkerContext::recieveVector(*(i->second), result);
                        assert(res == true);
                        v += result;
                    }
                }
                // A' * (A * vPrev) * invM/4

                // Extra scale
                v *= (invM / 4.0);

                maxEigenValue = v.vectorLinfNorm();
                //===========================================================================================
                v /= maxEigenValue;
                double error = (v - vPrev).vectorLinfNorm();

                if (error < epsTolerance)
                {
                    break;
                }
                //===========================================================================================
                dopt::CopyHelpers::swap(v, vPrev);
            }

            mathLf  = maxEigenValue + args.optPrb.lambda;
            
            if (numIteration == kMaxIteration)
            {
                if (printAtTraining)
                    std::cerr << "Maximum number of iterations has been exceeded";

                dopt::MutableData buffer;
                WorkerContext::requestHaltBySever(buffer);

                for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                {
                    dopt::Socket& controlSocket = *(i->second);
                    bool result = controlSocket.sendData(buffer.getPtr(), buffer.getFilledSize());
                    assert(result == true);
                }

                return TrainReturnCodes::eInternalError;
            }
            else
            {
                if (printAtTraining)
                    std::cout << "Compute L constants for objective (power iterations: " << numIteration << ")" << timer.timeStamp() << '\n';
            }
            
            // Report Dimension/Muf/Lf of Optimization Problem and Leave the Loop
            {
                OptProblemDescription descr = {};

                descr.d = d;
                descr.L_f = mathLf;
                descr.mu_f = mathMuf;
                descr.flags = (OptProblemDescriptionFlags::eDimension) | (OptProblemDescriptionFlags::eMu) | (OptProblemDescriptionFlags::eL);

                dopt::MutableData buffer;
                bool res_report = WorkerContext::reportInformationAboutOptProblemDescrBySever(buffer, descr);
                assert(res_report == true);

                bool res_request = WorkerContext::requestTerminateLoopBySever(buffer);
                assert(res_request == true);

                for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                {
                    // uint32_t clientId = i->first;
                    dopt::Socket& socket = *(i->second);
                    bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                    assert(result == true);                    
                    //bool res = WorkerContext::reportInformationAboutOptProblemDescrBySever(socket, descr);
                    //assert(res == true);
                }
            }
        }
    }
    //================================================================================================//
    // End of Loop in clients
    //================================================================================================//
    // Derived Quantitity from Specification

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

    std::vector<WorkerContext*>               localWorkersContext;
    std::vector<uint32_t>                     compressedHessianItemsOffset;
    localWorkersContext.reserve(nClients);
    compressedHessianItemsOffset.reserve(nClients);

    std::vector<dopt::RandomGenIntegerLinear> localWorkersRandomNumberGenerators;
    std::vector< std::vector<uint32_t>>       kCoordinatesReconstructForRandGenerator;
    std::vector< uint32_t>                    kCoordinatesReconstructForRandSeqKGenerator;

    // special strategies to save communication
    bool transfer_indicies_for_randk = cmdline.isFlagSetuped("transfer_indicies_for_randk");

    //===========================================================================================//
    // Need buffers to reconstruct indicies (When needed)
    if ( used_compressor == Compressor::eRandK || !transfer_indicies_for_randk )
    {
        localWorkersRandomNumberGenerators.reserve(nClients);
        kCoordinatesReconstructForRandGenerator.reserve(nClients);
    }

    if (used_compressor == Compressor::eRandSeqK || !transfer_indicies_for_randk)
    {
        localWorkersRandomNumberGenerators.reserve(nClients);
        kCoordinatesReconstructForRandSeqKGenerator.reserve(nClients);
    }
    //===========================================================================================//
    // Our vector semantics is actually copying.
    VectorND_d xCur = x0;

    // For tracking algorithm behaviour
    VectorND_d::TElementType function_aggregation = VectorND_d::TElementType();

    // For actual algorithm behabiour
    VectorND_d prev_global_gradient_estimation(d);

    // For actual algorithm behabiour
    VectorND_d global_gradient_estimation(d);

    // dopt::LightVectorND<VectorND_d> global_gradient_estimation_view(global_gradient_estimation.data(), d);

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

    // Initialize need matrices for FedNL
    if (is_algorithm_fednl1)
    {
        //S_in_place = MatrixNMD_d::getZeroSquareMatrix(d);
        H = MatrixNMD_d::getZeroSquareMatrix(d);
    }

    // Compute indicies once (possibly reconstruct)
    std::vector<uint32_t> indiciesOfUpperTriangularPart;
    uint32_t indiciesOfUpperTriangularPartSize = 0;

    if (!transfer_indicies_for_randk && (used_compressor == Compressor::eRandSeqK || used_compressor == Compressor::eRandK))
    {
        indiciesOfUpperTriangularPart = dopt::indiciesForUpperTriangularPart(H);
        indiciesOfUpperTriangularPartSize = indiciesOfUpperTriangularPart.size();
    }

    ServerContext serverContext;

    serverContext.rounds = rounds;
    serverContext.currenIterate = &xCur;

    serverContext.roundToStart = std::numeric_limits<size_t>::max();
    serverContext.clientPerRound = nSelectedClients;
    serverContext.selectedClients = &selectedClients;
    //serverContext.serverConnection = nullptr;
    
    //===========================================================================================//
    for (int c = 0; c < nClients; ++c)
    {
        {
            if (is_algorithm_gd)
            {
                WorkerContextForGD* ctx = new WorkerContextForGD();
                ctx->workerIndex = c;
                ctx->send_fi_from_worker = args.tracking.tracking_is_on;

                ctx->optProblem = nullptr;
                ctx->ctrBlock.gradIsReady = false;
                ctx->ctrBlock.fiIsReady = false;

                ctx->control = workersConnections[kControl][c].get();
                
                localWorkersContext.push_back(ctx);

                //ctx->messageToSendGradientsSocket = workersConnections[kGrad][c].get();
                //ctx->messageToSendFiSocket = workersConnections[kFi][c].get();
                //ctx->control = workersConnections[kControl][c].get();
                //ctx->auxInformation = workersConnections[kAux][c].get();
            }
            else if (is_algorithm_fednl1)
            {
                WorkerContextForFedNL* ctx = new WorkerContextForFedNL();
                
                ctx->workerIndex = c;
                ctx->transfer_indicies_for_randk = transfer_indicies_for_randk;
                ctx->send_fi_from_worker = args.tracking.tracking_is_on;

                ctx->optProblem = nullptr;

                ctx->ctrBlock.messageToSendGradientsIsReady = false;
                
                ctx->ctrBlock.messageToSendHessiansIndiciesIsReady = false;
                ctx->ctrBlock.messageToSendHessiansItemsIsReady = false;

                ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk = false;
                ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk = false;

                ctx->ctrBlock.messageToSendLearningHessiansIsReady = false;
                ctx->ctrBlock.messageToSendFiIsReady = false;
                ctx->ctrBlock.messageToSendLkIsReady = false;
                ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = false;
                ctx->ctrBlock.messageToSendClientHasBeenInitialized = false;
                
                ctx->compressorType = used_compressor;
                ctx->kForCompressor = kForCompressor;
                ctx->seedForRandomizedCompressor = size_t(123 + c); // hard coded seed for randomized compressor. TODO: If needed to make it more explicitly if needed.
                ctx->alpha = used_alpha_step_size;
                ctx->preScaleBeforeSend2Master = updateScalingInMaster;
                ctx->send_Lk_from_worker = (is_algorithm_fednl_has_option_b ? true : false);
                
                ctx->control = workersConnections[kControl][c].get();
                
                //ctx->messageToSendGradientsSocket = workersConnections[kGrad][c].get();
                //ctx->messageToSendLkSocket = workersConnections[kLk][c].get();
                //ctx->messageToSendFiSocket = workersConnections[kFi][c].get();
                //ctx->messageToSendHessiansIndiciesSocket = workersConnections[kHessianInd][c].get();
                //ctx->messageToSendHessiansItemsSocket = workersConnections[kHessianItems][c].get();
                //ctx->learningHessianSocket = workersConnections[kLearnHessians][c].get();
                //ctx->auxInformation = workersConnections[kAux][c].get();
                //ctx->control = workersConnections[kControl][c].get();

                if ( ctx->compressorType == Compressor::eRandK && !transfer_indicies_for_randk)
                {
                    dopt::RandomGenIntegerLinear g;
                    g.setSeed(ctx->seedForRandomizedCompressor);
                    localWorkersRandomNumberGenerators.push_back(g);
                    kCoordinatesReconstructForRandGenerator.emplace_back(std::vector<uint32_t>());
                }
                else if (ctx->compressorType == Compressor::eRandSeqK && !transfer_indicies_for_randk)
                {
                    dopt::RandomGenIntegerLinear g;
                    g.setSeed(ctx->seedForRandomizedCompressor);
                    localWorkersRandomNumberGenerators.push_back(g);
                    kCoordinatesReconstructForRandSeqKGenerator.push_back(uint32_t(0));
                }

                localWorkersContext.push_back(ctx);
                compressedHessianItemsOffset.push_back(0);
            }
            else
            {
                if (printAtTraining)
                    std::cerr << " Please specify valid algorithm name.\n";
                return TrainReturnCodes::eInternalError;
            }
        }
    }

    // Request compute Hessians
    if (is_algorithm_fednl1)
    {
        // Request learning hessians H^0
        {
            dopt::MutableData buffer;
            serverContext.prepareUpdate(buffer, ServerContext::sig_request_full_hessian_in_client);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                dopt::Socket& socket = *(i->second);
                bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                assert(result == true);
            }

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                uint32_t clientId = i->first;
                dopt::Socket& socketAux = *(i->second);

                uint8_t ctrl = 0;
                socketAux.recvData(&ctrl, sizeof(ctrl));
                assert(ctrl == ServerContext::ControlSignals::sig_response_full_hessian_in_client);

                MatrixNMD_d& hessian_local = ((WorkerContextForFedNL*)localWorkersContext[clientId])->learningHessian;
                hessian_local = MatrixNMD_d::getZeroSquareMatrix(d);

                socketAux.recvMatrixItems(hessian_local);
                H += hessian_local;
            }
            H *= inv_NClients;
        }

        // Request initial g^0
        {
            dopt::MutableData buffer;
            serverContext.prepareUpdate(buffer, ServerContext::sig_request_local_g_direction_in_client);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                dopt::Socket& socket = *(i->second);
                bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                assert(result == true);
            }

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                uint32_t clientId = i->first;
                dopt::Socket& socketAux = *(i->second);

                uint8_t ctrl = 0;
                socketAux.recvData(&ctrl, sizeof(ctrl));
                assert(ctrl == ServerContext::ControlSignals::sig_response_local_g_direction_in_client);

                VectorND_d& g_local = ((WorkerContextForFedNL*)localWorkersContext[clientId])->localGradient;
                g_local = VectorND_d(d);

                size_t szInBytes = socketAux.getUnsignedVaryingInteger();

                assert(szInBytes == g_local.sizeInBytes());
                socketAux.recvData(g_local.data(), g_local.sizeInBytes());
                
                prev_global_gradient_estimation += g_local;
            }
            prev_global_gradient_estimation *= inv_NClients;
        }

        // Request initial l^0
        {
            dopt::MutableData buffer;
            serverContext.prepareUpdate(buffer, ServerContext::sig_request_lk_difference_in_client);

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                dopt::Socket& socket = *(i->second);
                bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
                assert(result == true);
            }

            for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
            {
                uint32_t clientId = i->first;
                dopt::Socket& socketAux = *(i->second);

                uint8_t ctrl = 0;
                socketAux.recvData(&ctrl, sizeof(ctrl));
                assert(ctrl == ServerContext::ControlSignals::sig_response_lk_difference_in_client);

                size_t szInBytes = socketAux.getUnsignedVaryingInteger();

                auto& localLK = ((WorkerContextForFedNL*)localWorkersContext[clientId])->localLk;
                
                assert(szInBytes == sizeof(localLK));
                socketAux.recvData(&localLK, sizeof(localLK));
                
                prev_global_Li_fednl += localLK;
            }
            
            prev_global_Li_fednl *= inv_NClients;
        }
    }
    //std::cout << "!!" << H.frobeniusNormForSymmetricMatrixFromUpPart() << '\n';

    if (printAtTraining)
    {
        std::cout << "Main Loop preparation has been finished [OK]. " << timer.timeStamp() << '\n';
        std::cout << "=========================================================================\n";
        std::cout << "Main loop has been started: r:" << rounds << ",clients:" << nClients << ",d:" << d << ", n(total):" << totalSamples << '\n';
    }

    // placeholder for tracking results
    dopt::MutableData trackingData;

    double receivedBytesFromClientsForScalarsAccum = 0.0;
    double receivedBytesFromClientsForIndiciesAccum = 0.0;

    double receivedBytesFromClientsForScalarsLast = 0.0;
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

        trackingData.putInt32(totalSamples/nClients);
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
        WorkerContextForFedNL* restrict_ext wCtx = nullptr;
        VectorND_d* restrict_ext destinationsForGradient = nullptr;
    };
    
    auto processingRoutineForThPoolGradUpdate = [](const GradientUpdateTask& task, size_t thIndex) -> void 
    {
        task.destinationsForGradient[thIndex] += task.wCtx->localGradient;
    };
    //===========================================================================================================
    struct HessianUpdateTask
    {
        // Header information
        Compressor compressorType = Compressor();                                                  ///< Compressor
        
        bool transfer_indicies_for_randk = false;                                                  ///< Flag about transfering indicies
        
        size_t d = 0;                                                                              ///< Dimnesion for optimization problem
        
        WorkerContextForFedNL* restrict_ext wCtx = nullptr;                                        ///< Worker context
        
        const std::vector<uint32_t>* restrict_ext indiciesOfUpperTriangularPart = nullptr;         ///< Read only array
        
        MatrixNMD_d* restrict_ext H_in_place = nullptr;                                            ///< Parallel updates
        
        size_t compressedHessianItemsOffset = 0;                                                   ///< Offset for obtained items [need in case reconstructing indicies]

        uint32_t kCoordinatesReconstructForRandSeqKGenerator = 0;                                  ///< For RandSeqK
        
        std::vector<uint32_t>* restrict_ext kCoordinatesReconstructForRandGeneratorThis = nullptr; ///< For RandK

        // Actual Payload for task
        dopt::MutableData messageToSendHessiansItems;
        dopt::MutableData messageToSendHessiansIndicies;
    };

    auto processingRoutineForThPoolHessianUpdate = [](const HessianUpdateTask& task, size_t thIndex) -> void
    {
        Compressor used_compressor = task.compressorType;
        
        bool transfer_indicies_for_randk = task.transfer_indicies_for_randk;

        size_t d = task.d;

        size_t kObtainedItems = 0;

        if (used_compressor == Compressor::eNatural)
        {
            assert(task.messageToSendHessiansItems.getFilledSize() % 3 == 0);
            // 24bits == 3 bytes ENCODES 2 * FP64 components
            //  3 bytes corrensponds to 2 components
            kObtainedItems = (task.messageToSendHessiansItems.getFilledSize() * 2) / 3;
            assert(kObtainedItems % 2 == 0);
        }
        else
        {
            kObtainedItems = task.messageToSendHessiansItems.getFilledSize() / sizeof(VectorND_d::TElementType);
        }

        size_t compressedHessianItemsOffset = task.compressedHessianItemsOffset;
        
        // WARNING: memory in H in place is updated from several places
        MatrixNMD_d& H_in_place = *(task.H_in_place);
        
        //=====================================================================================================================
        dopt::Data messageHessianItems = dopt::Data(task.messageToSendHessiansItems.getPtr(),
                                                    task.messageToSendHessiansItems.getFilledSize(),
                                                    dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        
        dopt::Data messageHessianIndicies = dopt::Data(task.messageToSendHessiansIndicies.getPtr(),
                                                       task.messageToSendHessiansIndicies.getFilledSize(),
                                                       dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        //======================================================================================================================

        if (used_compressor == Compressor::eIdentical)
        {            
            // =================================================================
            // Reconstruct last index and skip old items
            // =================================================================
            size_t last_index = compressedHessianItemsOffset;

            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
                    if (last_index == 0)
                    {
                        if (messageHessianItems.isEmpty())
                        {
                            // We took all items
                            break;
                        }

                        double value = messageHessianItems.getDouble();
                        dopt::appendMT(H_in_place.getRaw(i, j), value);
                    }
                    else
                    {
                        // Skip (i, j)
                        last_index--;
                    }
                }

                if (messageHessianItems.isEmpty())
                {
                    // We took all items
                    break;
                }
            }
        }
        else if (used_compressor == Compressor::eNatural)
        {
            typedef dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, VectorND_d::TElementType> FPCompPack;
            FPCompPack packsBuffer[2];
            size_t componentsInBuffer = 0;
            uint32_t buffer = 0;

            // =================================================================
            // Reconstruct last index and skip old items
            // =================================================================
            size_t last_index = compressedHessianItemsOffset;

            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
                    if (last_index == 0)
                    {
                        if (messageHessianItems.isEmpty())
                        {
                            // We took all items
                            break;
                        }

                        if (componentsInBuffer == 0)
                        {
                            messageHessianItems.getBytes(&buffer, 3);
                            dopt::unpack2FP64NoMantissa(packsBuffer, buffer);
                            componentsInBuffer = 2;
                        }

                        double value = packsBuffer[0].real_value_repr; // get 0-component
                        double updateValue = task.wCtx->preScaleBeforeSend2Master * value;
                        
                        dopt::appendMT(H_in_place.getRaw(i, j), updateValue);

                        packsBuffer[0] = packsBuffer[1];               // copy 1-component to 0-component
                        componentsInBuffer--;                          // decrease number of components
                    }
                    else
                    {
                        // Skip (i, j)
                        last_index--;
                    }
                }

                if (messageHessianItems.isEmpty())
                {
                    // We took all items
                    break;
                }
            }
        }
        else if (used_compressor == Compressor::eTopK)
        {
            for (size_t i = 0; i < kObtainedItems; ++i)
            {
                auto index = messageHessianIndicies.getUint32();
                auto value = messageHessianItems.getDouble();
                dopt::appendMT(H_in_place.matrixByCols[index], value);
            }
        }
        else if (used_compressor == Compressor::eTopLEK)
        {
            for (size_t i = 0; i < kObtainedItems; ++i)
            {
                auto index = messageHessianIndicies.getUint32();
                auto value = messageHessianItems.getDouble();
                dopt::appendMT(H_in_place.matrixByCols[index], value);
            }
        }
        else if (used_compressor == Compressor::eRandK)
        {
            std::vector<uint32_t> kcoordinatesReconstruct;

            if (!transfer_indicies_for_randk)
            {
                for (size_t i = 0; i < kObtainedItems; ++i)
                {
                    auto index = (*task.kCoordinatesReconstructForRandGeneratorThis)[i + compressedHessianItemsOffset];
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
            else
            {
                for (size_t i = 0; i < kObtainedItems; ++i)
                {
                    auto index = messageHessianIndicies.getUint32();
                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
        }
        else if (used_compressor == Compressor::eRandSeqK)
        {
            uint32_t indiciesOfUpperTriangularPartSize = task.indiciesOfUpperTriangularPart->size();
            uint32_t kCoordinatesReconstructForRandSeqKGenerator = task.kCoordinatesReconstructForRandSeqKGenerator;

            if (!transfer_indicies_for_randk)
            {
                uint32_t offset = compressedHessianItemsOffset + kCoordinatesReconstructForRandSeqKGenerator;

                for (size_t i = 0; i < kObtainedItems; ++i)
                {
                    uint32_t index = (* task.indiciesOfUpperTriangularPart) [dopt::add_two_numbers_modN<uint32_t>(offset,
                                                                                                                  i,
                                                                                                                  indiciesOfUpperTriangularPartSize)];

                    double value = messageHessianItems.getDouble();
                    dopt::appendMT(H_in_place.matrixByCols[index], value);
                }
            }
            else
            {
                for (size_t i = 0; i < kObtainedItems; ++i)
                {
                    auto index = messageHessianIndicies.getUint32();
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

    std::vector<VectorND_d> gradientInMasterInPool;
    gradientInMasterInPool.reserve(kUseThreadPoolSize4Grad);

    for (size_t k = 0; k < kUseThreadPoolSize4Grad; ++k)
    {
        gradientInMasterInPool.emplace_back(VectorND_d(d));
    }

    dopt::ThreadPoolWithTaskQueue<HessianUpdateTask> threadPoolInMaster4HessiansUp(processingRoutineForThPoolHessianUpdate, kUseThreadPoolSize4Hessians, nClients);
    
    threadPoolInMaster4GradUp.signalResumeProcessing();   
    threadPoolInMaster4HessiansUp.signalResumeProcessing();
    //===========================================================================================================
    // POOLS INITIALIZATION IN MASTER END
    
    {
        dopt::MutableData activeBufferForRoundAndIterate;    ///< Memory buffer for sending round and iterate to active clients 
        dopt::MutableData passiveBufferForRound;             ///< Memory buffer for sending round to non-active clients

        double iterateAndRoundUpdateTimeMs = 0.0;      ///< Time of last update of iterate and round in server
        double xCurL2Norm = 0.0;
        
        for (int r = 0; r < rounds; ++r)
        {
            if (is_algorithm_gd)
            {
                // GD

                // Start new round
                serverContext.roundToStart = r;

                // Update information about time when send has been started
                iterateAndRoundUpdateTimeMs = timer.getTimeMs();

                activeBufferForRoundAndIterate.rewindToStart();
                serverContext.prepareUpdate(activeBufferForRoundAndIterate, ServerContext::sig_active_update_round_and_iterate);

                // Send update to all clients
                for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                {
                    uint32_t clientId = i->first;
                    dopt::Socket& socket = *(i->second);

                    bool result = socket.sendData(activeBufferForRoundAndIterate.getPtr(), 
                                                  activeBufferForRoundAndIterate.getFilledSize());
                        
                    assert(result == true);
                }

                xCurL2Norm = xCur.vectorL2Norm();
                
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

                        ctx->receiveUpdates(workersConnections[kControl][c].get());

                        {
                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.gradIsReady))
                            {
                                if (!silentRun)
                                    logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[GD/grad]", global_gradient_estimation.sizeInBytes());

                                waitlist_with_grad_info -= 1;
                                global_gradient_estimation += ctx->localGradient;
                                receivedBytesFromClientsForScalarsLast += global_gradient_estimation.sizeInBytes();
                            }

                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.fiIsReady))
                            {
                                if (!silentRun)
                                    logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[GD/fi]", sizeof(ctx->fiValue));

                                waitlist_with_function_info -= 1;

                                if (args.tracking.tracking_is_on)
                                {
                                    // Make function aggregation (if needed)
                                    function_aggregation += ctx->fiValue;
                                }
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

                // Inform clients about new round
                {
                    activeBufferForRoundAndIterate.rewindToStart();
                    serverContext.prepareUpdate(activeBufferForRoundAndIterate, ServerContext::sig_active_update_round_and_iterate);

                    passiveBufferForRound.rewindToStart();
                    serverContext.prepareUpdate(passiveBufferForRound, ServerContext::sig_nonactive_update_round);

                    iterateAndRoundUpdateTimeMs = timer.getTimeMs(); // Update information about time when send has been started
                    
                    for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                    {
                        uint32_t clientId = i->first;
                        dopt::Socket& socket = *(i->second);

                        if (clientHasBeenSelected(clientId, &serverContext))
                        {
                            bool result = socket.sendData(activeBufferForRoundAndIterate.getPtr(),
                                                          activeBufferForRoundAndIterate.getFilledSize());

                            assert(result == true);
                        }
                        else
                        {
                            bool result = socket.sendData(passiveBufferForRound.getPtr(),
                                                          passiveBufferForRound.getFilledSize());

                            assert(result == true);
                        }
                    }
                }

                // Norm of iterate before starting the round
                xCurL2Norm = xCur.vectorL2Norm();

                // Reset information about gradient and about function
                global_gradient_estimation.setAllToDefault();
                for (size_t k = 0; k < kUseThreadPoolSize4Grad; ++k) {
                    gradientInMasterInPool[k].setAllToDefault();
                }

                // Reset global Li
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
                // Reconstruct indicies for RandK
                //===============================================================================
                if (!transfer_indicies_for_randk)
                {
                    if (used_compressor == Compressor::eRandK)
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            if (clientHasBeenSelected(c, &serverContext))
                            {
                                kCoordinatesReconstructForRandGenerator[c] = dopt::generateRandKItemsInUpperTriangularPart(localWorkersRandomNumberGenerators[c], kForCompressor, H, indiciesOfUpperTriangularPart);
                            }
                        }
                    }
                    else if (used_compressor == Compressor::eRandSeqK)
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            if (clientHasBeenSelected(c, &serverContext))
                            {
                                kCoordinatesReconstructForRandSeqKGenerator[c] = dopt::generateRandSeqKItemsInUpperTriangularPartAsIndex(localWorkersRandomNumberGenerators[c], kForCompressor, H);
                            }
                        }
                    }
                }

                // Reset indicies
                for (int c = 0; c < nClients; ++c)
                    compressedHessianItemsOffset[c] = 0;

                //===============================================================================

                while (waitlist_with_grad_info >= 0 || waitlist_with_shifted_hessian_info >= 0 || waitlist_with_Lk_info >= 0)
                {
                    //std::cout << "waitlist_with_grad_info:" << waitlist_with_grad_info << "\n";
                    //std::cout << "waitlist_with_shifted_hessian_info:" << waitlist_with_shifted_hessian_info << "\n";
                    //std::cout << "waitlist_with_Lk_info:" << waitlist_with_Lk_info << "\n";
                    //std::cout << "\n" << std::flush;

                    for (int c = 0; c < nClients; ++c)
                    {
                        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];

                        ctx->receiveUpdates(ctx->control);

                        {
                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendGradientsIsReady))
                            {
                                if (!silentRun)
                                    logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[FedNL/gradient]", ctx->localGradient.sizeInBytes());

                                waitlist_with_grad_info -= 1;

                                receivedBytesFromClientsForScalarsLast += ctx->localGradient.sizeInBytes();

                                assert(ctx->localGradient.sizeInBytes() == d * sizeof(double));
                                assert(obtained_local_gradient.sizeInBytes() == d * sizeof(double));

                                if (kUseThreadPoolSize4Grad > 0)
                                {
                                    GradientUpdateTask task;
                                    task.wCtx = ctx;
                                    task.destinationsForGradient = gradientInMasterInPool.data();
                                    threadPoolInMaster4GradUp.addTask(task);
                                }
                                else
                                {
                                    global_gradient_estimation += ctx->localGradient;
                                }                                
                            }
                            
                            if (ctx->ctrBlock.messageToSendHessiansIndiciesIsReady && ctx->ctrBlock.messageToSendHessiansItemsIsReady)
                            {
                                ctx->ctrBlock.messageToSendHessiansIndiciesIsReady = false;
                                ctx->ctrBlock.messageToSendHessiansItemsIsReady = false;

                                if (!silentRun)
                                    logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[FedNL/hessian]", ctx->messageToSendHessiansItems.getFilledSize() + ctx->messageToSendHessiansIndicies.getFilledSize());

                                size_t kObtainedItems = 0;

                                if (used_compressor == Compressor::eNatural)
                                {
                                    assert(ctx->messageToSendHessiansItems.getFilledSize() % 3 == 0);
                                    // 24bits == 3 bytes ENCODES 2 * FP64 components
                                    //  3 bytes corrensponds to 2 components
                                    kObtainedItems = (ctx->messageToSendHessiansItems.getFilledSize() * 2) / 3;
                                    assert(kObtainedItems % 2 == 0);
                                }
                                else
                                {
                                    kObtainedItems = ctx->messageToSendHessiansItems.getFilledSize() / sizeof(VectorND_d::TElementType);
                                }

                                assert(ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk == ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk);

                                if (kUseThreadPoolSize4Hessians > 0)
                                {
                                    HessianUpdateTask task;
                                    task.wCtx = ctx;
                                    task.indiciesOfUpperTriangularPart = &indiciesOfUpperTriangularPart;
                                    task.H_in_place = &H;
                                    task.compressedHessianItemsOffset = compressedHessianItemsOffset[c];
                                    
                                    task.d = d;
                                    task.compressorType = used_compressor;
                                    task.transfer_indicies_for_randk = transfer_indicies_for_randk;

                                    if (!transfer_indicies_for_randk)
                                    {
                                        if (ctx->compressorType == Compressor::eRandK)
                                        {
                                            // For RandK
                                            task.kCoordinatesReconstructForRandGeneratorThis = &kCoordinatesReconstructForRandGenerator[c];
                                        }
                                        else if (ctx->compressorType == Compressor::eRandSeqK)
                                        {
                                            // For RandSeqK
                                            task.kCoordinatesReconstructForRandSeqKGenerator = kCoordinatesReconstructForRandSeqKGenerator[c];
                                        }
                                    }
                                    
                                    task.messageToSendHessiansItems = ctx->messageToSendHessiansItems;
                                    task.messageToSendHessiansIndicies = ctx->messageToSendHessiansIndicies;

                                    threadPoolInMaster4HessiansUp.addTask(task);
                                    // =================================================================
                                    compressedHessianItemsOffset[c] += kObtainedItems;
                                    
                                    if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                    {
                                        if (used_compressor == Compressor::eIdentical)
                                        {
                                            assert(compressedHessianItemsOffset[c] == (d * (d + 1)) / 2);
                                        }
                                        else if (used_compressor == Compressor::eNatural)
                                        {
                                            assert(compressedHessianItemsOffset[c] == (d * (d + 1)) / 2);
                                        }
                                        else if (used_compressor == Compressor::eTopK)
                                        {
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        else if (used_compressor == Compressor::eRandK)
                                        {
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        else if (used_compressor == Compressor::eRandSeqK)
                                        {
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                    }
                                    // =================================================================                                    
                                }
                                else
                                {
                                    // Unpack the results: start
                                    dopt::Data messageHessianItems = dopt::Data(ctx->messageToSendHessiansItems.getPtr(), ctx->messageToSendHessiansItems.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
                                    dopt::Data messageHessianIndicies = dopt::Data(ctx->messageToSendHessiansIndicies.getPtr(), ctx->messageToSendHessiansIndicies.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

                                    // Real number of bytes that need to be send
                                    receivedBytesFromClientsForScalarsLast += ctx->messageToSendHessiansItems.getFilledSize();
                                    receivedBytesFromClientsForIndiciesLast += ctx->messageToSendHessiansIndicies.getFilledSize();

                                    if (used_compressor == Compressor::eIdentical)
                                    {
                                        // =================================================================
                                        // Reconstruct last index and skip old items
                                        // =================================================================
                                        size_t last_index = compressedHessianItemsOffset[c];
                                    
                                        for (size_t j = 0; j < d; ++j)
                                        {
                                            for (size_t i = 0; i <= j; ++i)
                                            {
                                                if (last_index == 0)
                                                {
                                                    if (messageHessianItems.isEmpty())
                                                    {
                                                        // We took all items
                                                        break;
                                                    }

                                                    double value = messageHessianItems.getDouble();
                                                    H.getRaw(i, j) += value;
                                                }
                                                else
                                                {
                                                    // Skip (i, j)
                                                    last_index--;
                                                }
                                            }

                                            if (messageHessianItems.isEmpty())
                                            {
                                                // We took all items
                                                break;
                                            }
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                        // =================================================================
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == (d * (d + 1))/2);
                                        }
                                        // =================================================================
                                    }
                                    else if (used_compressor == Compressor::eNatural)
                                    {
                                        // =================================================================
                                        // Reconstruct last index and skip old items
                                        // =================================================================
                                        size_t last_index = compressedHessianItemsOffset[c];

                                        typedef dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, VectorND_d::TElementType> FPCompPack;
                                        FPCompPack packsBuffer[2];
                                        size_t componentsInBuffer = 0;
                                        uint32_t buffer = 0;

                                        for (size_t j = 0; j < d; ++j)
                                        {
                                            for (size_t i = 0; i <= j; ++i)
                                            {
                                                if (last_index == 0)
                                                {
                                                    if (messageHessianItems.isEmpty() && componentsInBuffer == 0)
                                                    {
                                                        // We took all items
                                                        break;
                                                    }

                                                    if (componentsInBuffer == 0)
                                                    {
                                                        messageHessianItems.getBytes(&buffer, 3);
                                                        dopt::unpack2FP64NoMantissa(packsBuffer, buffer);
                                                        componentsInBuffer = 2;
                                                    }

                                                    double value = packsBuffer[0].real_value_repr;
                                                    double updateValue = ctx->preScaleBeforeSend2Master * value;

                                                    H.getRaw(i, j) += updateValue;

                                                    packsBuffer[0] = packsBuffer[1];               // copy 1-component to 0-component
                                                    componentsInBuffer--;                          // decrease number of components
                                                }
                                                else
                                                {
                                                    // Skip (i, j)
                                                    last_index--;
                                                }
                                            }
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                        // =================================================================
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == 1 + (d * (d + 1)) / 2 ||
                                                   compressedHessianItemsOffset[c] == 2 + (d * (d + 1)) / 2);
                                        }
                                        // =================================================================
                                    }
                                    else if (used_compressor == Compressor::eTopK)
                                    {
                                        for (size_t i = 0; i < kObtainedItems; ++i)
                                        {
                                            auto index = messageHessianIndicies.getUint32();
                                            auto value = messageHessianItems.getDouble();
                                            H.matrixByCols[index] += value;
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                    
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        // =================================================================
                                    }
                                    else if (used_compressor == Compressor::eTopLEK)
                                    {                                    
                                        for (size_t i = 0; i < kObtainedItems; ++i)
                                        {
                                            auto index = messageHessianIndicies.getUint32();
                                            auto value = messageHessianItems.getDouble();
                                            H.matrixByCols[index] += value;
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        // =================================================================
                                    }
                                    else if (used_compressor == Compressor::eRandK)
                                    {
                                        if (!transfer_indicies_for_randk)
                                        {
                                            uint32_t offset = compressedHessianItemsOffset[c];

                                            for (size_t i = 0; i < kObtainedItems; ++i)
                                            {
                                                auto index = kCoordinatesReconstructForRandGenerator[c][i + offset];
                                                double value = messageHessianItems.getDouble();
                                                H.matrixByCols[index] += value;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t i = 0; i < kObtainedItems; ++i)
                                            {
                                                auto index = messageHessianIndicies.getUint32();
                                                double value = messageHessianItems.getDouble();
                                                H.matrixByCols[index] += value;
                                            }
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        // =================================================================
                                    }
                                    else if (used_compressor == Compressor::eRandSeqK)
                                    {
                                        if (!transfer_indicies_for_randk)
                                        {
                                            uint32_t offset = compressedHessianItemsOffset[c] + kCoordinatesReconstructForRandSeqKGenerator[c];

                                            for (size_t i = 0; i < kObtainedItems; ++i)
                                            {
                                                uint32_t index = indiciesOfUpperTriangularPart[dopt::add_two_numbers_modN<uint32_t>(offset, 
                                                                                                                                    i, 
                                                                                                                                    indiciesOfUpperTriangularPartSize)];
                                            
                                                double value = messageHessianItems.getDouble();
                                                H.matrixByCols[index] += value;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t i = 0; i < kObtainedItems; ++i)
                                            {
                                                uint32_t index = messageHessianIndicies.getUint32();
                                                double value = messageHessianItems.getDouble();
                                                H.matrixByCols[index] += value;
                                            }
                                        }
                                        // =================================================================
                                        compressedHessianItemsOffset[c] += kObtainedItems;
                                    
                                        if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk && 
                                            ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk)
                                        {
                                            // Check that all upper triangular part items has been transfered
                                            assert(compressedHessianItemsOffset[c] == kForCompressor);
                                        }
                                        // =================================================================
                                    }
                                }

                                if (ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk &&
                                    ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk
                                    )
                                {
                                    ctx->ctrBlock.messageToSendHessiansIndiciesIsLastChunk = false;
                                    ctx->ctrBlock.messageToSendHessiansItemsIsLastChunk = false;
                                    waitlist_with_shifted_hessian_info -= 1;
                                }

                            }

                            if (is_algorithm_fednl_has_option_b)
                            {
                                if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendLkIsReady))
                                {
                                    if (!silentRun)
                                        logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[FedNL/Lk]", sizeof(ctx->messageToSendLk));

                                    waitlist_with_Lk_info -= 1;

                                    double Li_fednl = ctx->messageToSendLk;

                                    global_Li_fednl += Li_fednl;
                                    receivedBytesFromClientsForScalarsLast += sizeof(ctx->messageToSendLk);
                                }
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
                if (kTrackHessianDifference)
                {
                    // Wait all client to finish
                    //=========================================================================//
                    size_t finished = 0;                    
                    do
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];
                            ctx->receiveUpdates(ctx->control);
                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished))
                                finished++;
                        }
                    } while (finished != nClients);

                    // Request all client to send full current shift
                    //======================================================================================//
                    {
                        dopt::MutableData buffer;
                        serverContext.prepareUpdate(buffer, ServerContext::sig_request_full_hessian_in_client);

                        for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                        {
                            bool result = i->second->sendData(buffer.getPtr(), buffer.getFilledSize());
                            assert(result == true);
                        }
                    }

                    // Collect all client to send full current shift
                    //======================================================================================//
                    MatrixNMD_d LL = MatrixNMD_d::getZeroSquareMatrix(d);
                    
                    for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
                    {
                        uint8_t ctrl = i->second->getUint8();
                        assert(ctrl == ServerContext::ControlSignals::sig_response_full_hessian_in_client);

                        MatrixNMD_d hessian_local = MatrixNMD_d::getZeroSquareMatrix(d);
                        i->second->recvMatrixItems(hessian_local);
                        LL += hessian_local;
                    }                    
                    LL *= inv_NClients;
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
                            ctx->receiveUpdates(ctx->control);

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
                        
                        xCur = dopt::gausEleminationSolver(matrixOfSlau, global_gradient_estimation);

                        if constexpr (kDebugOutputForSolve)
                        {
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
                
                // Function aggregation if needed
                if (args.tracking.tracking_is_on)
                {
                    int waitlist_with_function_info = nSelectedClients;

                    while (waitlist_with_function_info >= 0)
                    {
                        for (int c = 0; c < nClients; ++c)
                        {
                            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)localWorkersContext[c];
                            ctx->receiveUpdates(ctx->control);

                            if (dopt::checkAndResetIfSet(&ctx->ctrBlock.messageToSendFiIsReady))
                            {
                                if (!silentRun)
                                    logComSpeed(iterateAndRoundUpdateTimeMs, timer.getTimeMs(), c, "[FedNL/fi]", sizeof(ctx->messageToSendFi));

                                waitlist_with_function_info -= 1;
                                
                                function_aggregation += ctx->messageToSendFi;
                                receivedBytesFromClientsForScalarsLast += sizeof(ctx->messageToSendFi);
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
            }

            // Report norm of full gradient
            if (!silentRun && printAtTraining)
            {
                if (args.tracking.tracking_is_on)
                {
                    std::cout << "  Round #" << r
                              << ": norm of full gradient: " << (global_gradient_estimation).vectorL2Norm()
                              << ": f(x): " << function_aggregation
                              << ": l: " << global_Li_fednl
                              << ": avg. time per round: " << timer.getTimeMs() / double(r + 1) << " msec"
                              << ": rounds/second: " << double(r + 1) / timer.getTimeSec()
                              << '\n';
                }
                else
                {
                    std::cout << "  Round #" << r
                              << ": norm of full gradient: " << (global_gradient_estimation).vectorL2Norm()
                              << ": f(x): " << "N/A"
                              << ": l: " << global_Li_fednl
                              << ": avg. time per round: " << timer.getTimeMs() / double(r + 1) << " msec"
                              << ": rounds/second: " << double(r + 1) / timer.getTimeSec()
                              << '\n';
                }
                std::cout << std::flush;
            }

            if (debugFlags.debugMemInfo && printAtTraining)
                printMemoryInformation();

            if (args.tracking.tracking_is_on)
            {
                trackingData.putInt32(r);
                
                VectorND_d fullGrad4tracking = computeFullGradient(localWorkersContext, serverContext, xCur);
                
                trackingData.putDouble(fullGrad4tracking.vectorL2Norm());
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
    
    VectorND_d fullGradientInLastIterate = computeFullGradient(localWorkersContext, serverContext, xCur);
    
    {
        // Send terminate signal to all workers
        dopt::MutableData buffer;
        serverContext.prepareUpdate(buffer, ServerContext::sig_terminate);

        for (auto i = workersConnections[kControl].begin(); i != workersConnections[kControl].end(); ++i)
        {
            dopt::Socket& socket = *(i->second);
            bool result = socket.sendData(buffer.getPtr(), buffer.getFilledSize());
            assert(result == true);
        }

        // Deinitialize Network Subsystem
        dopt::Socket::deinitNetworkSubSystem();
    }

    if (printAtTraining) {
        std::cout << "=========================================================================\n";
        std::cout << "Main loop has been finished: r:" << rounds <<
                     ",clients:"   << nClients <<
                     ",d:"         << d <<
                     ",n (total):" << totalSamples << '\n';
    }

    double elapsedTime = timer.getTimeMs();

    if (printAtTraining)
    {
        std::cout << timer.timeStamp() << '\n';

        std::cout << "=========================================================================\n";
        std::cout << "Statistics\n\n";
        std::cout << "  Average time per round: " << elapsedTime / rounds << " milliseconds\n";
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
        std::cout << "  Received MBytes from the clients: " << (receivedBytesFromClientsForScalarsAccum + receivedBytesFromClientsForIndiciesAccum) / (1024.0 * 1024.0) << '\n';
        std::cout << "  Clients compressor: " << args.optAlgo.compressor << '\n';
        std::cout << "  Clients compressor K: " << kForCompressor << " / Hessian shape is [d,d] where d: " << d
              << " / Maximum K ( {d(d+1)}/2 ): " << (d * (d + 1)) / 2 << '\n';
        std::cout << '\n';
        std::cout << "  Last iterate CRC-32: " << dopt::crc32(xCur.rawData(), xCur.sizeInBytes(), dopt::crc32Seed()) << '\n';
        // std::cout << "  Gradient in last iterate CRC-32: " << dopt::crc32(global_gradient_estimation.rawData(), global_gradient_estimation.sizeInBytes(), dopt::crc32Seed()) << '\n';
        std::cout << "  Norm of last iterate: " << xCur.vectorL2Norm() << '\n';
        // std::cout << "  Estimate of norm of full gradient in last iterate: " << global_gradient_estimation.vectorL2Norm() << '\n';

        std::cout << "  True of norm of full gradient in last iterate: " << fullGradientInLastIterate.vectorL2Norm() << '\n';

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

    if (resultCallback) {
        resultCallback(xCur.size());
    }

    return TrainReturnCodes::eOk;
}
