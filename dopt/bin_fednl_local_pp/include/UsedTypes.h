#pragma once

#include "Args.h"

// System includes
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/system/include/threads/Semaphore.h"

// ML relative incldes
#include "dopt/optimization_problems/include/ml/dataset_loader_fs.h"
#include "dopt/optimization_problems/include/ml/logistic_regression.h"

// C++/C standart headers
#include <atomic>
#include <stddef.h>
#include <stdint.h>

enum Compressor : std::uint8_t
{
    eIdentical = 0,
    eRandK     = 1,
    eTopK      = 2,
    eRandSeqK  = 3,
    eTopLEK    = 4,
    eNatural   = 5
};

struct ServerContext
{
    size_t rounds = 0 ;                          ///< Total number of rounds during which worker should provide gradient estimators

    VectorND_d* currenIterate = 0;               ///< Current iterare (pointer to variable belong to another thread)

    std::atomic<size_t> roundToStart = 0;        ///< Signal from master about round number that can be started

    size_t clientPerRound = 0;                   ///< Number if clients/round
    
    VectorND<IndexType>* selectedClients = 0;    ///< Selected clients
    
#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
    dopt::DefaultSemaphore startSempahore;   ///< Start semaphore [to reduce spinning]
#endif

    /** Dimension of optimization problem
    * @return dimension of optimization problem
    */
    size_t getDimension() const{
        assert(currenIterate != nullptr);
        return currenIterate->size();
    }
};

// Worker Context
struct WorkerContext
{
    WorkerContext() 
    {
    }

    virtual ~WorkerContext()
    {
    }
};

struct GDControlBlock
{
    std::atomic<bool> resultIsReady;  ///< Signal setuped by worker when result is ready
};

struct WorkerContextForGD final: public WorkerContext
{
    WorkerContextForGD()
    : optProblem(nullptr)
    , workerIndex(-1)
    {}

    virtual ~WorkerContextForGD()
    {
        delete optProblem;
    }

    typedef dopt::L2RegulirizeLogisticRegression < MatrixNMD_d, VectorND_d, MatrixNMD_d::TElementType> OptimizationProblem;

    OptimizationProblem* optProblem; ///< Optimization Problem
    int workerIndex;                 ///< Index of worker
    VectorND_d localGradient;        ///< Result from last iteration (direction)
    GDControlBlock ctrBlock;         ///< Control Block
};

struct WorkerContextForFedNLControlBlock
{
    std::atomic<bool> messageToSendGradientsIsReady;         ///< Signal setuped by worker when gradients are ready
    std::atomic<bool> messageToSendHessiansIsReady;          ///< Signal setuped by worker when hessians are ready
    std::atomic<bool> messageToSendLkIsReady;                ///< Signal setuped by worker when Lk is ready
    std::atomic<bool> messageToSendFiIsReady;                ///< Signal setuped by worker when fi are ready
    std::atomic<bool> messageToSendRoundWorkHasBeenFinished; ///< Signal setuped by worker when update is ready

    std::atomic<bool> messageToSendClientHasBeenInitialized; ///< Signal setuped by worker when it has been initialized
};

// Worker Context
struct WorkerContextForFedNL final: public WorkerContext
{
    WorkerContextForFedNL()
    : workerIndex(-1)
    , transfer_indicies_for_randk(false)
    , send_fi_from_worker(false)
    , optProblem(nullptr)
    , compressorType(Compressor::eIdentical)
    , kForCompressor(0)
    , seedForRandomizedCompressor(0)
    , preScaleBeforeSend2Master(0.0)
    , alpha(0.0)
    , send_Lk_from_worker(false)
    , localLk(0.0)
    {}

    ~WorkerContextForFedNL()
    {
        delete optProblem;
    }

    //======================================================================================================================//
    // Optimization problem
    dopt::L2RegulirizeLogisticRegression<MatrixNMD_d, VectorND_d, MatrixNMD_d::TElementType>* optProblem;

    //======================================================================================================================//
    // Result are here [indirectly written]
    VectorND_d localGradient;          ///< Result from last iteration (direction)
    MatrixNMD_d learningHessian;       ///< Local Hessian (learnable)
    double localLk;                    ///< Local Lk
    //======================================================================================================================//

    // External (connection with outside world)
    dopt::MutableData messageToSendGradients;        ///< Uncompressed information about gradients
    dopt::MutableData messageToSendLk;               ///< Auxilirary information to send (one scalar Lk)
    dopt::MutableData messageToSendFi;               ///< Function value in current iterate

    dopt::MutableData messageToSendHessiansIndicies; ///< Compressed information about Hessian items indicies that are going to be send to master
    dopt::MutableData messageToSendHessiansItems;    ///< Compressed information about Hessian items values that are going to be send to master
    //======================================================================================================================//
    

    // Configured (outside) [WRITE ONCE]
    //======================================================================================================================//
    double preScaleBeforeSend2Master;       ///< Hessian updates before sending to master are prescaled with this coefficient: (alpha * 1/n)

    double alpha;                           ///< Hessian learning rate

    size_t seedForRandomizedCompressor;     ///< Used seed for randomized compressor

    size_t kForCompressor;                  ///< Number of coordinates sending back from compressor

    int16_t workerIndex;                    ///< Index of worker

    Compressor compressorType;              ///< Used compressor

    bool transfer_indicies_for_randk;       ///< Flag which specify that client should transfer indicies for RandK compressor.

    bool send_fi_from_worker;               ///< Flag denoted to fact that worker should compute fi and fill messageToSendFi.

    bool send_Lk_from_worker;               ///< Request for compute Lk by worker and send it

    //======================================================================================================================//

    WorkerContextForFedNLControlBlock ctrBlock;
};

inline bool clientHasBeenSelected(int16_t clientNumber, const ServerContext* serverCtx)
{
    size_t clientPerRound = serverCtx->clientPerRound;

    for (size_t cc = 0; cc < clientPerRound; ++cc)
    {
        if (serverCtx->selectedClients->get(cc) == clientNumber)
        {
            return true;
        }
    }

    return false;
}

// Binary format for output of experimental results:
// 
//  FIELD                             TYPE            COMMENT 
// 
// data set path                      [ASCIZ]         Base name of the file.
// algorithm name                     [ASCIZ]         Optimziation Algorithm name.
// compressor name                    [ASCIZ]         Compression Algorithm name.
// 
// number of samples per client       int32           Number of samples per client.
// total number of samples            int32           Total number of samples in all dataset.
// number of rounds                   int32           Number of round during which we have performed training.
// number of clients                  int32           Total number of clients.
// dimension of optimization variable int32           Dimenstion of optimization problem.
// k for comressor                    int32           Used "k" value for TopK, RandK compressors.
// 
// k compressor as d mult             fp64            Used "k" value for TopK, RandK compressors as a fraction of "d"
// global step size                   fp64            Used global step size for GD, not to FedNL.
// alpha step size                    fp64            Used alpha step size for FedNL, not to GD.
// lambda                             fp64            Used lambda regulirization mutiplier for optimization problem. For optimization problem: "f(x) = 1/m \sum_i log(1 + exp(-bi(ai' * x))) + \lambda/2 \|x\|^2"
// 
// <Records in format described below in amount of total number of rounds>
// {
// Round Number                                  int32
// <L2 norm of Full Gradient>                    fp64
// <Objective function value>                    fp64
// <L2 norm of iterate>                          fp64
// Amount of total bytes received from clients   fp64
// Second from start of training                 fp64
// }
//
