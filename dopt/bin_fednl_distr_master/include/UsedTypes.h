#pragma once

#include "Args.h"

// System includes
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/Data.h"
#include "dopt/system/include/threads/Semaphore.h"

// ML relative incldes
#include "dopt/optimization_problems/include/ml/dataset_loader_fs.h"
#include "dopt/optimization_problems/include/ml/logistic_regression.h"

// C++/C standart headers
#include <atomic>
#include <stddef.h>
#include <stdint.h>

#ifndef D_OPT_PACK_TRANSFERED_MESSAGES
    #define D_OPT_PACK_TRANSFERED_MESSAGES 1
#endif

#ifndef D_OPT_USE_ATOMICS_IN_DISTRIB_IMPL
    #define D_OPT_USE_ATOMICS_IN_DISTRIB_IMPL 0 ///< If set to "0" please be extremely carefull with non-atomic access in case if binary use several threads
#endif


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
    size_t rounds;                           ///< Total number of rounds during which worker should provide gradient estimators

    VectorND_d* currenIterate;               ///< Current iterare (pointer to variable belong to another thread)

#if D_OPT_USE_ATOMICS_IN_DISTRIB_IMPL
    
    std::atomic<size_t> roundToStart;        ///< Signal from master about round number that can be started

    std::atomic<size_t> lineSearchRound;     ///< Signal from master about (external loop) round number for line search that can be started [rounds less then this value will be ignored]

    std::atomic<size_t> lineSearchIteration; ///< Signal from master about (internal loop) line search iteration that can be started [iteration can be started only if lineSearchRound is equal to current line-search iteration number]
    
#else
    
    size_t roundToStart;                     ///< Signal from master about round number that can be started
    
    size_t lineSearchRound;                  ///< Signal from master about (external loop) round number for line search that can be started [rounds less then this value will be ignored]

    size_t lineSearchIteration;              ///< Signal from master about (internal loop) line search iteration that can be started [iteration can be started only if lineSearchRound is equal to current line-search iteration number]

#endif
    
    /** Dimension of optimization problem
    * @return dimension of optimization problem
    */
    size_t getDimension() const{
        assert(currenIterate != nullptr);
        return currenIterate->size();
    }

    bool terminate = false;             ///< Flag for terminate training process

    bool get_hessian = false;           ///< Flag for evaluate hessian at current iterate

    // Control socket signals
    
    enum ControlSignals : uint8_t
    {
        sig_terminate = 10,
        sig_update_round_and_iterate = 11,
        sig_request_full_hessian_in_client = 12,
        sig_response_full_hessian_in_client = 13,
        sig_update_linesearch_round_and_linesearch_iteration = 14,
        sig_update_linesearch_round_and_linesearch_iteration_and_iterate = 15
    };

    template<bool blockForFirstRecv = false>
    void receiveUpdates(dopt::Socket* serverConnection)
    {
        //=====================================================================
        if constexpr (blockForFirstRecv)
        {
            // Do nothing. No need to check available bytes
        }
        else
        {
            // Non-blocking call. If no bytes available return.
            if (serverConnection->getAvailableBytesForRead() == 0)
                return /*false*/;
        }
        //=====================================================================

        // for (;;)
        {
            uint8_t ctrl = 0;
            serverConnection->recvData(&ctrl, sizeof(ctrl));
            // uint64_t sz = serverConnection->getUint64();
            uint64_t sz = serverConnection->getUnsignedVaryingInteger();
            
            ControlSignals ctrl_action = static_cast<ControlSignals>(ctrl);

            switch (ctrl_action)
            {
                case sig_terminate:
                {
                    terminate = true;
                    break;
                }
                case sig_update_round_and_iterate:
                {
                    serverConnection->recvData(&roundToStart, sizeof(roundToStart));
                
                    sz -= sizeof(roundToStart);
                
                    if (currenIterate == nullptr)
                    {
                        currenIterate = new VectorND_d(sz / sizeof(VectorND_d::TElementType));
                    }
                    else if (currenIterate->sizeInBytes() != sz)
                    {
                        currenIterate->resize(sz / sizeof(VectorND_d::TElementType));
                    }
                
                    serverConnection->recvData(currenIterate->data(), sz);
                
                    break;
                }
                case sig_update_linesearch_round_and_linesearch_iteration:
                {
                    serverConnection->recvData(&lineSearchRound, sizeof(lineSearchRound));
                    sz -= sizeof(lineSearchRound);

                    serverConnection->recvData(&lineSearchIteration, sizeof(lineSearchIteration));
                    sz -= sizeof(lineSearchIteration);

                    assert(sz == 0);

                    break;
                }
                case sig_update_linesearch_round_and_linesearch_iteration_and_iterate:
                {
                    serverConnection->recvData(&lineSearchRound, sizeof(lineSearchRound));
                    sz -= sizeof(lineSearchRound);

                    serverConnection->recvData(&lineSearchIteration, sizeof(lineSearchIteration));
                    sz -= sizeof(lineSearchIteration);

                    if (currenIterate == nullptr)
                    {
                        currenIterate = new VectorND_d(sz / sizeof(VectorND_d::TElementType));
                    }
                    else if (currenIterate->sizeInBytes() != sz)
                    {
                        currenIterate->resize(sz / sizeof(VectorND_d::TElementType));
                    }
                    serverConnection->recvData(currenIterate->data(), sz);

                    break;
                }
                case sig_request_full_hessian_in_client:
                {
                    if (currenIterate == nullptr)
                    {
                        currenIterate = new VectorND_d(sz / sizeof(VectorND_d::TElementType));
                    }
                    else if (currenIterate->sizeInBytes() != sz)
                    {
                        currenIterate->resize(sz / sizeof(VectorND_d::TElementType));
                    }
                    serverConnection->recvData(currenIterate->data(), sz);
                    get_hessian = true;

                    break;
                }
                default:
                {
                    assert(!"ERROR");
                }
            }

            // if (serverConnection->getAvailableBytesForRead() == 0)
            //    break;
        }

        return/*true*/;
    }

    void prepareUpdate(dopt::MutableData& buffer, ControlSignals ctrl) const
    {
        switch (ctrl)
        {
            case sig_terminate:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingIntegerKnowAtCompileTime< uint8_t, uint8_t(0) >();
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingIntegerKnowAtCompileTime< uint8_t, uint8_t(0) > ();
#endif
                break;
            }
            case sig_request_full_hessian_in_client:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(currenIterate->sizeInBytes());
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(currenIterate->sizeInBytes());
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#endif
                break;
            }
            case sig_update_round_and_iterate:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(roundToStart) + currenIterate->sizeInBytes());
                buffer.putBytes(&roundToStart, sizeof(roundToStart));
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(roundToStart) + currenIterate->sizeInBytes());
                buffer.putValueToStream(roundToStart);
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#endif
                break;
            }
            case sig_update_linesearch_round_and_linesearch_iteration:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(lineSearchRound) + sizeof(lineSearchIteration));
                buffer.putValueToStream(lineSearchRound);
                buffer.putValueToStream(lineSearchIteration);
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(lineSearchRound) + sizeof(lineSearchIteration));
                buffer.putValueToStream(lineSearchRound);
                buffer.putValueToStream(lineSearchIteration);
#endif
                break;
            }
            case sig_update_linesearch_round_and_linesearch_iteration_and_iterate:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(lineSearchRound) + sizeof(lineSearchIteration) + currenIterate->sizeInBytes());
                buffer.putValueToStream(lineSearchRound);
                buffer.putValueToStream(lineSearchIteration);
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(lineSearchRound) + sizeof(lineSearchIteration) + currenIterate->sizeInBytes());
                buffer.putValueToStream(lineSearchRound);
                buffer.putValueToStream(lineSearchIteration);
                buffer.putBytes(currenIterate->dataConst(), currenIterate->sizeInBytes());
#endif
                break;
            }
            default:
            {
                assert(!"ERROR");
            }
        }
    }
};

// Worker Desciption
struct ClientDescription
{
    uint32_t clientId = 0;                  ///< Client ID
    uint32_t dimension = 0;                 ///< Dimension of optimization variable
    uint32_t samplesInClient = 0;           ///< Number of samples
    uint32_t rounds = 0;                    ///< Number of rounds for optimization process
    bool hasInterceptTerm = false;          ///< Dataset includes intercept term
};

enum OptProblemDescriptionFlags
{
    eDimension = 0x1 << 0,
    eMu = 0x1 << 1,
    eL = 0x1 << 2
};

// Worker Desciption
struct OptProblemDescription
{
    uint32_t flags;
    uint32_t d;
    double L_f;
    double mu_f;
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


    // Control socket signals
    enum ControlSignals : uint8_t
    {
        sig_get_worker_desciption = 20,
        sig_request_matrix_vector_multiplication_with_samples = 21,
        sig_request_matrix_vector_multiplication_with_samples_tranpose = 22,        
        sig_terminate_loop = 23,
        //sig_enter_loop     = 24,
        sig_halt           = 25,
        sig_info_about_opt_problem        = 26,
        //sig_info_about_opt_problem_L_and_mu_constant = 27,
        
        sig_response_worker_desciption                                  = 28,
        sig_response_matrix_vector_multiplication_with_samples          = 29,
        sig_response_matrix_vector_multiplication_with_samples_tranpose = 30        
    };

    static bool reportInformationAboutOptProblemDescrBySever(dopt::MutableData& controlBuffer, const OptProblemDescription& descr)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_info_about_opt_problem);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<size_t, (sizeof(descr.flags) + sizeof(descr.d) + sizeof(descr.L_f) + sizeof(descr.mu_f))> ();

        controlBuffer.putBytes(&descr.flags, sizeof(descr.flags));
        controlBuffer.putBytes(&descr.d, sizeof(descr.d));
        controlBuffer.putBytes(&descr.L_f, sizeof(descr.L_f));
        controlBuffer.putBytes(&descr.mu_f, sizeof(descr.mu_f));

        return true;
#else
        controlBuffer.putByte(sig_info_about_opt_problem);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<size_t, (sizeof(descr.flags) + sizeof(descr.d) + sizeof(descr.L_f) + sizeof(descr.mu_f))> ();

        controlBuffer.putValueToStream(descr.flags);
        controlBuffer.putValueToStream(descr.d);
        controlBuffer.putValueToStream(descr.L_f);
        controlBuffer.putValueToStream(descr.mu_f);

        return true;
#endif        
    }
    
    static bool requestTerminateLoopBySever(dopt::MutableData& controlBuffer)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_terminate_loop);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime< uint8_t, uint8_t(0) >();
        return true;
#else
        controlBuffer.putByte(sig_terminate_loop);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime< uint8_t, uint8_t(0) >();
        return true;
#endif
    }

    static bool requestHaltBySever(dopt::MutableData& controlBuffer)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_halt);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
        return true;
#else
        controlBuffer.putByte(sig_halt);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
        return true;
#endif        
    }

    static bool requestWorkerDescriptionBySever(dopt::MutableData& controlBuffer)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_get_worker_desciption);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
        return true;
#else
        controlBuffer.putByte(sig_get_worker_desciption);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
        return true;
#endif        
    }
    
    static bool requestMatrixVectorMultiplyWithSamples(dopt::MutableData& controlBuffer, const VectorND_d& vec)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_request_matrix_vector_multiplication_with_samples);
        controlBuffer.putUnsignedVaryingInteger(vec.sizeInBytes());
        controlBuffer.putBytes(vec.dataConst(), vec.sizeInBytes());
        return true;
#else
        controlBuffer.putByte(sig_request_matrix_vector_multiplication_with_samples);
        controlBuffer.putUnsignedVaryingInteger(vec.sizeInBytes());
        controlBuffer.putBytes(vec.dataConst(), vec.sizeInBytes());
        return true;
#endif
    }

    static bool requestMatrixVectorMultiplyWithSamplesTranpose(dopt::MutableData& controlBuffer, const VectorND_d& vec)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_request_matrix_vector_multiplication_with_samples_tranpose);
        controlBuffer.putUnsignedVaryingInteger(vec.sizeInBytes());
        controlBuffer.putBytes(vec.dataConst(), vec.sizeInBytes());
        return true;
#else
        controlBuffer.putByte(sig_request_matrix_vector_multiplication_with_samples_tranpose);
        controlBuffer.putUnsignedVaryingInteger(vec.sizeInBytes());
        controlBuffer.putBytes(vec.dataConst(), vec.sizeInBytes());
        return true;
#endif        
    }

    static bool sendWorkerDecription(dopt::MutableData& controlBuffer, const ClientDescription& desc)
    {        
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_response_worker_desciption);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<size_t, (sizeof(uint32_t) * 4 + sizeof(char))> ();
        controlBuffer.putValueToStream(desc.clientId);
        controlBuffer.putValueToStream(desc.dimension);
        controlBuffer.putValueToStream(desc.samplesInClient);
        controlBuffer.putValueToStream(desc.rounds);
        controlBuffer.putValueToStream(desc.hasInterceptTerm);

        return true;
#else
        controlBuffer.putByte(sig_response_worker_desciption);
        controlBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<size_t, (sizeof(uint32_t) * 4 + sizeof(char))> ();
        controlBuffer.putValueToStream(desc.clientId);
        controlBuffer.putValueToStream(desc.dimension);
        controlBuffer.putValueToStream(desc.samplesInClient);
        controlBuffer.putValueToStream(desc.rounds);
        controlBuffer.putValueToStream(desc.hasInterceptTerm);

        return true;
#endif        
    }

    static bool sendMatrixVectorMultiplyWithSamplesTranpose(dopt::MutableData& controlBuffer, const VectorND_d& res)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_response_matrix_vector_multiplication_with_samples_tranpose);
        controlBuffer.putUnsignedVaryingInteger(res.sizeInBytes());
        controlBuffer.putBytes(res.dataConst(), res.sizeInBytes());
        return true;
#else
        controlBuffer.putByte(sig_response_matrix_vector_multiplication_with_samples_tranpose);
        controlBuffer.putUnsignedVaryingInteger(res.sizeInBytes());
        controlBuffer.putBytes(res.dataConst(), res.sizeInBytes());
        return true;
#endif

    }

    static bool sendMatrixVectorMultiplyWithSamples(dopt::MutableData& controlBuffer, const VectorND_d& res)
    {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
        controlBuffer.putByte(sig_response_matrix_vector_multiplication_with_samples);
        controlBuffer.putUnsignedVaryingInteger(res.sizeInBytes());
        controlBuffer.putBytes(res.dataConst(), res.sizeInBytes());
        return true;
#else
        controlBuffer.putByte(sig_response_matrix_vector_multiplication_with_samples);
        controlBuffer.putUnsignedVaryingInteger(res.sizeInBytes());
        controlBuffer.putBytes(res.dataConst(), res.sizeInBytes());
        return true;
#endif        
    }

    static bool ignoreResidualFromCommand(ControlSignals nextCommand, dopt::Socket& controlSocket)
    {
        bool result = true;
        //uint64_t sz = controlSocket.getUint64();
        uint64_t sz = controlSocket.getUnsignedVaryingInteger();

        // if (!result)
        //    return false;

        assert(sz == 0);

        // Manually skip the data
        {
            char buff[1024];            
            for (; sz > 0;)
            {
                if (sz >= sizeof(buff))
                {
                    controlSocket.recvData(buff, sizeof(buff));
                    sz -= sizeof(buff);
                }
                else
                {
                    controlSocket.recvData(buff, sz);
                    sz = 0;
                }
            }
        }

        return result;
    }

    template<bool blockForFirstRecv = false>
    [[nodiscard]] static bool receiveCommand(ControlSignals& nextCommand, dopt::Socket& controlSocket)
    {
        //=====================================================================
        if constexpr (blockForFirstRecv)
        {
            // Do nothing. No need to check available bytes
        }
        else
        {
            // Non-blocking call. If no bytes available return.
            if (controlSocket.getAvailableBytesForRead() == 0)
                return false;
        }
        //=====================================================================

        bool result = controlSocket.recvData(&nextCommand, sizeof(nextCommand));

        return result;
    }
    
    static OptProblemDescription extractProblemDescrFromCommand(const ControlSignals nextCommand, dopt::Socket& controlSocket)
    {
        bool result = true;
        //uint64_t szInBytes = controlSocket.getUint64();
        uint64_t szInBytes = controlSocket.getUnsignedVaryingInteger();
        //result &= controlSocket.recvData(&szInBytes, sizeof(szInBytes));

        OptProblemDescription descr = {};
        assert(sizeof(descr.flags) + sizeof(descr.d) + sizeof(descr.L_f) + sizeof(descr.mu_f) == szInBytes);

        result &= controlSocket.recvData(&descr.flags, sizeof(descr.flags));
        result &= controlSocket.recvData(&descr.d,     sizeof(descr.d));
        result &= controlSocket.recvData(&descr.L_f,   sizeof(descr.L_f));
        result &= controlSocket.recvData(&descr.mu_f,  sizeof(descr.mu_f));

        assert(result == true);

        return descr;
    }

    static VectorND_d extractVectorArgFromCommand(const ControlSignals nextCommand, dopt::Socket& controlSocket)
    {
        bool result = true;
        
        //uint64_t szInBytes = controlSocket.getUint64();
        uint64_t szInBytes = controlSocket.getUnsignedVaryingInteger();
        //result &= controlSocket.recvData(&szInBytes, sizeof(szInBytes));

        uint64_t szInElements = szInBytes / sizeof(VectorND_d::TElementType);

        assert(szInBytes % sizeof(VectorND_d::TElementType) == 0);
        assert(nextCommand == sig_request_matrix_vector_multiplication_with_samples || nextCommand == sig_request_matrix_vector_multiplication_with_samples_tranpose);
        
        VectorND_d resultVector(szInElements);
        
        result &= controlSocket.recvData( resultVector.data(), 
                                          resultVector.sizeInBytes() );

        assert(result == true);
        
        return resultVector;
    }

#if 0
    static bool receiveCommand(ControlSignals& nextCommand, 
                               VectorND_d* nextCommandVectorArgument, 
                               uint64_t* nextCommandScalarArgument, 
                               dopt::Socket& controlSocket)
    {
        if (controlSocket.getAvailableBytesForRead() == 0)
            return false;

        static_assert(sizeof(nextCommand) == sizeof(uint8_t));

        bool result = true;
        
        result &= controlSocket.recvData(&nextCommand, sizeof(nextCommand));

        //uint64_t sz = controlSocket.getUint64();
        uint64_t sz = controlSocket.getUnsignedVaryingInteger();
        //result &= controlSocket.recvData(&sz, sizeof(sz));

        if (nextCommand == sig_request_matrix_vector_multiplication_with_samples || 
            nextCommand == sig_request_matrix_vector_multiplication_with_samples_tranpose)
        {
            uint64_t szInElements = sz / sizeof(VectorND_d::TElementType);
            
            assert(sz % sizeof(VectorND_d::TElementType) == 0);
            assert(nextCommandVectorArgument != nullptr);

            if (nextCommandVectorArgument->size() != szInElements)
                *nextCommandVectorArgument = VectorND_d(szInElements);

            result &= controlSocket.recvData(nextCommandVectorArgument->data(), nextCommandVectorArgument->sizeInBytes());
        }
        else if (nextCommand == sig_info_about_opt_problem_dimenstion)
        {
            assert(sz == sizeof(uint64_t));
            assert(nextCommandScalarArgument != nullptr);
            
            result &= controlSocket.recvData(nextCommandScalarArgument, sizeof(*nextCommandScalarArgument));
        }
        else if (nextCommand == sig_info_about_opt_problem_L_and_mu_constant)
        {
            assert(sz == sizeof(VectorND_d::TElementType) * 2);
            assert(sz % sizeof(VectorND_d::TElementType) == 0);
            assert(nextCommandVectorArgument != nullptr);

            uint64_t szInElements = sz / sizeof(VectorND_d::TElementType);

            if (nextCommandVectorArgument->size() != szInElements)
                *nextCommandVectorArgument = VectorND_d(szInElements);

            result &= controlSocket.recvData(nextCommandVectorArgument->data(), nextCommandVectorArgument->sizeInBytes());            
        }
        else
        {
            assert(sz == 0);
        }
        
        return result;
    }
#endif
    
    static bool recvWorkerDecription(dopt::Socket& controlSocket, ClientDescription& desc)
    {
        bool result = true;
        
        ControlSignals nextCommand = ControlSignals();

        result &= controlSocket.recvData(&nextCommand, sizeof(nextCommand));
        //uint64_t sz = controlSocket.getUint64();
        uint64_t sz = controlSocket.getUnsignedVaryingInteger();
        //controlSocket.recvData(&sz, sizeof(sz));

        assert(nextCommand == ControlSignals::sig_response_worker_desciption);
        assert(sz == 4 * (32/8) + 1);

        result &= controlSocket.recvData(&desc.clientId, sizeof(desc.clientId));
        result &= controlSocket.recvData(&desc.dimension, sizeof(desc.dimension));
        result &= controlSocket.recvData(&desc.samplesInClient, sizeof(desc.samplesInClient));
        result &= controlSocket.recvData(&desc.rounds, sizeof(desc.rounds));
        result &= controlSocket.recvData(&desc.hasInterceptTerm, sizeof(desc.hasInterceptTerm));
        
        return result;
    }

    static bool recieveVector(dopt::Socket& controlSocket, VectorND_d& result)
    {
        bool resBool = true;

        ControlSignals nextCommand = ControlSignals();
        resBool &= controlSocket.recvData(&nextCommand, sizeof(nextCommand));

        //uint64_t sz = 0;
        //resBool &= controlSocket.recvData(&sz, sizeof(sz));
        //uint64_t sz = controlSocket.getUint64();
        
        uint64_t sz = controlSocket.getUnsignedVaryingInteger();
        
        uint64_t szInElements = sz / sizeof(VectorND_d::TElementType);
        
        if (result.size() != szInElements)
        {
            result = VectorND_d(szInElements);
        }
        
        resBool &= controlSocket.recvData(result.data(), result.sizeInBytes());
        
        return resBool;
    }

    dopt::MutableData scratchBuffer;                         ///< Scratch buffer
    dopt::Socket* control = 0;                               ///< Socket to send all information
 };

struct GDControlBlock
{
#if D_OPT_USE_ATOMICS_IN_DISTRIB_IMPL
    std::atomic<bool> gradIsReady;  ///< Signal setuped by worker when result is ready
    std::atomic<bool> fiIsReady;    ///< Signal setuped by worker when result is ready
#else
    bool gradIsReady;               ///< Signal setuped by worker when result is ready
    bool fiIsReady;                 ///< Signal setuped by worker when result is ready
#endif
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

    bool send_fi_from_worker;            ///< Flag denoted to fact that worker should compute fi and fill messageToSendFi.
    
    VectorND_d::TElementType fiValue;                      ///< Function value in current iterate


    // dopt::Socket* messageToSendGradientsSocket = 0;        ///< Uncompressed information about gradients
    // dopt::Socket* messageToSendFiSocket = 0;               ///< Function value in current iterate
    // dopt::Socket* control = 0;
    // dopt::Socket* auxInformation = 0;

    // Control socket signals
    enum ControlSignalsGD : uint8_t
    {
        sig_messageToSendGradient = 30,
        sig_messageToSendFi       = 31
    };

    void prepareUpdate(dopt::MutableData& buffer, ControlSignalsGD ctrl) 
    {
        switch (ctrl)
        {
            case ControlSignalsGD::sig_messageToSendGradient:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());
                buffer.putBytes(localGradient.dataConst(), localGradient.sizeInBytes());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());
                buffer.putBytes(localGradient.dataConst(), localGradient.sizeInBytes());               
#endif
                ctrBlock.gradIsReady = true;
                break;
            }

            case ControlSignalsGD::sig_messageToSendFi:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(fiValue));
                buffer.putBytes(&fiValue, sizeof(fiValue));
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(fiValue));                
                buffer.putValueToStream(fiValue);
#endif
                ctrBlock.fiIsReady = true;
                break;
            }

            default:
            {
                assert(!"ERROR");
            }
        }
    }

    void sendUpdate(ControlSignalsGD ctrl) 
    {
        switch (ctrl)
        {
            case ControlSignalsGD::sig_messageToSendGradient:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(localGradient.sizeInBytes());
                control->sendData(localGradient.dataConst(), localGradient.sizeInBytes());
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
                control->sendData(localGradient.dataConst(), localGradient.sizeInBytes());
#endif
                ctrBlock.gradIsReady = true;
                break;
            }

            case ControlSignalsGD::sig_messageToSendFi:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(sizeof(fiValue));
                control->sendData(&fiValue, sizeof(fiValue));
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(sizeof(fiValue));
                scratchBuffer.putValueToStream(fiValue);
                
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
#endif
                ctrBlock.fiIsReady = true;
                break;
            }

            default:
            {
                assert(!"ERROR");
            }
        }
    }

    template<bool blockForFirstRecv = false>
    void receiveUpdates(dopt::Socket* control)
    {
        //=====================================================================
        if constexpr (blockForFirstRecv)
        {
            // Do nothing. No need to check available bytes
        }
        else
        {
            // Non-blocking call. If no bytes available return.
            if (control->getAvailableBytesForRead() == 0)
                return/*false*/;
        }
        //=====================================================================
        
        // for (;;)
        {
            uint8_t ctrl = 0;
            
            if (control->recvData(&ctrl, sizeof(ctrl)))
            {
                //uint64_t sz = 0;
                //control->recvData(&sz, sizeof(sz));                

                //uint64_t sz = control->getUint64();
                uint64_t sz = control->getUnsignedVaryingInteger();
                uint64_t szInElements = sz / sizeof(VectorND_d::TElementType);

                assert(sz % sizeof(VectorND_d::TElementType) == 0);

                switch ((ControlSignalsGD)ctrl)
                {
                    case ControlSignalsGD::sig_messageToSendGradient:
                    {
                        if (localGradient.size() != szInElements)
                            localGradient.resize(szInElements);
                        
                        dopt::Socket* messageToSendGradientsSocket = control;
                        messageToSendGradientsSocket->recvData(localGradient.data(), localGradient.sizeInBytes());
                        
                        ctrBlock.gradIsReady = true;
                        break;
                    }

                    case ControlSignalsGD::sig_messageToSendFi:
                    {
                        dopt::Socket* messageToSendFiSocket = control;
                        messageToSendFiSocket->recvData(&fiValue, sizeof(fiValue));
                        
                        ctrBlock.fiIsReady = true;
                        break;
                    }

                    default:
                    {
                        assert(!"ERROR");
                    }
                }                
            }

            //if (control->getAvailableBytesForRead() == 0)
            //{
            //    break;
            //}
        }

        return /*true*/;
    }
};

struct WorkerContextForFedNLControlBlock
{
#if D_OPT_USE_ATOMICS_IN_DISTRIB_IMPL
    std::atomic<bool> messageToSendGradientsIsReady;         ///< Signal setuped by worker when gradients are ready
    std::atomic<bool> messageToSendHessiansIndiciesIsReady;  ///< Signal setuped by worker when hessians are ready
    std::atomic<bool> messageToSendHessiansItemsIsReady;     ///< Signal setuped by worker when hessians are ready
    std::atomic<bool> messageToSendHessiansIndiciesIsLastChunk;
    std::atomic<bool> messageToSendHessiansItemsIsLastChunk;
    std::atomic<bool> messageToSendLkIsReady;                ///< Signal setuped by worker when Lk is ready
    std::atomic<bool> messageToSendFiIsReady;                ///< Signal setuped by worker when fi are ready
    std::atomic<bool> messageToSendRoundWorkHasBeenFinished; ///< Signal setuped by worker when update is ready
    std::atomic<bool> messageToSendLearningHessiansIsReady;  ///< Learning hessian has been sent [for Debug]
#else
    bool messageToSendGradientsIsReady;         ///< Signal setuped by worker when gradients are ready
    bool messageToSendHessiansIndiciesIsReady;  ///< Signal setuped by worker when hessians are ready
    bool messageToSendHessiansItemsIsReady;     ///< Signal setuped by worker when hessians are ready
    bool messageToSendHessiansIndiciesIsLastChunk;
    bool messageToSendHessiansItemsIsLastChunk;
    bool messageToSendLkIsReady;                ///< Signal setuped by worker when Lk is ready
    bool messageToSendFiIsReady;                ///< Signal setuped by worker when fi are ready
    bool messageToSendRoundWorkHasBeenFinished; ///< Signal setuped by worker when update is ready
    bool messageToSendLearningHessiansIsReady;  ///< Learning hessian has been sent [for Debug]
#endif
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
    , raise_finish_flag_for_worker(false)
    , messageToSendLk(0.0)
    , messageToSendFi(0.0)
    , ctrBlock{}
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
    VectorND_d localGradient;          ///< Result from last iteration (gradient direction)
    MatrixNMD_d learningHessian;       ///< Local Hessian (learnable)

    // External (connection with outside world)
    // dopt::MutableData messageToSendGradients;     ///< Uncompressed information about gradients
    VectorND_d::TElementType messageToSendLk;        ///< Auxilirary information to send (one scalar Lk)
    VectorND_d::TElementType messageToSendFi;        ///< Function value in current iterate (one scalar)

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

    bool raise_finish_flag_for_worker;      ///< Flag which specify that client should raise flag for master that work is finished in control block.

    //======================================================================================================================//

    WorkerContextForFedNLControlBlock ctrBlock;

    //======================================================================================================================//
    // External (connection with outside world)
    //dopt::Socket* messageToSendGradientsSocket = 0;        ///< Uncompressed information about gradients
    //dopt::Socket* messageToSendLkSocket = 0;               ///< Auxilirary information to send (one scalar Lk)
    //dopt::Socket* messageToSendFiSocket = 0;               ///< Function value in current iterate
    //dopt::Socket* messageToSendHessiansIndiciesSocket = 0; ///< Compressed information about Hessian items indicies that are going to be send to master
    //dopt::Socket* messageToSendHessiansItemsSocket = 0;    ///< Compressed information about Hessian items values that are going to be send to master
    //dopt::Socket* learningHessianSocket = 0;               ///< Debug socket to send Hessian from clients
    //dopt::Socket* auxInformation = 0;                      ///< Aux information

    //dopt::Socket* control = 0;                               ///< Socket to send all information

    // Control
    enum ControlSignals : uint8_t
    {
        sig_messageGradient = 41,

        sig_messageToSendHessiansIndicies = 42,
        sig_messageToSendHessiansIndiciesLastChunk = 43,

        sig_messageToSendHessiansItems = 44,
        sig_messageToSendHessiansItemsLastChunk = 45,

        sig_messageToSendLearningHessians = 46,
        sig_messageToSendLk = 47,
        sig_messageToSendFi = 48,
        
        sig_messageToSendRoundWorkHasBeenFinished = 49
    };
    
    void prepareUpdate(dopt::MutableData& buffer, ControlSignals ctrl)
    {
        switch (ctrl)
        {
            case sig_messageGradient:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());
                buffer.putBytes(localGradient.dataConst(), localGradient.sizeInBytes());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());
                buffer.putBytes( localGradient.dataConst(), localGradient.sizeInBytes() );
#endif
                ctrBlock.messageToSendGradientsIsReady = true;
                
                break;
            }
            
            case sig_messageToSendHessiansIndicies: 
            case sig_messageToSendHessiansIndiciesLastChunk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(messageToSendHessiansIndicies.getFilledSize());
                buffer.putBytes(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(messageToSendHessiansIndicies.getFilledSize());
                buffer.putBytes(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());
#endif
                ctrBlock.messageToSendHessiansIndiciesIsReady = true;
                ctrBlock.messageToSendHessiansIndiciesIsLastChunk = (ctrl == sig_messageToSendHessiansIndiciesLastChunk ? true : false);
                break;
            }

            case sig_messageToSendHessiansItems:
            case sig_messageToSendHessiansItemsLastChunk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(messageToSendHessiansItems.getFilledSize());
                buffer.putBytes(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(messageToSendHessiansItems.getFilledSize());
                buffer.putBytes(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());
#endif
                ctrBlock.messageToSendHessiansItemsIsReady = true;
                ctrBlock.messageToSendHessiansItemsIsLastChunk = (ctrl == sig_messageToSendHessiansItemsLastChunk ? true : false);
                break;
            }

            case sig_messageToSendLearningHessians:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putMatrixItems(learningHessian);
#else
                buffer.putByte(ctrl);
                buffer.putMatrixItems(learningHessian);
#endif
                ctrBlock.messageToSendLearningHessiansIsReady = true;
                break;
            }
            
            case sig_messageToSendLk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(messageToSendLk));
                buffer.putBytes(&messageToSendLk, sizeof(messageToSendLk));
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(messageToSendLk));
                buffer.putValueToStream(messageToSendLk);
#endif
                ctrBlock.messageToSendLkIsReady = true;
                break;
            }
            
            case sig_messageToSendFi:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(messageToSendFi));
                buffer.putBytes(&messageToSendFi, sizeof(messageToSendFi));
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingInteger(sizeof(messageToSendFi));
                buffer.putBytes(&messageToSendFi, sizeof(messageToSendFi));
#endif
                ctrBlock.messageToSendFiIsReady = true;
                break;
            }
            
            case sig_messageToSendRoundWorkHasBeenFinished:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
#else
                buffer.putByte(ctrl);
                buffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
#endif
                ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
                break;
            }
            
            default:
            {
                assert(!"ERROR");
            }
        }
    }

    void sendUpdate(ControlSignals ctrl)
    {
        switch (ctrl)
        {
            case sig_messageGradient:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(localGradient.sizeInBytes());
                control->sendData(localGradient.dataConst(), localGradient.sizeInBytes());
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(localGradient.sizeInBytes());

                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
                control->sendData(localGradient.dataConst(), localGradient.sizeInBytes());
#endif
                ctrBlock.messageToSendGradientsIsReady = true;
                
                break;
            }
            
            case sig_messageToSendHessiansIndicies:
            case sig_messageToSendHessiansIndiciesLastChunk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(messageToSendHessiansIndicies.getFilledSize());
                control->sendData(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(messageToSendHessiansIndicies.getFilledSize());
                scratchBuffer.putBytes(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());
                
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
                //control->sendData(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());
#endif
                ctrBlock.messageToSendHessiansIndiciesIsReady = true;
                ctrBlock.messageToSendHessiansIndiciesIsLastChunk = (ctrl == sig_messageToSendHessiansIndiciesLastChunk ? true : false);
                break;
            }

            case sig_messageToSendHessiansItems:
            case sig_messageToSendHessiansItemsLastChunk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES

                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(messageToSendHessiansItems.getFilledSize());
                control->sendData(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(messageToSendHessiansItems.getFilledSize());

                scratchBuffer.putBytes(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
                //control->sendData(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());
#endif
                ctrBlock.messageToSendHessiansItemsIsReady = true;
                ctrBlock.messageToSendHessiansItemsIsLastChunk = (ctrl == sig_messageToSendHessiansItemsLastChunk ? true : false);
                break;
            }

            case sig_messageToSendLearningHessians:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                
                control->sendByte(ctrl);
                control->sendMatrixItems(learningHessian);
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putMatrixItems(learningHessian);
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
#endif
                ctrBlock.messageToSendLearningHessiansIsReady = true;
                break;
            }
            
            case sig_messageToSendLk:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES

                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(sizeof(messageToSendLk));
                control->sendData(&messageToSendLk, sizeof(messageToSendLk));
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(sizeof(messageToSendLk));
                scratchBuffer.putValueToStream(messageToSendLk);
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
#endif
                ctrBlock.messageToSendLkIsReady = true;
                break;
            }
            
            case sig_messageToSendFi:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingInteger(sizeof(messageToSendFi));
                control->sendData(&messageToSendFi, sizeof(messageToSendFi));
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingInteger(sizeof(messageToSendFi));
                scratchBuffer.putValueToStream(messageToSendFi);

                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
#endif
                ctrBlock.messageToSendFiIsReady = true;
                break;
            }
            
            case sig_messageToSendRoundWorkHasBeenFinished:
            {
#if !D_OPT_PACK_TRANSFERED_MESSAGES
                control->sendByte(ctrl);
                control->sendUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
#else
                scratchBuffer.rewindToStart();
                scratchBuffer.putByte(ctrl);
                scratchBuffer.putUnsignedVaryingIntegerKnowAtCompileTime<uint8_t, uint8_t(0)> ();
                control->sendData(scratchBuffer.getPtr(), scratchBuffer.getFilledSize());
#endif
                ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
                break;
            }
            
            default:
            {
                assert(!"ERROR");
            }
        }
    }

    template<bool blockForFirstRecv = false>
    void receiveUpdates(dopt::Socket* control)
    {
        //=====================================================================
        if (blockForFirstRecv)
        { 
            // Do nothing. No need to check available bytes
        }
        else
        {
            // Non-blocking call. If no bytes available return.
            if (control->getAvailableBytesForRead() == 0)
                return/*false*/;
        }
        //=====================================================================
        //for (;;)
        {
            {
                uint8_t ctrl = 0;
            
                if (control->recvData(&ctrl, sizeof(ctrl)))
                {
                    //uint64_t sz = 0;
                    //control->recvData(&sz, sizeof(sz));
                    // uint64_t sz = control->getUint64();
                    uint64_t sz = control->getUnsignedVaryingInteger();

                    switch ((ControlSignals)ctrl)
                    {
                        case sig_messageGradient:
                        {
                            // messageToSendGradients.seekStart(sz);
                            dopt::Socket* messageToSendGradientsSocket = control;
                            // messageToSendGradientsSocket->recvData(messageToSendGradients.getPtr(), messageToSendGradients.getFilledSize());

                            assert( sz % sizeof(VectorND_d::TElementType) == 0);

                            uint64_t szInComponents = sz / sizeof(VectorND_d::TElementType);
                            
                            if (localGradient.size() != szInComponents)
                                localGradient.resize(szInComponents);
                            
                            messageToSendGradientsSocket->recvData( localGradient.data(), localGradient.sizeInBytes() );

                            ctrBlock.messageToSendGradientsIsReady = true;

                            break;
                        }
                    
                        case sig_messageToSendHessiansIndicies:
                        case sig_messageToSendHessiansIndiciesLastChunk:
                        {
                            messageToSendHessiansIndicies.seekStart(sz);

                            dopt::Socket* messageToSendHessiansIndiciesSocket = control;
                            messageToSendHessiansIndiciesSocket->recvData(messageToSendHessiansIndicies.getPtr(), messageToSendHessiansIndicies.getFilledSize());

                            ctrBlock.messageToSendHessiansIndiciesIsReady = true;
                            ctrBlock.messageToSendHessiansIndiciesIsLastChunk = (ctrl == sig_messageToSendHessiansIndiciesLastChunk ? true : false);
                            break;
                        }

                        case sig_messageToSendHessiansItems:
                        case sig_messageToSendHessiansItemsLastChunk:
                        {
                            messageToSendHessiansItems.seekStart(sz);

                            dopt::Socket* messageToSendHessiansItemsSocket = control;
                            messageToSendHessiansItemsSocket->recvData(messageToSendHessiansItems.getPtr(), messageToSendHessiansItems.getFilledSize());

                            ctrBlock.messageToSendHessiansItemsIsReady = true;
                            ctrBlock.messageToSendHessiansItemsIsLastChunk = (ctrl == sig_messageToSendHessiansItemsLastChunk ? true : false);
                            break;
                        }

                        case sig_messageToSendLearningHessians:
                        {
                            assert(learningHessian.sizeInBytesNoPadding() == sz);
                        
                            dopt::Socket* learningHessianSocket = control;
                            learningHessianSocket->recvMatrixItems(learningHessian);
                        
                            ctrBlock.messageToSendLearningHessiansIsReady = true;
                            break;
                        }

                        case sig_messageToSendLk:
                        {
                            dopt::Socket* messageToSendLkSocket = control;
                            messageToSendLkSocket->recvData(&messageToSendLk, sizeof(messageToSendLk));
                        
                            ctrBlock.messageToSendLkIsReady = true;
                            break;
                        }
                    
                        case sig_messageToSendFi:
                        {
                            dopt::Socket* messageToSendFiSocket = control;
                            messageToSendFiSocket->recvData(&messageToSendFi, sizeof(messageToSendFi));
                        
                            ctrBlock.messageToSendFiIsReady = true;
                            break;
                        }

                        case sig_messageToSendRoundWorkHasBeenFinished:
                        {
                            ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
                            break;
                        }

                        default:
                        {
                            assert(!"ERROR");
                        }
                    }
                }
            }

            //if (control->getAvailableBytesForRead() == 0)
            //{
            //    break;
            //}
        }
        //=====================================================================
        return/*true*/;
    }
};

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
