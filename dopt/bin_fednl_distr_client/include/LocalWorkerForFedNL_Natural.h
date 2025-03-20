#pragma once

#include "dopt/system/include/threads/Thread.h"
#include "dopt/math_routines/include/SpecialMathRoutinesForMatrix.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include "UsedTypes.h"
#include <vector>

#ifndef DOPT_POSTPONE_NAT_COMPRESS_AFTER_SEND
    #define DOPT_POSTPONE_NAT_COMPRESS_AFTER_SEND 0
#endif

inline int32_t workerThreadTrainLoopForFedNL1_NaturalCompressor(void* arg1, void* arg2)
{
    //========================================================================================================
    // Typical size for TCPv4/IPv4 header sizes
    constexpr size_t kMaxIPHeaderSizeV6 = 40;
    constexpr size_t kMaxIPHeaderSizeV4 = 20;
    constexpr size_t kMaxIPHeaderSize = (kMaxIPHeaderSizeV6 > kMaxIPHeaderSizeV4) ? (kMaxIPHeaderSizeV6) : (kMaxIPHeaderSizeV4);
    constexpr size_t kTCPHeaderSize = 20;

    // The MTU of the link - layer protocol places a hard limit on the length of an IP datagram
    constexpr size_t kMtuSize = D_OPT_NETWORK_MTU_SIZE;

    // For evaluate kChunkSizeForIndicies size we need exclude TCP/IP header bytes and also our 3 bytes header: 1 byte command, and 2 bytes most likely for the length
    constexpr size_t kChunkSizeForIndicies = (kMtuSize -
                                              kMaxIPHeaderSize -
                                              kTCPHeaderSize - 1 /*packet type*/ - 2 /*packet length 16 bits will allow to have 2^14=16384 length packet*/)
                                              /
                                              sizeof(uint32_t);

    constexpr size_t kChunkSizeForItems = (kMtuSize -
                                           kMaxIPHeaderSize -
                                           kTCPHeaderSize - 1 /*packet type*/ - 2 /*packet length 16 bits will allow to have 2^14=16384 length packet*/)
                                           /
                                           sizeof(MatrixNMD_d::TElementType);

    constexpr size_t kChunkSize = kChunkSizeForIndicies;
    //========================================================================================================

    const std::vector<WorkerContext*>& wctx = *(std::vector<WorkerContext*>*)(arg1);
    ServerContext* serverCtx = (ServerContext*)arg2;

    size_t ctxToProcess = wctx.size();

    size_t rounds = serverCtx->rounds;
    size_t d = serverCtx->getDimension();

    std::vector<dopt::MutableData> fiBuffer(ctxToProcess);
    std::vector<decltype(WorkerContextForFedNL::messageToSendFi)*> fiBufferPtrs(ctxToProcess);

    std::vector<dopt::MutableData> LkBuffer(ctxToProcess);
    std::vector<decltype(WorkerContextForFedNL::messageToSendLk)*> LkBufferPtrs(ctxToProcess);

    //========================================================================================================
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [START]
    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];

        ctx->messageToSendHessiansItems.reserveMemory(kChunkSize * sizeof(MatrixNMD_d::TElementType));
        ctx->prepareUpdate(fiBuffer[ctxIndex], WorkerContextForFedNL::sig_messageToSendFi);
        fiBufferPtrs[ctxIndex] = reinterpret_cast <decltype(WorkerContextForFedNL::messageToSendFi)*> (fiBuffer[ctxIndex].getPtr() + fiBuffer[ctxIndex].getFilledSize() - sizeof(WorkerContextForFedNL::messageToSendFi));
        
        ctx->prepareUpdate(LkBuffer[ctxIndex], WorkerContextForFedNL::sig_messageToSendLk);
        LkBufferPtrs[ctxIndex] = reinterpret_cast <decltype(WorkerContextForFedNL::messageToSendLk)*> (LkBuffer[ctxIndex].getPtr() + LkBuffer[ctxIndex].getFilledSize() - sizeof(WorkerContextForFedNL::messageToSendLk));
    }
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [END]
    //========================================================================================================


    for (size_t r = 0; /*r < rounds*/; ++r)
    {
        // Wait for permission by worker to start. Potential problems if use the same sever context for multiple workers/threads
        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];

            // Wait for permission by worker to start
            for (;;)
            {
                serverCtx->receiveUpdates</*block*/true>(ctx->control);

                if (serverCtx->get_hessian)
                {
                    serverCtx->get_hessian = false;

                    dopt::MutableData buffer;
                    buffer.putByte(ServerContext::ControlSignals::sig_response_full_hessian_in_client);
                    buffer.putMatrixItems(ctx->learningHessian);
                    ctx->control->sendData(buffer.getPtr(), buffer.getFilledSize());
                }

                if (r == serverCtx->roundToStart || serverCtx->terminate)
                {
                    // no need to reset roundToStart because it's incremented in each round
                    // no need to reset terminate because terminate means the halt of the client
                    break;
                }
            }
        }

        // Terminate if it has been requested
        if (serverCtx->terminate)
            break;

        // Start round
        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];

            double preScaleBeforeSend2Master = ctx->preScaleBeforeSend2Master;

            const VectorND_d& xi = *(serverCtx->currenIterate);
            VectorND_d margin = ctx->optProblem->evaluateClassificationMargin(xi);
            VectorND_d margin_sigmoid = ctx->optProblem->evaluateClassificationMarginSigmoid(margin);

            ctx->localGradient = ctx->optProblem->evaluateGradient(xi, margin, margin_sigmoid);
            ctx->sendUpdate(WorkerContextForFedNL::sig_messageGradient);

            // Step-1-b: Compute Hessian and compute Hessian difference.
#if DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian< /*bool kSymmetriseHessian*/ false >(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = MatrixNMD_d::computeDifferenceWithUpperTriangularPart(hessian_at_xi, ctx->learningHessian);
#else
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = (hessian_at_xi - ctx->learningHessian);
#endif

            // Put Lk
            if (ctx->send_Lk_from_worker)
            {
                *(LkBufferPtrs[ctxIndex]) = difference.frobeniusNormForSymmetricMatrixFromUpPart();
                ctx->control->sendData(LkBuffer[ctxIndex].getPtr(), LkBuffer[ctxIndex].getFilledSize());
                ctx->messageToSendLk = *(LkBufferPtrs[ctxIndex]);
            }

#if !DOPT_POSTPONE_NAT_COMPRESS_AFTER_SEND
            MatrixNMD_d differenceSparsified = difference;
            differenceSparsified.applyNaturalCompressor();
#endif
            
            // Step-1-c: Generate Compression pattern
            ctx->messageToSendHessiansIndicies.rewindToStart();
            size_t send_coordinates_current_chunk = 0;

            // Scan through upper triangular part
            typedef dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, VectorND_d::TElementType> FPCompPack;
            FPCompPack packs[2];
            size_t filledPacks = 0;

            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
#if !DOPT_POSTPONE_NAT_COMPRESS_AFTER_SEND
                    // apply compressor and then get item
                    auto item = differenceSparsified.get(i, j);
                    auto packedItem = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item);
                    assert(packedItem.components.mantissa == 0);
#else
                    // get item before applying compressor
                    auto item = difference.get(i, j);
                    auto packedItem = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item);
#endif

                    packs[filledPacks++] = packedItem;

                    // Time to send packs to master
                    if (filledPacks == 2)
                    {
                        uint32_t packed2send24bits = pack2FP64NoMantissa(packs[0], packs[1]);
                        ctx->messageToSendHessiansItems.putBytes(&packed2send24bits, 3);
                        send_coordinates_current_chunk += 2;
                        filledPacks = 0;
                    }
                    
                    // Flush current chunk
                    if (send_coordinates_current_chunk == kChunkSize || send_coordinates_current_chunk == kChunkSize - 1)
                    {
                        ctx->sendUpdate(WorkerContextForFedNL::sig_messageToSendHessiansIndicies);
                        ctx->sendUpdate(WorkerContextForFedNL::sig_messageToSendHessiansItems);
                        ctx->messageToSendHessiansIndicies.rewindToStart();
                        ctx->messageToSendHessiansItems.rewindToStart();
                        send_coordinates_current_chunk = 0;
                    }
                }
            }

            // For simplify parsing code in master, we send 2 packs at the end (client should inore if no need)
            {
                uint32_t packed2send24bits = pack2FP64NoMantissa(packs[0], packs[1]);
                ctx->messageToSendHessiansItems.putBytes(&packed2send24bits, 3);
                send_coordinates_current_chunk += 2;
                filledPacks = 0;

                ctx->sendUpdate(WorkerContextForFedNL::sig_messageToSendHessiansIndiciesLastChunk);
                ctx->sendUpdate(WorkerContextForFedNL::sig_messageToSendHessiansItemsLastChunk);
                ctx->messageToSendHessiansIndicies.rewindToStart();
                ctx->messageToSendHessiansItems.rewindToStart();
                send_coordinates_current_chunk = 0;
            }
            
            // Put fi
            if (ctx->send_fi_from_worker)
            {
                *(fiBufferPtrs[ctxIndex]) = ctx->optProblem->evaluateFunction(xi, margin, margin_sigmoid);
                ctx->control->sendData(fiBuffer[ctxIndex].getPtr(), fiBuffer[ctxIndex].getFilledSize());
                ctx->messageToSendFi = *(fiBufferPtrs[ctxIndex]);
            }

            // Learn hessian locally. In-place update.
#if !DOPT_POSTPONE_NAT_COMPRESS_AFTER_SEND
            ;
#else
            MatrixNMD_d differenceSparsified = difference;
            differenceSparsified.applyNaturalCompressor();
#endif
            ctx->learningHessian += ctx->alpha * differenceSparsified;

            if (ctx->raise_finish_flag_for_worker)
            {
                ctx->sendUpdate(WorkerContextForFedNL::sig_messageToSendRoundWorkHasBeenFinished);
            }
        }

        // Line search process
        {
            for (size_t k = 0; ;)
            {
                if (serverCtx->lineSearchRound > r)
                {
                    break;
                }
                else if (k == serverCtx->lineSearchIteration)
                {
                    const VectorND_d& xi = *(serverCtx->currenIterate);

                    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
                    {
                        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
                        VectorND_d margin = ctx->optProblem->evaluateClassificationMargin(xi);
                        VectorND_d margin_sigmoid = ctx->optProblem->evaluateClassificationMarginSigmoid(margin);

                        *(fiBufferPtrs[ctxIndex]) = ctx->optProblem->evaluateFunction(xi, margin, margin_sigmoid);
                        ctx->control->sendData(fiBuffer[ctxIndex].getPtr(), fiBuffer[ctxIndex].getFilledSize());
                        ctx->messageToSendFi = *(fiBufferPtrs[ctxIndex]);
                    }

                    ++k;
                }
                else
                {
                    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
                    {
                        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
                        serverCtx->receiveUpdates</*block*/true>(ctx->control);
                        //break;
                    }
                }
            }
        }
    }

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        ctx->localGradient = VectorND_d();
    }

    return 0;
}
