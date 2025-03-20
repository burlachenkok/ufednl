#pragma once

#include "dopt/system/include/threads/Thread.h"
#include "dopt/system/include/FloatUtils.h"

#include "UsedTypes.h"
#include <vector>

inline int32_t workerThreadTrainLoopForFedNL1_NaturalDitheringCompressor(void* arg1, void* arg2)
{
    const std::vector<WorkerContext*>& wctx = *(std::vector<WorkerContext*>*)(arg1);
    ServerContext* serverCtx = (ServerContext*)arg2;

    size_t ctxToProcess = wctx.size();

    size_t rounds = serverCtx->rounds;

    size_t d = serverCtx->getDimension();

    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [START]
    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        ctx->messageToSendHessiansItems.reserveMemory( ( ((d + 1) * (d)) / 2) * sizeof(MatrixNMD_d::TElementType));
        ctx->messageToSendGradients.reserveMemory(d * sizeof(VectorND_d::TElementType));
    }
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [END]

    for (size_t r = 0; r < rounds; ++r)
    {
        // Wait for permission by worker to start
#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
        serverCtx->startSempahore.acquire();
#endif
        for (;;)
        {           
            if (r == serverCtx->roundToStart)
                break;
            else
                dopt::DefaultThread::yeildCurrentThInHotLoop();
        }

        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
            
            // double preScaleBeforeSend2Master = ctx->preScaleBeforeSend2Master;

            // Step-1-a: Compute Local gradient at xi. Put uncompressed dense gradient
            const VectorND_d& xi = *(serverCtx->currenIterate);

            VectorND_d margin = ctx->optProblem->evaluateClassificationMargin(xi);
            VectorND_d margin_sigmoid = ctx->optProblem->evaluateClassificationMarginSigmoid(margin);

            ctx->localGradient = ctx->optProblem->evaluateGradient(xi, margin, margin_sigmoid);

            //ctx->messageToSendGradients.rewindToStart();
            ctx->messageToSendGradients.rewindAndPutPODs(ctx->localGradient.dataConst(), d);
            ctx->ctrBlock.messageToSendGradientsIsReady = true;

            // Step-1-b: Compute Hessian and compute Hessian difference.
#if DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian< /*bool kSymmetriseHessian*/ false >(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = MatrixNMD_d::computeDifferenceWithUpperTriangularPart(hessian_at_xi, ctx->learningHessian);
#else
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = (hessian_at_xi - ctx->learningHessian);
#endif
            // Step-2-a: Compress Hessian difference
            // Reset head in writing buffer
            //ctx->messageToSendHessiansIndicies.rewindToStart();

            ctx->messageToSendHessiansItems.rewindToStart();

            MatrixNMD_d differenceSparsified = MatrixNMD_d::getZeroSquareMatrix(d);
            
            // Put sparsifed hessian into the buffer and construct differenceSparsified
            size_t send_coordinates = 0;

            // Configuration
            //==========================================================================================
            constexpr uint32_t pNorm = 2;                  // norm for compression
            constexpr uint32_t sNumLevels = 15;            // 1.0/2^[0], 1.0/2^[1],..., 1/2.0^[s-1], 0.0
            constexpr uint32_t sNumActualLevels = sNumLevels + 1;
            constexpr uint32_t sNumLevelsBitsStorage = 4;  // number of bits to store [sNumLevels+1] levels dopt::log2IntCeil(sNumLevels+1);
            //==========================================================================================
            // struct CompressedItems
            // {
            //    unsigned int first_item_sign_is_positive : 1;
            //    unsigned int first_item_used_level : sNumLevelsBitsStorage;

            //    unsigned int first_item_sign_is_positive : 1;
            //    unsigned int first_item_used_level : sNumLevelsWithOneBits;
            //};
            
            // Step-1: Compute Lp Norm. Scan through upper triangular part.           
            double xp_norm = double();

            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
                    auto item = difference.get(i, j);
                    xp_norm += dopt::powerNatural(dopt::abs(item), pNorm);
                }
            }
            xp_norm = dopt::powerReal(xp_norm, double(1.0) / double(pNorm) );
            
            ctx->messageToSendHessiansItems.putDouble(xp_norm);

            // Step-2 and Step-3: Normalize and find need level
            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
                    auto item = difference.get(i, j);
                    auto itemAbsNormalized = dopt::abs(item) / xp_norm;

                    double itemNewValue = 0.0;
                    size_t itemLevel = 0;
                    bool itemIsPositive = (item >= 0.0 ? true : false);

                    // Compression per item
                    {
                        size_t L_prev = 0;
                        size_t L_cur = 1;
                        double L_prev_value = 1.0;
                        double L_cur_value = 1.0/2;
                       
                        for (; L_cur <= sNumLevels; ++L_prev, ++L_cur)
                        {
                            // Corner Case
                            if (L_cur == sNumLevels)
                                L_cur_value = 0.0;

                            // L_prev_value  -- L[u]
                            // L_cur_value   -- L[u+1]

                            if (L_cur_value < itemAbsNormalized)
                            {                                
                                double denom = L_prev_value - L_cur_value;
                                double num   = itemAbsNormalized - L_cur_value;

                                // TODO: sample
                                double randomNumber = 0.75;

                                
                                if (randomNumber < denom / num)
                                {
                                    // set level to Lu
                                    itemNewValue = L_prev_value;
                                    itemLevel = L_prev;
                                }
                                else
                                {
                                    itemNewValue = L_cur_value;
                                    itemLevel = L_cur;
                                }
                            }
                            else
                            {
                                L_prev_value = L_cur_value;
                                L_cur_value /= 2.0;
                            }
                        }                        
                    }      

                    double itemNewValueFromLevel;
                    
                    if (itemLevel == sNumLevels)
                    {
                        itemNewValueFromLevel = 0.0;
                    }
                    else
                    {
                        itemNewValueFromLevel = dopt::powerNatural(0.5, uint32_t(itemLevel));
                        if (!itemIsPositive)
                            itemNewValueFromLevel = -itemNewValueFromLevel;                            
                    }
                    
                    differenceSparsified.set(i, j, itemNewValueFromLevel);
                    differenceSparsified.set(j, i, itemNewValueFromLevel);                   
                    // TODO: remove itemNewValue -- only for debug
                    // (sNumLevels+1) bits for levels
                    // TODO: put item into stream
                    // ctx->messageToSendHessiansItems.putDouble(xp_norm);
                }
            }

#if 0
            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i)
                {
                    auto item             = difference.get(i, j);                    
                    auto packedItem       = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(item);
                    packedItem.components.mantissa = 0;                    
                    differenceSparsified.set(i, j, packedItem.real_value_repr);
                    differenceSparsified.set(j, i, packedItem.real_value_repr);

                    packs[filledPacks++] = packedItem;
                    
                    // Time to send packs to master
                    if (filledPacks == 2)
                    {
                        uint32_t packed2send24bits = pack2FP64NoMantissa(packs[0], packs[1]);                        
                        ctx->messageToSendHessiansItems.putBytes(&packed2send24bits, 3);
                        send_coordinates += 2;
                        filledPacks = 0;
                    }
                }
            }
            
            // For simplify parsing code in master, we send 2 packs at the end
            {
                uint32_t packed2send24bits = pack2FP64NoMantissa(packs[0], packs[1]);
                ctx->messageToSendHessiansItems.putBytes(&packed2send24bits, 3);
                send_coordinates += 2;                
                filledPacks = 0;
            }
            
            assert(ctx->kForCompressor + 1 == send_coordinates ||
                   ctx->kForCompressor + 2 == send_coordinates);
#endif
            ctx->ctrBlock.messageToSendHessiansIsReady = true;

            ctx->learningHessian += ctx->alpha * differenceSparsified;

            // Put Lk
            if (ctx->send_Lk_from_worker)
            {
                ctx->messageToSendLk.rewindAndPutDouble(difference.frobeniusNormForSymmetricMatrixFromUpPart());
                ctx->ctrBlock.messageToSendLkIsReady = true;
            }

            // Put fi
            if (ctx->send_fi_from_worker)
            {
                double fi = ctx->optProblem->evaluateFunction(xi, margin, margin_sigmoid);
                ctx->messageToSendFi.rewindAndPutDouble(fi);
                ctx->ctrBlock.messageToSendFiIsReady = true;
            }

            if (ctx->raise_finish_flag_for_worker)
            {
                ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
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

                        double fi = ctx->optProblem->evaluateFunction(xi, margin, margin_sigmoid);
                        ctx->messageToSendFi.rewindAndPutDouble(fi);
                        ctx->ctrBlock.messageToSendFiIsReady = true;
                    }

                    ++k;
                }
                else
                {
                    dopt::DefaultThread::yeildCurrentThInHotLoop();
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
