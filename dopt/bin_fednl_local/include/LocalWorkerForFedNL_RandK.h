#pragma once

#include "dopt/system/include/threads/Thread.h"
#include "dopt/math_routines/include/SpecialMathRoutinesForMatrix.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"

#include "UsedTypes.h"
#include <vector>

inline int32_t workerThreadTrainLoopForFedNL1_RandKCompressor(void* arg1, void* arg2)
{
    const std::vector<WorkerContext*>& wctx = *(std::vector<WorkerContext*>*)(arg1);
    ServerContext* serverCtx = (ServerContext*)arg2;

    size_t ctxToProcess = wctx.size();

    size_t rounds = serverCtx->rounds;
    size_t d = serverCtx->getDimension();
    size_t totalPossibleItems2Send = ((d + 1) * d) / 2;

    std::vector<uint32_t> kcoordinatesForUpperTriangPart;

    std::vector<dopt::RandomGenIntegerLinear> generators(ctxToProcess);
    std::vector<std::vector<uint32_t>> kcoordinatesPerCtx(ctxToProcess);

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        generators[ctxIndex].setSeed(ctx->seedForRandomizedCompressor);
        ctx->localGradient = VectorND_d();
    }

    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [START]
    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        ctx->messageToSendHessiansItems.reserveMemory(ctx->kForCompressor * sizeof(MatrixNMD_d::TElementType));
        ctx->messageToSendGradients.reserveMemory(d * sizeof(VectorND_d::TElementType));
        if (ctx->transfer_indicies_for_randk)
            ctx->messageToSendHessiansIndicies.reserveMemory(ctx->kForCompressor * sizeof(uint32_t));
    }
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [END]

    for (size_t r = 0; r < rounds; ++r)
    {
        // While waiting for master generate need indicies for all contexts
        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];

            if (kcoordinatesForUpperTriangPart.empty())
            {
                kcoordinatesForUpperTriangPart = dopt::indiciesForUpperTriangularPart(ctx->learningHessian);
            }

            size_t kForCompressor = ctx->kForCompressor;
            
            kcoordinatesPerCtx[ctxIndex] = dopt::generateRandKItemsInUpperTriangularPart(generators[ctxIndex], 
                                                                                         kForCompressor,
                                                                                         ctx->learningHessian, 
                                                                                         kcoordinatesForUpperTriangPart);
        }
        
        // Wait for permission by worker to start
#if DOPT_USE_SEMPAHORE_DURING_SYNC_IN_CLIENTS
        serverCtx->startSempahore.acquire();
#endif

        for (;;)
        {
            if (r == serverCtx->roundToStart)
            {
                break;
            }
            else
            {
                dopt::DefaultThread::yeildCurrentThInHotLoop();
            }
        }
        
        // Start round
        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
            double preScaleBeforeSend2Master = ctx->preScaleBeforeSend2Master;
            double alpha = ctx->alpha;
            
            size_t kForCompressor = ctx->kForCompressor;

            double compressorMultiplier = double(totalPossibleItems2Send) / double(kForCompressor);

            const VectorND_d& xi = *(serverCtx->currenIterate);
            VectorND_d margin = ctx->optProblem->evaluateClassificationMargin(xi);
            VectorND_d margin_sigmoid = ctx->optProblem->evaluateClassificationMarginSigmoid(margin);

            ctx->localGradient = ctx->optProblem->evaluateGradient(xi, margin, margin_sigmoid);
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
            // Put Lk
            if (ctx->send_Lk_from_worker)
            {
                ctx->messageToSendLk.rewindAndPutDouble(difference.frobeniusNormForSymmetricMatrixFromUpPart());
                ctx->ctrBlock.messageToSendLkIsReady = true;
            }

            const std::vector<uint32_t>& kcoordinates = kcoordinatesPerCtx[ctxIndex];
            
            assert(ctx->kForCompressor == kcoordinates.size());

            if (ctx->transfer_indicies_for_randk)
            {
                ctx->messageToSendHessiansIndicies.rewindAndPutPODs(&kcoordinates[0], ctx->kForCompressor);
            }

            ctx->messageToSendHessiansItems.rewindToStart();
            
            // (apply need scaling) and put sparsifed hessian into the buffer
            for (size_t k = 0; k < ctx->kForCompressor; ++k)
            {
                size_t index = kcoordinates[k];
                const auto compressedItem = difference.matrixByCols.get(index) * compressorMultiplier;
                ctx->messageToSendHessiansItems.putDouble(compressedItem * preScaleBeforeSend2Master);
            }
            ctx->ctrBlock.messageToSendHessiansIsReady = true;

            // Learn hessian locally. In-place update.
            for (size_t k = 0; k < ctx->kForCompressor; ++k)
            {
                size_t indexFirst = kcoordinates[k];

#if DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL
                const auto compressedItem = difference.matrixByCols.get(indexFirst) * compressorMultiplier;
                ctx->learningHessian.matrixByCols[indexFirst] += alpha * compressedItem;
#else
                size_t iRow = 0;
                size_t iCol = 0;
                difference.getPositionFromFlatternIndex(iRow, iCol, indexFirst);

                const auto compressedItem = difference.matrixByCols.get(indexFirst) * compressorMultiplier;

                if (iRow == iCol)
                {
                    ctx->learningHessian.matrixByCols[indexFirst] += alpha * compressedItem;
                }
                else
                {
                    ctx->learningHessian.matrixByCols[indexFirst]  += alpha * compressedItem;

                    size_t indexSecond = difference.getFlattenIndexFromPosition(iCol, iRow);
                    ctx->learningHessian.matrixByCols[indexSecond] += alpha * compressedItem;
                }
#endif
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
