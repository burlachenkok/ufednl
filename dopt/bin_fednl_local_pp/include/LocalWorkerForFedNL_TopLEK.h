#pragma once

#include "dopt/system/include/threads/Thread.h"
#include "dopt/math_routines/include/SpecialMathRoutinesForMatrix.h"

#include "UsedTypes.h"
#include <vector>

inline int32_t workerThreadTrainLoopForFedNL1_TopLEKCompressor(void* arg1, void* arg2)
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
        const VectorND_d& x0 = *(serverCtx->currenIterate);

        ctx->learningHessian = ctx->optProblem->evaluateHessian(x0);
        ctx->localLk = 0;
        ctx->localGradient = MatrixNMD_d::matrixVectorMultiply(ctx->learningHessian, x0)  /* + ctx->localLk * x0 */ - ctx->optProblem->evaluateGradient(x0);

        ctx->ctrBlock.messageToSendClientHasBeenInitialized = true;
    }
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [END]

    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [START]  (INTERNAL)
    std::vector<uint32_t> kcoordinatesForUpperTriangPart;

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        ctx->messageToSendHessiansItems.reserveMemory(ctx->kForCompressor * sizeof(MatrixNMD_d::TElementType));
        ctx->messageToSendGradients.reserveMemory(d * sizeof(VectorND_d::TElementType));
        ctx->messageToSendHessiansIndicies.reserveMemory(ctx->kForCompressor * sizeof(uint32_t));
    }
    // PREPARE BUFFER FOR BETTER RUNTIME BEHAVIOR [END]  (INTERNAL)

    for (size_t r = 0; r < rounds; ++r)
    {
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
                // No-Op
                dopt::DefaultThread::yeildCurrentThInHotLoop();
            }
        }

        // Start round

        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];

            if (!clientHasBeenSelected(ctx->workerIndex, serverCtx))
            {
                ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
                continue;
            }

            size_t kForCompressor = ctx->kForCompressor;

            double preScaleBeforeSend2Master = ctx->preScaleBeforeSend2Master;

            double alpha = ctx->alpha;

            double totalPossibleItems2Send_Inverse = 1.0 / double( ((d + 1) * d) / 2 );
            double delta = double(kForCompressor) * totalPossibleItems2Send_Inverse;

            const VectorND_d& xi = *(serverCtx->currenIterate);
            VectorND_d margin = ctx->optProblem->evaluateClassificationMargin(xi);
            VectorND_d margin_sigmoid = ctx->optProblem->evaluateClassificationMarginSigmoid(margin);

            // Step-1-b: Compute Hessian and compute Hessian difference.
#if DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian< /*bool kSymmetriseHessian*/ false >(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = MatrixNMD_d::computeDifferenceWithUpperTriangularPart(hessian_at_xi, ctx->learningHessian);
#else
            MatrixNMD_d hessian_at_xi = ctx->optProblem->evaluateHessian(xi, margin, margin_sigmoid);
            MatrixNMD_d difference = (hessian_at_xi - ctx->learningHessian);
#endif

            // Step-1-c: Generate Compression pattern
            std::vector<uint32_t> kcoordinates;

            if (kcoordinatesForUpperTriangPart.empty())
                kcoordinatesForUpperTriangPart = dopt::indiciesForUpperTriangularPart(difference);

            kcoordinates = dopt::getTopLEKFromUpperDiagonalPart<true>(difference, kForCompressor, kcoordinatesForUpperTriangPart, delta);
            size_t kcoordinatesSize = kcoordinates.size();

            // Step-2-a: Compress Hessian difference
            ctx->messageToSendHessiansIndicies.rewindToStart();
            ctx->messageToSendHessiansIndicies.putUint32(kcoordinatesSize);
            if (kcoordinatesSize > 0)
            {
                ctx->messageToSendHessiansIndicies.putPODs(kcoordinates.data(), kcoordinatesSize);
            }
            
            ctx->messageToSendHessiansItems.rewindToStart();
            assert(ctx->kForCompressor >= kcoordinates.size());

            // put sparsifed hessian items indicies into the dedicated buffer
            for (size_t k = 0; k < kcoordinatesSize; ++k)
            {
                size_t index = kcoordinates[k];
                const auto compressedItem = difference.matrixByCols.get(index);
                ctx->messageToSendHessiansItems.putDouble(compressedItem * preScaleBeforeSend2Master);
            }
            ctx->ctrBlock.messageToSendHessiansIsReady = true;

            // Learn hessian locally. In-place update.
            for (size_t k = 0; k < kcoordinatesSize; ++k)
            {
                size_t indexFirst = kcoordinates[k];

#if DOPT_USE_HESSIAN_SYMMETRY_IN_FEDNL
                const auto compressedItem = difference.matrixByCols.get(indexFirst);
                ctx->learningHessian.matrixByCols[indexFirst] += alpha * compressedItem;
#else
                size_t iRow = 0;
                size_t iCol = 0;
                difference.getPositionFromFlatternIndex(iRow, iCol, indexFirst);

                const auto compressedItem = difference.matrixByCols.get(indexFirst);

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

            // Put Lk
            if (ctx->send_Lk_from_worker)
            {
                double newLk = (ctx->learningHessian - hessian_at_xi).frobeniusNormForSymmetricMatrixFromUpPart();
                ctx->messageToSendLk.rewindAndPutDouble(newLk - ctx->localLk);
                ctx->localLk = newLk;
                ctx->ctrBlock.messageToSendLkIsReady = true;
            }

            // Put gradient like direction
            VectorND_d dir = (xi * ctx->localLk) + MatrixNMD_d::matrixVectorMultiply(ctx->learningHessian, xi) - ctx->optProblem->evaluateGradient(xi, margin, margin_sigmoid);
            VectorND_d g_ = dir - ctx->localGradient;
            ctx->messageToSendGradients.rewindAndPutPODs(g_.dataConst(), d);
            ctx->localGradient = dir;
            ctx->ctrBlock.messageToSendGradientsIsReady = true;

            // Put fi
            if (ctx->send_fi_from_worker)
            {
                double fi = ctx->optProblem->evaluateFunction(xi, margin, margin_sigmoid);
                ctx->messageToSendFi.rewindAndPutDouble(fi);

                ctx->ctrBlock.messageToSendFiIsReady = true;
            }

            ctx->ctrBlock.messageToSendRoundWorkHasBeenFinished = true;
        }
    }

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForFedNL* ctx = (WorkerContextForFedNL*)wctx[ctxIndex];
        ctx->localGradient = VectorND_d();
        ctx->learningHessian = MatrixNMD_d();
    }

    return 0;
}
