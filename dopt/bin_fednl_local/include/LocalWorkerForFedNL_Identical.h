#pragma once

#include "dopt/system/include/threads/Thread.h"

#include "UsedTypes.h"
#include <vector>

inline int32_t workerThreadTrainLoopForFedNL1_IdenticalCompressor(void* arg1, void* arg2)
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
            
            double preScaleBeforeSend2Master = ctx->preScaleBeforeSend2Master;

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
            MatrixNMD_d differenceSparsified = MatrixNMD_d::getZeroSquareMatrix(d);

            // Reset head in writing buffer
            //ctx->messageToSendHessiansIndicies.rewindToStart();

            ctx->messageToSendHessiansItems.rewindToStart();

            // Put sparsifed hessian into the buffer and construct differenceSparsified
            size_t send_coordinates = 0;

            // Scan through upper triangular part
            for (size_t j = 0; j < d; ++j)
            {
                for (size_t i = 0; i <= j; ++i, ++send_coordinates)
                {
                    const auto& item = difference.get(i, j);
                    ctx->messageToSendHessiansItems.putDouble(item * preScaleBeforeSend2Master);

                    differenceSparsified.set(i, j, item);
                    differenceSparsified.set(j, i, item);
                }
            }
            assert(ctx->kForCompressor == send_coordinates);
            ctx->ctrBlock.messageToSendHessiansIsReady = true;

            ctx->learningHessian += ctx->alpha * differenceSparsified;

            // Put Lk
            if (ctx->send_Lk_from_worker)
            {
                //ctx->messageToSendLk.rewindToStart();
                ctx->messageToSendLk.rewindAndPutDouble(difference.frobeniusNormForSymmetricMatrixFromUpPart());
                ctx->ctrBlock.messageToSendLkIsReady = true;
            }

            // Put fi
            if (ctx->send_fi_from_worker)
            {
                //ctx->messageToSendFi.rewindToStart();
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
