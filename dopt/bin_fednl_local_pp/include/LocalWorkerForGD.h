#pragma once

#include "dopt/system/include/threads/Thread.h"

#include "UsedTypes.h"
#include <vector>

inline int32_t workerThreadTrainLoopForGD(void* arg1, void* arg2)
{
    const std::vector<WorkerContext*>& wctx = *(std::vector<WorkerContext*>*)(arg1);
    ServerContext* serverCtx = (ServerContext*)arg2;

    size_t ctxToProcess = wctx.size();
    size_t rounds = serverCtx->rounds;

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
            WorkerContextForGD* ctx = (WorkerContextForGD*)wctx[ctxIndex];
            const VectorND_d& xi = *(serverCtx->currenIterate);

            ctx->localGradient = ctx->optProblem->evaluateGradient(xi);

            ctx->ctrBlock.resultIsReady = true;
        }
    }

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForGD* ctx = (WorkerContextForGD*)wctx[ctxIndex];
        ctx->localGradient = VectorND_d();
    }

    return 0;
}
