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

    for (size_t r = 0; /*r < rounds*/; ++r)
    {
        // Wait for permission by worker to start
        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForGD* ctx = (WorkerContextForGD*)wctx[ctxIndex];


            for (;;)
            {
                serverCtx->receiveUpdates</*block*/true>(ctx->control);

                {
                    if (serverCtx->get_hessian)
                    {
                        serverCtx->get_hessian = false;
                        assert(!"INTERNAL ERROR");
                        
                        // Request for full Hessian
                        //ctx->control->sendByte(ServerContext::ControlSignals::sig_response_full_hessian_in_client);
                        //ctx->control->sendMatrixItems(ctx->learningHessian);
                    }

                    if (r == serverCtx->roundToStart || serverCtx->terminate)
                    {
                        // no need to reset roundToStart because it's incremented in each round
                        // no need to reset terminate because terminate means the halt of the client
                        break;
                    }
                }
            }
        }
        
        // Terminate if it has been requested
        if (serverCtx->terminate)
            break;

        for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
        {
            WorkerContextForGD* ctx = (WorkerContextForGD*)wctx[ctxIndex];
            const VectorND_d& xi = *(serverCtx->currenIterate);

            ctx->localGradient = ctx->optProblem->evaluateGradient(xi);

            dopt::MutableData buffer;
            ctx->prepareUpdate(buffer, WorkerContextForGD::sig_messageToSendGradient);
            ctx->control->sendData(buffer.getPtr(), buffer.getFilledSize());

            if (ctx->send_fi_from_worker)
            {
                ctx->fiValue = ctx->optProblem->evaluateFunction(xi);

                dopt::MutableData buffer;
                ctx->prepareUpdate(buffer, WorkerContextForGD::sig_messageToSendFi);
                ctx->control->sendData(buffer.getPtr(), buffer.getFilledSize());
            }
        }
    }

    for (size_t ctxIndex = 0; ctxIndex < ctxToProcess; ++ctxIndex)
    {
        WorkerContextForGD* ctx = (WorkerContextForGD*)wctx[ctxIndex];
        ctx->localGradient = VectorND_d();
    }

    return 0;
}
