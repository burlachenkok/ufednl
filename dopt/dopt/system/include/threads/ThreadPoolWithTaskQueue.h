/** @file
* Thread with task queue
*/

#pragma once

#include "dopt/system/include/threads/Thread.h"

#include "dopt/system/include/threads/TasksBufferMt.h"
#include "dopt/system/include/threads/TasksBufferLockFree.h"

#include <vector>
#include <atomic>

#include <stddef.h>
#include <stdint.h>

namespace dopt
{
    template<class Task>
    class ThreadPoolWithTaskQueue
    {
    private:

        static int32_t workRoutine(void* arg1, void* arg2)
        {
            ThreadPoolWithTaskQueue* thWithTasks = (ThreadPoolWithTaskQueue*)arg1; ///< Thread pool
            size_t thIndex = reinterpret_cast<size_t>(arg2);                       ///< Worker index

            Task currentTask = Task();
            bool currentTaskHasBeenSetuped = false;
            
            for (;;)
            {
                //===============================================================================//
                // PROCESSING SIGNALS
                //===============================================================================//
                if (thWithTasks->terminateExecution)
                {
                    // First: Handle teminate signal. If occured finish ASAP.
                    break;
                }

                if (thWithTasks->suspendProcessing)
                {
                    // Handle request signals for suspend processing for this thread
                    thWithTasks->threadInSuspendState[thIndex] = uint8_t(1);
                    dopt::internal::memoryFence();
                    // Now thread is in the suspend state
                    dopt::DefaultThread::yeildCurrentThInHotLoop();
                    // Back to spinning
                    continue;
                }

                if (thWithTasks->suspendProcessingAfterProcessAllTasks)
                {
                    // Handle request signal to suspend, but after all tasks will be finished only.

                    // Now proessing all avilable tasks
                    for (;;)
                    {
                        bool pickTask = thWithTasks->tasksInWaitList.popTask(currentTask, false /*wait*/);
                        
                        if (!pickTask)
                        {
                            // no tasks => finish
                            break;
                        }
                        else
                        {
                            // process task
                            thWithTasks->processingRoutine(currentTask, thIndex);
                        }
                    }
                    
                    // Response: report that thread is in the suspend state
                    thWithTasks->threadInSuspendState[thIndex] = uint8_t(1);
                    dopt::internal::memoryFence();

                    for (;;)
                    {
                        // Suspend amd wait for allowance for continue
                        if (thWithTasks->suspendProcessingAfterProcessAllTasks == false || thWithTasks->terminateExecution == true)
                        {
                            break;
                        }
                        else
                        {
                            dopt::DefaultThread::yeildCurrentThInHotLoop();
                        }
                    }
                    continue;
                }
                //===============================================================================//
                // PROCESSING SINGLE TASK      
                //===============================================================================//
                if (thWithTasks->tasksInWaitList.popTask(currentTask, false /*wait*/))
                {
                    thWithTasks->processingRoutine(currentTask, thIndex);
                }
                else
                {
                    DefaultThread::yeildCurrentThInHotLoop();
                }
                //===============================================================================//
            }

            return 0;
        }

    public:
        typedef void (*ProcessingRoutine)(const Task& taskToProcess, size_t thIndex);

        /** Ctor.
        * @param theProcessingRoutine custom thread processing routine.
        * @param threadPoolSize size of a thread pool.
        * @param bufferCapacity maximum capacity of incoming buffer to process the tasks
        */
        ThreadPoolWithTaskQueue(ProcessingRoutine theProcessingRoutine, size_t threadPoolSize, size_t bufferCapacity)
        : processingRoutine(theProcessingRoutine)
        , tasksInWaitList(threadPoolSize > 0 ? bufferCapacity : 0)
        , threadInSuspendState(threadPoolSize, 0)
        {
            workingThreads.reserve(threadPoolSize);

            for (size_t i = 0; i < threadPoolSize; ++i)
            {
                static_assert(sizeof(size_t) <= sizeof(void*), "To have low-level casting this invariant should hold");
                workingThreads.push_back(new dopt::DefaultThread(workRoutine, this, reinterpret_cast<void*>(i)));
            }
        }

        /** Dtor. Terminate all workers after compleion current in-progress works.
        */
        ~ThreadPoolWithTaskQueue()
        {
            terminate();
        }

        void terminate()
        {
            // Corner case - no working threads
            if (workingThreads.size() == 0)
                return;

            // Signal everybody to terminate
            signalToTerminate();

            // Wait for completion
            waitForTermination();

            // Actual deletion of working threads
            for (size_t i = 0; i < workingThreads.size(); ++i)
            {
                delete workingThreads[i];
            }

            workingThreads.clear();
            tasksInWaitList.clear();
        }

        /** Add task to the thread pool
        */
        void addTask(const Task& task) 
        {
            tasksInWaitList.pushTask(task);
        }

        /** Cancedl all tasks
        */
        void cancelAllFutureTasks()  
        {
            tasksInWaitList.clear();
        }

        /** Are any tasks that are currently in the wait list to be executed
        * @return true if there are any tasks that are in the wait list
        */
        bool isWaitingTasks() {
            return tasksInWaitList.isEmpty() == false;
        }

        /** Wait for all threads in a thread pool to finish
        */
        void waitForTermination()
        {
            size_t nThreads = workingThreads.size();

            for (size_t i = 0; i < nThreads; ++i)
            {
                workingThreads[i]->join();
            }
        }
        
        /** Get the thread pool size in threads
        * @return the thread pool size in threads
        */
        size_t threadPoolSize() const
        {
            return workingThreads.size();
        }

        /** Signal to terminate every worker thread without executing new jobs
        */
        void signalToTerminate()
        {
            terminateExecution = true;
        }

        /** Signal to suspend processing
        * @remark After exiting from the call there is no garantee that workers are actually in a suspend state. They will be in suspend state during attempt to pickup next work.
        * @sa waitForAllThreadsToSuspend
        */
        void signalSuspendProcessing()
        {
            if (suspendProcessing == false)
            {
                for (size_t i = 0; i < threadInSuspendState.size(); ++i)
                {
                    threadInSuspendState[i] = 0;
                }
            }
            dopt::internal::memoryFence();

            suspendProcessing = true;
        }

        /** Signal to resume processing
        * @remark If working thread are in suspend state they actually
        */
        void signalResumeProcessing() 
        {
            suspendProcessing = false;
        }

        /** Wait for a moment when all workers reaches suspend state. For this waiting make sense.
        * @remark Implemented with spinning
        */
        bool waitForAllThreadsToSuspend()
        {
            if (!suspendProcessing && !suspendProcessingAfterProcessAllTasks)
            {
                assert(!"PLEASE FIRSTLY MAKE SUSPEND PROCESSING REQUEST");
                return false;
            }

            for (;;)
            {
                bool res = true;
                
                for (size_t i = 0; i < threadInSuspendState.size(); ++i)
                {
                    if (threadInSuspendState[i] != 1)
                    {
                        res = false;
                        break;
                    }
                }

                if (res)
                {
                    break;
                }
                else
                {
                    DefaultThread::yeildCurrentThInHotLoop();
                    continue;
                }
            }

            return true;
        }

        /** Wait for processing all tasks included: currently executed and in the wait list
        */
        bool waitForCurrentJobsCompletion()
        {
            if (suspendProcessing)
                return false;
            
            // reset suspend state
            for (size_t i = 0; i < threadInSuspendState.size(); ++i)
                threadInSuspendState[i] = 0;

            dopt::internal::memoryFence();

            // signal
            suspendProcessingAfterProcessAllTasks = true;

            // wait
            waitForAllThreadsToSuspend();

            // allow to resume
            suspendProcessingAfterProcessAllTasks = false;

            return true;
        }

    private:
        ProcessingRoutine processingRoutine;         ///< Main routine        
        std::vector<DefaultThread*> workingThreads;  ///< Execution threads
        TasksBufferLockFree<Task> tasksInWaitList;   ///< Stack of enqueud tasks which are in the wait list
       
        std::atomic<bool> terminateExecution = false;                        ///< Marker to finish the processing thread
        std::atomic<bool> suspendProcessing = true;                          ///< Marker to suspend processing
        std::atomic<bool> suspendProcessingAfterProcessAllTasks = false;     ///< Marker to suspend processing once there is no more tasks available
        
        std::vector<uint8_t> threadInSuspendState;                           ///< Response to suspend processing request
    };
}
