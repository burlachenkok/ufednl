/** @file
* Thread with task queue
*/

#pragma once

#include "dopt/system/include/threads/Thread.h"

#include <vector>
#include <atomic>
#include <set>


namespace dopt
{

    /* Task stack.
    * @warning Please do not write to TasksStackMt::head variable directly.
    */
    template<class Task>
    class TasksBufferMt
    {
    private:
        std::vector<Task> tasks;
        std::atomic<bool> tasks_lock = false;

    public:
        
        /** Ctor
        */
        TasksBufferMt(size_t capacity) {
            tasks.reserve(capacity);
        }

        ~TasksBufferMt()
        {
            while (!isEmpty())
            {
                dopt::DefaultThread::yeildCurrentThInHotLoop();
                continue;
            }
        }

        TasksBufferMt(const TasksBufferMt&) = delete;
        TasksBufferMt& operator = (const TasksBufferMt&) = delete;

        TasksBufferMt(TasksBufferMt&&) = delete;
        TasksBufferMt& operator = (const TasksBufferMt&&) = delete;

       
        /** Add Task
        */
        bool pushTask(const Task& task, bool wait = true)
        {
            if (wait)
            {
                for (;;)
                {
                    dopt::lockWithAtomic(&tasks_lock);
                    size_t size = tasks.size();

                    if (size >= tasks.capacity())
                    {
                        dopt::unlockWithAtomic(&tasks_lock);
                        dopt::DefaultThread::yeildCurrentTh();                        
                        continue;
                    }
                    else
                    {
                        tasks.push_back(task);
                        dopt::unlockWithAtomic(&tasks_lock);
                        break;
                    }
                }
                
                return true;
            }
            else
            {
                bool res = false;
                
                {
                    dopt::lockWithAtomic(&tasks_lock);
                    size_t size = tasks.size();

                    if (size >= tasks.capacity())
                    {
                        dopt::unlockWithAtomic(&tasks_lock);
                        res = false;
                    }
                    else
                    {
                        tasks.push_back(task);
                        dopt::unlockWithAtomic(&tasks_lock);
                        res = true;
                    }
                }

                return res;
            }
        }

        /** Pop Task
        */
        bool popTask(Task& result, bool wait = true)
        {
            if (wait)
            {
                for (;;)
                {
                    dopt::lockWithAtomic(&tasks_lock);
                    size_t size = tasks.size();

                    if (size == 0)
                    {
                        dopt::unlockWithAtomic(&tasks_lock);
                        dopt::DefaultThread::yeildCurrentThInHotLoop();
                        continue;
                    }
                    else
                    {
                        result = std::move(tasks.back());
                        tasks.pop_back();
                        dopt::unlockWithAtomic(&tasks_lock);
                        break;
                    }
                }

                return true;
            }
            else
            {
                bool res = false;

                {
                    dopt::lockWithAtomic(&tasks_lock);
                    size_t size = tasks.size();

                    if (size == 0)
                    {
                        dopt::unlockWithAtomic(&tasks_lock);
                        res = false;
                    }
                    else
                    {
                        result = std::move(tasks.back());
                        tasks.pop_back();
                        dopt::unlockWithAtomic(&tasks_lock);
                        res = true;
                    }
                }

                return res;
            }
        }

        /** Is stack empty
        */
        bool isEmpty()
        {
            dopt::lockWithAtomic(&tasks_lock);
            bool res = tasks.empty();
            dopt::unlockWithAtomic(&tasks_lock);
            return res;
        }

        /** Clean up the stack completely
        */
        void clear()
        {
            dopt::lockWithAtomic(&tasks_lock);
            tasks.clear();
            dopt::unlockWithAtomic(&tasks_lock);            
        }
    };
}
