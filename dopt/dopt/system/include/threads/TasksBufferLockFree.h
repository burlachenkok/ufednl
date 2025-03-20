/** @file
* Thread with task queue
*/

#pragma once

#include "dopt/system/include/threads/Thread.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include <vector>
#include <atomic>
#include <set>

#include <stddef.h>

namespace dopt
{

    /* Task stack.
    * @warning Please do not write to TasksStackMt::head variable directly.
    */
    template<class Task>
    class TasksBufferLockFree
    {
    private:
        Task* tasks;                      ///< head at the begin of the array, tail from the end
        size_t capacity;                  ///< capacity
        size_t capacity_minus_one;        ///< capacity minus 1
        
        unsigned char _padding_a[64 - sizeof(Task*) - sizeof(size_t) - sizeof(size_t)];

#if 0
        //std::atomic<size_t> head;        ///< index [0, capacity-1] in which next pushTask will occur
        //std::atomic<size_t> elements;    ///< number of elements
#else
        volatile size_t head;              ///< index [0, capacity-1] in which next pushTask will occur
        volatile size_t elements;          ///< number of elements
#endif
        std::atomic<bool> locker;          ///< locker for synchronization
        unsigned char _padding_b[64 - sizeof(size_t) - sizeof(size_t) - sizeof(std::atomic<bool>)];

    private:

        forceinline_ext size_t incIndex(size_t value) 
        {
            return dopt::add_two_numbers_modN(value, size_t(1), capacity);
        }
        
        forceinline_ext size_t decIndex(size_t value)
        {
            if (value == 0)
            {
                return capacity_minus_one;
            }
            else
            {
                return value - 1;
            }
        }

    public:

        /** Ctor
        */
        TasksBufferLockFree(size_t theCapacity)
        : tasks(nullptr)
        , capacity(theCapacity)
        , capacity_minus_one(theCapacity - 1)
        , head(0)
        , elements(0)
        , locker(false)
        , _padding_a {}
        , _padding_b {}
        {
            if (capacity == 0)
            {
                tasks = nullptr;
            }
            else
            {
                tasks = new Task[capacity];
            }
        }

        ~TasksBufferLockFree()
        {
            while (!isEmpty())
            {
                dopt::DefaultThread::yeildCurrentThInHotLoop();
                continue;
            }
            
            delete[] tasks;
        }


        TasksBufferLockFree(const TasksBufferLockFree&) = delete;
        TasksBufferLockFree& operator = (const TasksBufferLockFree&) = delete;

        TasksBufferLockFree(TasksBufferLockFree&&) = delete;
        TasksBufferLockFree& operator = (const TasksBufferLockFree&&) = delete;

        
        /** Add Task
        */
        bool pushTask(const Task& task, bool wait = true)
        {
            if (wait)
            {
                for (;;)
                {
                    if (isFull())
                    {
                        dopt::DefaultThread::yeildCurrentTh();
                        continue;
                    }

                    dopt::lockWithAtomic(&locker);

                    if (isFull())
                    {
                        dopt::unlockWithAtomic(&locker);
                        // dopt::DefaultThread::yeildCurrentTh();
                        continue;
                    }
                    else
                    {
                        tasks[head] = task;
                        head = decIndex(head);
                        elements += 1;
                        
                        // dopt::internal::memoryFence();
                        dopt::unlockWithAtomic(&locker);
                        break;
                    }
                }

                return true;
            }
            else
            {
                if (isFull())
                    return false;
                
                dopt::lockWithAtomic(&locker);

                if (isFull())
                {
                    dopt::unlockWithAtomic(&locker);
                    return false;
                }
                else
                {
                    tasks[head] = task;
                    head = decIndex(head);
                    elements += 1;

                    //dopt::internal::memoryFence();
                    dopt::unlockWithAtomic(&locker);
                    return true;
                }
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
                    if (isEmpty())
                    {
                        dopt::DefaultThread::yeildCurrentTh();
                        continue;
                    }

                    dopt::lockWithAtomic(&locker);

                    if (isEmpty())
                    {
                        dopt::unlockWithAtomic(&locker);
                        //dopt::DefaultThread::yeildCurrentTh();
                        continue;
                    }
                    else
                    {
                        head = incIndex(head);
                        result = std::move(tasks[head]);
                        elements -= 1;

                        //dopt::internal::memoryFence();
                        dopt::unlockWithAtomic(&locker);
                        break;
                    }
                }

                return true;
            }
            else
            {
                if (isEmpty())
                    return false;

                dopt::lockWithAtomic(&locker);

                if (isEmpty())
                {
                    dopt::unlockWithAtomic(&locker);
                    return false;
                }
                else
                {
                    head = incIndex(head);
                    result = std::move(tasks[head]);
                    elements -= 1;

                    //dopt::internal::memoryFence();
                    dopt::unlockWithAtomic(&locker);
                    return true;
                }
            }
        }

        /** Is buffer empty
        */
        bool isEmpty()
        {
            return elements == 0;
        }

        /** Is buffer full
        */
        bool isFull() const 
        {
            return elements == capacity;
        }

        /** Clean up the stack completely
        */
        void clear()
        {
            elements = 0;            
        }
    };
}
