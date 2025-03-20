/** @file
* Implementation of D-way min-heap based on random access container
* Complete binary tree layout:
* - all nodes stored by levels, starting from the root,
* - left to right
* - root contains min element (item from beginning of the array).
* - the "TContainer& data" should be considered be immutable outside scope of MaxHep wrapepr. It should probive: size_t size(); operator [](size_t index)
*/

#pragma once

#include "dopt/copylocal/include/Copier.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <vector>
#include <type_traits>

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    /** D-way min-heap, Johnson, 1975. It's tree datastructures, but keys are implicitly stored in associated container in specific order.
    * - Keys in nodes
    * - Parent key no bigger than children's keys
    * @tparam D branching factor
    */
    template<size_t D = 2>
    class MinHeap
    {
    public:
        static const size_t BranchingFactor = D; ///< Branching factor

        /** Change the value in the maxheap pyramid by index
        * @param data input container with random-access
        * @param i index of item which should be changed
        * @param setupValue new value of item
        * @return true if item have been changed successfully
        * @remark complexity is ~log_d(N)
        */
        template<class TData, class TContainer, class Converter>
        static bool changeInMinHeap(TContainer& data, size_t i, const TData setupValue, const Converter convertToComparable)
        {
            if (data.size() == 0)
            {
                assert(!"TRY TO RECEIVE MIN IN HEAP, BUT THE HEAP IS EMPTY!\n");
                return false;
            }

            if (convertToComparable(data[i]) >= convertToComparable(setupValue))
            { 
                // Decrease key
                data[i] = setupValue;
                while ( i != 0 && convertToComparable(data[parent(i)]) > convertToComparable(data[i]) )
                {
                    dopt::CopyHelpers::swapDifferentObjects(&data[parent(i)], &data[i]);
                    i = parent(i);
                }
            }
            else
            {
                // increase key
                data[i] = setupValue;
                minHeapify<TContainer, Converter>(data, i, data.size(), convertToComparable);
            }

            return true;
        }

        /** Change the value in the maxheap pyramid by index
        * @param data input container with random-access
        * @param i index of item which should be changed
        * @param setupValue new value of item
        * @return true if item have been changed successfully
        * @remark complexity is ~log_d(N)
        */
        template<class TData, class TContainer, class Converter>
        static void increaseMinimumInMinHeap(TContainer& data, const TData setupValue, const Converter convertToComparable)
        {
            assert( convertToComparable(setupValue) >= convertToComparable(data[0]) );

            data[0] = setupValue;
            minHeapify<TContainer, Converter>(data, 0, data.size(), convertToComparable);
        }

        /** Add an element to the min-heap.
        * @param data input container with random-access
        * @param value item to append
        * @remark complexity is ~log_d(N)
        */
        template<class TData, class TContainer, class Converter>
        static void insertInMinHeap(TContainer& data, TData value, const Converter convertToComparable)
        {
            data.push_back(value);
            size_t i = data.size() - 1;

            while ( i != 0 && convertToComparable(data[parent(i)]) > convertToComparable(data[i]) )
            {
                dopt::CopyHelpers::swapDifferentObjects(&data[parent(i)], &data[i]);
                i = parent(i);
            }
        }

        /** Peek the minimum element in the min-heap pyramid, but do not eject it
        * @param data input container with random-access
        * @return reference to maximum element, which lie in top of the heap
        */
        template<class TData, class TContainer>
        static const auto& peekFromMinHeap(TContainer& data)
        {
            if (data.size() == 0)
            {
                assert(!"TRY TO RECEIVE MIN IN HEAP, BUT THE HEAP IS EMPTY!\n");
            }

            return data[0];
        }

        /** Copy the minimum element from the min-heap pyramid, and eject it
        * @param data input container with random-access
        * @return value of maximum element
        * @remark complexity is ~d*log_d(N)
        */
        template<class TData, class TContainer, class Converter>
        static TData extracFromMinHeap(TContainer& data, const Converter convertToComparable)
        {
            if (data.size() == 0)
            {
                assert(!"TRY TO RECEIVE MIN IN HEAP, BUT THE HEAP IS EMPTY!\n");
            }

            TData res = data[0];
            size_t heap_size = data.size();

            dopt::CopyHelpers::swap(&data[0], &data[heap_size-1]);
            
            minHeapify<TContainer, Converter>(data, 0, heap_size - 1, convertToComparable);
            data.resize(heap_size - 1);
            return res;
        }

        /** Construct from a random array min-heap pyramid with the bottom-up strategy.
        * @param data input container with random-access which used to create heap inplace.
        * @remark Proof of the fact that it takes linear time - pp. 186-187 Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein - INTRODUCTION TO ALGORITHMS SECOND EDITION, 2005
        */
        template<class TContainer, class Converter>
        static void buildMinHeap(TContainer& data, const Converter convertToComparable)
        {
            assert(D >= 2);

            size_t sz = data.size();

            // I select to get parent of last item. 
            // This start parent "i" point to childs that have no childs. So they are by itself are leafs and correct min-heaps
            
            // parent(i) := (i - 1) / D
            // parent(sz-1) := ( (sz-1) - 1) / D
            // parent(sz-1+D) := ( (D+sz-1) - 1) / D = (sz + D - 2) / D
#if 0
            if constexpr (hintDataIsNotEmpty)
            {
                assert(sz >= 1);
            }
            else
            {
                if (sz <= 1)
                    return;
            }
            size_t i = (sz - 2) / D;
#endif

            
            // For purpose of initialization it can be a case that there are items in (i, sz-1].
            // However they can be considered as leaves
            // Previously: size_t ii = (sz + D - 2) / D;
            size_t i = (sz + D - 2) / D;

            while(true)
            {
                minHeapify<TContainer, Converter>(data, i, sz, convertToComparable);
                if (i == 0)
                {
                    break;
                }
                else
                {
                    --i;
                }
            }
        }

        /** Check the fact that data is the correct min-heap
        * @param data input container
        * @return true if data presented corrected min-heap
        */
        template<class TData, class TContainer, class  Converter>
        static bool checkMinHeapPropperty(TContainer& data, const Converter convertToComparable)
        {
            for (size_t i = 1; i < data.size(); ++i)
            {
                if ( convertToComparable(data[parent(i)]) > convertToComparable(data[i]) )
                {
                    return false;
                }
            }

            return true;
        }


        /* Get the parent of the element with the number i.
        * @param i zero-based index of the element for which the request is made
        * @return zero-based index of parent inside container
        */
        constexpr const_func_ext static size_t parent(size_t i)
        {
            assert(i > 0);
            //return (i+1)/2 - 1;
            // http://blog.mischel.com/2013/10/05/the-d-ary-heap/
            // Also own derivations in d_way_heap.docx/pdf
            return (i - 1) / D;
        }

        constexpr const_func_ext static size_t firstChildren(size_t node)
        {
            size_t childIndex = D * node + 1 + 0;
            return childIndex;
        }

        constexpr const_func_ext static size_t children(size_t node, size_t childNumber)
        {
            assert(childNumber < D);
            // http://blog.mischel.com/2013/10/05/the-d-ary-heap/
            // Also own derivations in d_way_heap.docx/pdf

            /** For binary heap
            * left(node)  ~ children(node, 0) AND
            * right(node) ~ children(node, 1)
            */
            size_t childIndex =  D * node + 1 + childNumber;            
            return childIndex;
        }

    protected:

        /**
        * Adjusts the specified element in the container to maintain the min-heap property.
        *
        * @param data Container containing elements of the heap.
        * @param i Index of the element to start heapifying from.
        * @param heapSize Size of the heap.
        * @param convertToComparable Function to convert elements to comparable values.
        */
        template<class TContainer, class  Converter>
        static void minHeapify(TContainer& data, size_t i, size_t heapSize, const Converter convertToComparable)
        {
            // Initially lowestValue value of i-th node
            auto lowestValue = convertToComparable(data[i]);

            for (;;)
            {
                size_t lowest = i;
                size_t firstChild = firstChildren(i);
                size_t lastChildBound  = dopt::minimum(firstChild + D, heapSize);

                for (size_t child = firstChild; child < lastChildBound; ++child)
                {                    
                    auto childValue = convertToComparable(data[child]);

                    if (lowestValue > childValue)
                    {
                        lowest = child;
                        lowestValue = childValue;
                    }
                }

                if (lowest != i)
                {
                    dopt::CopyHelpers::swapDifferentObjects(&data[i], &data[lowest]);
                    i = lowest;
                    // because i := lowest 
                    // => value of i-th node is equal to lowestValue
                }
                else
                {
                    break;
                }
            }
        }
    };
}
