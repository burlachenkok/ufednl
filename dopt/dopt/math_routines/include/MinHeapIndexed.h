/** @file
* Cross-platform implementation of min-heap with indexing feature.
* D-way min-heap, Johnson, 1975. Tree datastructure, keys are implicitly stored in associated container in specific order.
* Complete binary tree layout:
* - all nodes stored by levels, starting from the root,
* - left to right
* - root contains min element (item from beginning of the array).
* - the "TContainer& data" should be considered be immutable outside scope of MaxHep wrapper. It should provive: size_t size(); operator [](size_t index)
*/

#pragma once

#include "dopt/copylocal/include/Copier.h"

#include <vector>

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    /** Advanced version of minheap with possibilities:
    * - extract min item from minheap
    * - perform work with items by it's original indicies
    * - current implementation ytou need to specify number of items in it in advanced
    */
    template<class T,
             class Index,
             class TContainter,
             class IndexContainer,
             size_t DBranchingFactor,
             class Comparator>
    class MinHeapIndexed
    {
    private:
        TContainter    keys;               ///< Data in some order. 'keyindex' is used to index into this array
        IndexContainer heapIndex2KeyIndex; ///< heapIndex2KeyIndex[i] is the index of the "key" in data array "keys", which is correspond to heap index "i"
        IndexContainer keyIndex2HeapIndex; ///< keyIndex2HeapIndex[i] is the heap position of the "key" with index "i" in keys array "keys"
        size_t         numElementsInHeap;  ///< Current number of setuped elements
        Comparator     IsGreater;          ///< IsGreater(a,b) should return true if a greater then b. Assume that "a" and "b" are totally ordered.

    public:
        static const size_t BranchingFactor = DBranchingFactor;     ///< Branching factor for D-way minheap

        std::vector<T> getSetOfItems() const
        {
            size_t keySize = keys.size();

            if (numElementsInHeap == keySize)
            {
                return keys;
            }
            else
            {
                std::vector<T> resultSet;
                for (size_t i = 0; i < keySize; ++i)
                {
                    if (keyIndex2HeapIndex[i] != Index(-1))
                        resultSet.push_back(keys[i]);
                }
                return resultSet;
            }
        }

        /** Ctor
        * @param maxElements maximum possible number of elements.
        */
        explicit MinHeapIndexed(size_t maxElements, Comparator isGreater)
        : keys(maxElements, T())
        , heapIndex2KeyIndex(maxElements, Index())
        , keyIndex2HeapIndex(maxElements, Index())
        , numElementsInHeap(0)
        , IsGreater(isGreater)
        {
            // If keyIndex2HeapIndex[index] == -1 iff item has not been set.
            for (size_t i = 0; i < maxElements; ++i)
                keyIndex2HeapIndex[i] = Index(-1);
        }

        /** Check that container is empty
        * @return false if there is something inside for this heap
        */
        bool isEmpty() const
        {
            return numElementsInHeap == 0;
        }

        /** Is any elements stored in index i
        * @param keyIndex plain/original index in which you're interesting in
        * @return true if there is something inside
        */
        bool hasItem(Index keyIndex)
        {
            if (keyIndex >= keys.size())
                return false;
            else if (keyIndex2HeapIndex[keyIndex] == Index(-1))
                return false;
            else
                return true;
        }

        /** Number of all elements that container store
        * @return number elements inside. I.e. in fact saved elements
        */
        size_t size() const {
            return numElementsInHeap;
        }

        /** Maximum number of all elements that container can store inside
        * @return number elements inside
        */
        size_t capacity() const {
            return keys.size();
        }

        /** Insert item at position index
        * @param keyIndex original position of the element
        * @pram item value of element
        * @return false if element have already been inserted, true - in other case
        */
        bool insert(Index keyIndex, const T& item)
        {
            if (keyIndex >= keys.size())
                return false;

            if (hasItem(keyIndex))
                return false;

            keys[keyIndex] = item;
            keyIndex2HeapIndex[keyIndex] = numElementsInHeap; // assign last heap index

            heapIndex2KeyIndex[numElementsInHeap] = keyIndex;

            size_t i = numElementsInHeap; // keyIndex2HeapIndex[index];
            numElementsInHeap += 1;       // increase number of elements in the heap

            while(true)
            {
                if (i == 0) // while we have some parent
                    break;

                const auto& parentValue  = keys[heapIndex2KeyIndex[parent(i)]];
                const auto& currentValue = keys[heapIndex2KeyIndex[i]];

                if (IsGreater(parentValue, currentValue))
                {
                    // Parent has bigger value swap current "item" with parent via changing indicies
                    Index j = parent(i);

                    dopt::CopyHelpers::swapDifferentObjects(&heapIndex2KeyIndex[j], &heapIndex2KeyIndex[i]);

                    // restore key index
                    keyIndex2HeapIndex[heapIndex2KeyIndex[j]] = j;
                    keyIndex2HeapIndex[heapIndex2KeyIndex[i]] = i;

                    // setup i to it's parent
                    i = j;
                }
                else
                {
                    break;
                }
            }

            return true;
        }

        /** Key index of element with minimum value
        * @return index of element
        */
        Index minIndex()
        {
            if (numElementsInHeap == 0)
            {
                assert(!"TRY TO RECEIVE MIN ELEMENT INDEX IN HEAP, BUT THE HEAP IS EMPTY!\n");
            }
            return heapIndex2KeyIndex[0];
        }

        /** Min element in the heap
        * @return reference to minimum element
        */
        const T& minElement()
        {
            if (numElementsInHeap == 0)
            {
                assert(!"TRY TO RECEIVE MIN ITEM FROM HEAP, BUT THE HEAP IS EMPTY!\n");
            }
            return keys[heapIndex2KeyIndex[0]];
        }

        /** Get element by index
        * @param keyIndex plain index key index of element
        * @return const reference to element
        */
        const T& get(Index keyIndex)
        {
            return keys[keyIndex];
        }

        /** Change value in index. Support both increase/decrease/first setup operations.
        * @param index key index of element
        * @param setupValue new value
        */
        void change(Index index, const T& setupValue)
        {
            if (!hasItem(index)) 
            {
                insert(index, setupValue);
            }

            if (IsGreater(keys[index], setupValue))
            {
                // Decrease key. Probably need to bubble up.
                keys[index] = setupValue;
                Index i = keyIndex2HeapIndex[index];

                while(true)
                {
                    if (i == 0)
                        break;

                    const auto& parentKey = keys[heapIndex2KeyIndex[parent(i)]];
                    const auto& curKey = keys[heapIndex2KeyIndex[i]];

                    if (IsGreater(parentKey, curKey))
                    {
                        Index j = parent(i);

                        dopt::CopyHelpers::swapDifferentObjects(&heapIndex2KeyIndex[j], &heapIndex2KeyIndex[i]);

                        // restore keyIndex2HeapIndex
                        keyIndex2HeapIndex[heapIndex2KeyIndex[j]] = j;
                        keyIndex2HeapIndex[heapIndex2KeyIndex[i]] = i;

                        // setup i to it's parent
                        i = j;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            else if (IsGreater(setupValue, keys[index]))
            {
                // Increase key. Probably need to bubble down.
                keys[index] = setupValue;
                minHeapify(keyIndex2HeapIndex[index]);
            }
            else
            {
                // Value has not been changed. Assume IsGreater is total order.
            }
        }

        void increaseMinimumElement(Index index, const T& setupValue)
        {
            // Increase key. Probably need to bubble down.
            keys[index] = setupValue;
            minHeapify(0 /* heap index of root */);
        }

        /** Extract minimum element from min heap.
        * @return value of minimum element
        * @remark in case of extracting element from empty heap behavior is undefined.
        */
        T extractMinimum()
        {
            T res = keys[heapIndex2KeyIndex[0]];

            // Swap min and last-element in the heap order. Swapping can lead to potential increase of the key
            Index j = 0;
            Index i = numElementsInHeap - 1;

            dopt::CopyHelpers::swap(&heapIndex2KeyIndex[j], &heapIndex2KeyIndex[i]);
            keyIndex2HeapIndex[heapIndex2KeyIndex[j]] = j;
            keyIndex2HeapIndex[heapIndex2KeyIndex[i]] = i;

            keyIndex2HeapIndex[heapIndex2KeyIndex[numElementsInHeap - 1]] = Index(-1);
            numElementsInHeap -= 1;

            // Process possible increase key. Probably need to bubble down.
            minHeapify(keyIndex2HeapIndex[heapIndex2KeyIndex[0]]);

            return res;
        }


        /** Peek the minimum element in the min-heap pyramid, but do not eject it
        * @param data input container with random-access
        * @return reference to maximum element, which lie in top of the heap
        */
        const T& peekMinimum()
        {
            const auto& minimumKeyIndex = heapIndex2KeyIndex[0];
            return keys[minimumKeyIndex];
        }

    protected:
        /* Get the parent of the element with the number i.
        * @param heapIndex zero-based index of the element for which the request is made
        * @return zero-based index of parent inside container
        * @remark Konstantin B. has derivation for this formula
        */
        static Index parent(Index heapIndex) {
            return (heapIndex - 1) / BranchingFactor;
        }

        /* Get the child of the element "node" with the number "i"
        * @param i the index of the element for which the request is made with zero indexing.
        * @return zero-based index of parent inside container
        * @remark For optimization purposes in fact children(node, i+1) = children(node, i) + 1
        */
        static Index children(Index nodeHeapIndex, Index childNumber)
        {
            assert(childNumber < BranchingFactor);
            /** For binary heap
            * left(node)  ~ children(node, 0) AND
            * right(node) ~ children(node, 1)
            */

            Index childHeapIndex = BranchingFactor * nodeHeapIndex + 1 + childNumber;
            return childHeapIndex;
        }

        /** Heapify/Bubble Down.
        * Push item with index i to down and stop when no need in extra updates. I.e. we reconstruct heap invariants.
        * @param  heapIndex parent node from which we start process.
        * @remark All "D" child of "heapIndex" if they are presented are normal and valid heaps.
        */
        void minHeapify(Index heapIndex)
        {
            while(true)
            {
                Index lowest = heapIndex;

                for (Index j = 0; j < BranchingFactor; ++j)
                {
                    Index child = children(heapIndex, j);
                    if (child >= numElementsInHeap)
                        break;

                    const auto& childKey = keys[heapIndex2KeyIndex[child]];
                    const auto& lowestKey = keys[heapIndex2KeyIndex[lowest]];
                    if (IsGreater(lowestKey, childKey))
                    {
                        lowest = child;
                    }
                }

                if (lowest != heapIndex)
                {
                    dopt::CopyHelpers::swapDifferentObjects(&heapIndex2KeyIndex[heapIndex], &heapIndex2KeyIndex[lowest]);
                    // restore keyIndex2HeapIndex
                    keyIndex2HeapIndex[heapIndex2KeyIndex[heapIndex]] = heapIndex;
                    keyIndex2HeapIndex[heapIndex2KeyIndex[lowest]] = lowest;

                    // recurse
                    heapIndex = lowest;
                }
                else
                {
                    break;
                }
            }
        }
    };
}
