/** @file
* Merge sort
*/

#pragma once

#include "dopt/copylocal/include/Copier.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stdint.h>

#if DOPT_WINDOWS
    #include <malloc.h>
#else
    #include <alloca.h>
#endif

#include <iostream>
#include <array>
#include <type_traits>

namespace dopt
{
    /** Insertion sort.
    *  - sort is in place.
    *  - sort is stable. I.e. sort preserves the relative order of equal elements.
    *  - fast for small arrays, and not at all for large arrays.
    *  - Complexity:
    *    + Linear(~N) time for already sorted array
    *    + Good for partially sorted array with speed ~N
    *    + In average ~1/4n*n
    */
    template<class TElement, class Comparator>
    static void insertionSort(TElement* restrict_ext first, size_t len, Comparator cmpIsLess)
    {
        if (len <= 1)
            return;

        for (size_t j = 1; j < len; ++j)
        {
            TElement key = first[j];

            size_t i = j;

            // key < first[i - 1]
            while (i != 0 && cmpIsLess(key, first[i - 1]))
            {
                first[i] = first[i - 1];
                --i;
            }
            // key >= first[i - 1] => break

            first[i] = key;
        }
    }

    /**
     * Sorts the first `k` elements of the given range using the selection sort algorithm.
     *
     * @param first Pointer to the first element of the range to be sorted.
     * @param len The number of elements in the range.
     * @param k The number of elements at the beginning of the range to sort.
     * @param cmpIsLess Comparator function that returns true if the first argument is less than the second argument.
    */
    template<class TElement, class Comparator>
    static void selectionSortExtended(TElement* restrict_ext first, size_t len, size_t k, Comparator cmpIsLess)
    {
        if (len <= 1)
            return;

        size_t iBound = minimum(k, len-1);

        for (size_t i = 0; i < iBound; ++i)
        {
            size_t jMin = i;
            for (size_t j = i + 1; j < len; ++j)
            {
                if (cmpIsLess(first[j], first[jMin]))
                {
                    jMin = j;
                }
            }

            if (jMin != i)
                dopt::CopyHelpers::swapDifferentObjects(&first[jMin], &first[i]);
        }
    }

    /**
     * Merges two sorted input ranges into a single sorted output range.
     *
     * @param output The iterator where the merged output should be written.
     * @param firstA The beginning iterator of the first sorted input range.
     * @param endA The ending iterator of the first sorted input range.
     * @param firstB The beginning iterator of the second sorted input range.
     * @param endB The ending iterator of the second sorted input range.
     * @param cmpIsLess A comparison function object that returns true if the first argument is less than the second.
    */
    template<class IteratorOut, class IteratorIn, class Cmp>
    static void mergeInternal(IteratorOut output,
                              IteratorIn firstA, IteratorIn endA,
                              IteratorIn firstB, IteratorIn endB,
                              Cmp cmpIsLess)
    {
        for (;;)
        {
            if (firstA != endA && firstB != endB)
            {                
                if (cmpIsLess(*firstB, *firstA))  // In greater case elements from right subarray
                {
                    *output = *firstB;
                    ++output;
                    ++firstB;
                }
                else                              // In equal case or smaller case take element from left subarray
                {
                    *output = *firstA;
                    ++output;
                    ++firstA;
                }
            }
            else if (firstA != endA)
            {
                CopyHelpers::copyWithIterators(output, firstA, endA);
                break;
            }
            else if (firstB != endB)
            {
                CopyHelpers::copyWithIterators(output, firstB, endB);
                break;
            }
            else
            {
                break;
            }
        }
    }

    /**
     * @brief A structure to hold the result of a merge sort operation.
     *
     * This structure is used to encapsulate the output of the merge sort algorithm,
     * providing an easy way to access the sorted array.
     */
    template<class TElement>
    struct MergeSortResult
    {
        /**
         * Constructor for the MergeSortResult struct.
         *
         * @param theOutput A pointer to an array of TElement which holds the sorted output.
         * @return An instance of MergeSortResult initialized with the provided sorted output.
         */
        MergeSortResult(TElement* theOutput)
        : output(theOutput)
        {
        }

        /**
         * @brief Pointer to the array holding the sorted elements.
         */
        TElement* output;
    };

    /**
     * @brief Internal recursive implementation of the merge sort algorithm.
     *
     * This function performs a merge sort on an array of elements using a provided comparator for ordering.
     * It is optimized to handle small arrays with insertion sort and reduces memory overhead by using a draft storage buffer.
     *
     * @tparam TElement The type of elements to be sorted.
     * @tparam Comparator The type of the comparator function used to compare elements.
     * @tparam thresholdForInsertionSort The threshold below which the function uses insertion sort.
     *
     * @param first Pointer to the first element of the array to be sorted.
     * @param len The number of elements in the array.
     * @param draftStorage Pointer to the temporary storage used during the merge process.
     * @param cmpIsLess Comparator function to determine the order of elements. Should return true if the first
     *        argument is less than the second argument.
     *
     * @return A structure containing the sorted array.
     */
    template<class TElement, class Comparator, size_t thresholdForInsertionSort>
    static MergeSortResult<TElement> mergeSortInternal(TElement* first, size_t len, TElement* draftStorage, Comparator cmpIsLess)
    {
        if (len <= 1)
            return MergeSortResult<TElement>(first);

        if (len <= thresholdForInsertionSort)
        {
            insertionSort(first, len, cmpIsLess);
            return MergeSortResult<TElement>(first);
        }

        size_t pivot = len / 2;
        size_t firstHalfLen = pivot;
        size_t secondHalfLen = len - pivot;

        MergeSortResult<TElement> firstHalfSortResult = mergeSortInternal<TElement, Comparator, thresholdForInsertionSort> (first, firstHalfLen, draftStorage, cmpIsLess);
        MergeSortResult<TElement> seconHalfSortResult = mergeSortInternal<TElement, Comparator, thresholdForInsertionSort> (first + pivot, secondHalfLen, draftStorage + pivot, cmpIsLess);
        
        bool outFirst_InFirstBuffer = (firstHalfSortResult.output == first);
        bool outFirst_InSecondBuffer = (firstHalfSortResult.output == draftStorage);

        bool outSecond_InFirstBuffer = (seconHalfSortResult.output == first + pivot);
        bool outSecond_InSecondBuffer = (seconHalfSortResult.output == draftStorage + pivot);

        assert(outFirst_InFirstBuffer ^ outFirst_InSecondBuffer);
        assert(outSecond_InFirstBuffer ^ outSecond_InSecondBuffer);

        if (cmpIsLess(*(seconHalfSortResult.output), 
                      *(firstHalfSortResult.output + firstHalfLen - 1)))
        {
            if (outFirst_InSecondBuffer) // firstHalfSortResult.output == draftStorage
                draftStorage = first;
            
            // outFirst_InFirstBuffer & outSecond_InFirstBuffer => safe to merge
            // outFirst_InSecondBuffer & outSecond_InSecondBuffer => safe to merge
            // outFirst_InFirstBuffer & outSecond_InSecondBuffer => safe to merge because during merge it's ok to have it
            // outFirst_InSecondBuffer & outSecond_InFirstBuffer => safe to merge because during merge it's ok to have it

            mergeInternal(draftStorage, firstHalfSortResult.output, firstHalfSortResult.output + firstHalfLen, 
                                        seconHalfSortResult.output, seconHalfSortResult.output + secondHalfLen, cmpIsLess);
            // Swap draft storage and result storage
            return MergeSortResult<TElement>(draftStorage);
        }
        else
        {
            // Improvement: Corner case when there is nothing to merge
            // 
            // ... < first[pivot - 2] < first[pivot - 1] <= first[pivot] < first[pivot+1] < ....

            if (outFirst_InFirstBuffer && outSecond_InFirstBuffer)
            {
                // If output is contigious lie inside first
                return MergeSortResult<TElement>(first);
            }
            else if (outFirst_InSecondBuffer && outSecond_InSecondBuffer)
            {
                // If output is contigious lie inside draftStorage
                return MergeSortResult<TElement>(draftStorage);
            }
            else
            {
                // Outputs due to logic and various optimization lie in different arrays
                if (outFirst_InSecondBuffer)
                    draftStorage = first;

                mergeInternal(draftStorage, firstHalfSortResult.output, firstHalfSortResult.output + firstHalfLen,
                                            seconHalfSortResult.output, seconHalfSortResult.output + secondHalfLen, cmpIsLess);
                
                // Swap draft storage and result storage
                return MergeSortResult<TElement>(draftStorage);
            }
        }
    }

    /** Modification of merge sort.
    * About Merge Sort:
    *  - sort is not in place.
    *  - sort is stable. I.e. sort preserves the relative order of equal elements.
    *  - beats insertion sort approx. for n > 30
    *  - Complexity: 6N lg(N) array access and Nlg(N) comparisions
    *
    * Modification:
    *  - back to insertion sort if number of elements to sort is less or equal to 30
    */
    template<class TElement, class Comparator, size_t thresholdForInsertionSort = 32>
    inline void mergeSort(TElement* firstItem, size_t length,
                          TElement* draftStorage,
                          Comparator cmpIsLess)
    {
        MergeSortResult<TElement> result = mergeSortInternal<TElement, Comparator, thresholdForInsertionSort> (firstItem, length, draftStorage, cmpIsLess);
        if (result.output != firstItem)
        {
            CopyHelpers::copy(firstItem, result.output, length);
        }
    }

    /**
     * @brief Merges two sorted subarrays into a single output array up to a specified number of elements.
     *
     * This function performs a partial merge of two sorted input ranges, placing the result into an output
     * range. The merging process stops as soon as 'k' elements have been merged.
     *
     * @param output The iterator pointing to the start of the destination range where the merged result will be placed.
     * @param firstA The iterator pointing to the start of the first input range.
     * @param endA The iterator pointing to the end of the first input range.
     * @param firstB The iterator pointing to the start of the second input range.
     * @param endB The iterator pointing to the end of the second input range.
     * @param k The maximum number of elements to merge into the output range.
     * @param cmpIsLess A comparison functor that returns true if the first argument is less than the second.
     */
    template<class IteratorOut, class IteratorIn, class Cmp>
    static void mergeInternalPartial(IteratorOut output,
                                     IteratorIn firstA, IteratorIn endA,
                                     IteratorIn firstB, IteratorIn endB,
                                     size_t k,
                                     Cmp cmpIsLess)
    {
        size_t mergedItems = 0;

        for (;;)
        {
            if (k == mergedItems)
                break;

            if (firstA != endA && firstB != endB)
            {
                if (cmpIsLess(*firstB, *firstA))  // In greater case elements from right subarray
                {
                    *output = *firstB;
                    ++output;
                    ++firstB;
                }
                else                              // In equal case or smaller case take element from left subarray
                {
                    *output = *firstA;
                    ++output;
                    ++firstA;
                }
                ++mergedItems;
            }
            else if (firstA != endA)
            {
                //size_t toMerge = minimum(k - mergedItems, size_t(endA - firstA));
                size_t toMerge = minimum(k - mergedItems, size_t(endA - firstA));
                CopyHelpers::copyWithIterators(output, firstA, firstA + toMerge);
                mergedItems += toMerge;
                break;
            }
            else if (firstB != endB)
            {
                // size_t toMerge = minimum(k - mergedItems, size_t(endB - firstB));
                size_t toMerge = minimum(k - mergedItems, size_t(endB - firstB));
                CopyHelpers::copyWithIterators(output, firstB, firstB + toMerge);
                mergedItems += toMerge;
                break;
            }
            else
            {
                break;
            }
        }

        assert(mergedItems <= k);
    }

    /**
     * @brief A structure to hold partial results of a merge sort operation.
     *
     * This structure is used to store and access intermediate results during the
     * merge sort algorithm.
     */
    template<class TElement>
    struct MergeSortResultPartial
    {
        /**
         * @brief Constructs a partial result of the merge sort operation.
         *
         * This constructor initializes the output array with a given pointer.
         *
         * @param theOutput A pointer to a TElement array which will hold the partial result of the merge sort.
         */
        MergeSortResultPartial(TElement* theOutput)
        : output(theOutput)
        {
        }

        /**
         * @brief A pointer to the array holding the sorted elements.
         *
         * This pointer is part of the MergeSortResultPartial structure and
         * references the array that contains the sorted elements after the
         * partial merge sort operation has completed.
         */
        TElement* output;
    };

    /**
     * Performs an internal partial merge sort on an array segment.
     *
     * @param first A pointer to the first element of the array segment to be sorted.
     * @param len The length of the array segment to be sorted.
     * @param k A size threshold for managing partition sizes.
     * @param draftStorage A pointer to an auxiliary array used during the merge process.
     * @param cmpIsLess A comparator function that returns true if the first argument is less than the second.
     * @return A MergeSortResultPartial<TElement> instance containing a pointer to the partially sorted array segment.
    */
    template<class TElement, class Comparator, size_t thresholdForInsertionSort>
    static MergeSortResultPartial<TElement> mergeSortInternalPartial(TElement* first, size_t len, size_t k, TElement* draftStorage, Comparator cmpIsLess)
    {
        if (len <= 1)
            return MergeSortResultPartial<TElement>(first);

        if (len <= thresholdForInsertionSort)
        {
            insertionSort(first, len, cmpIsLess);
            return MergeSortResultPartial<TElement>(first);
        }

        size_t pivot = len / 2;
        size_t firstHalfLen = pivot;
        size_t secondHalfLen = len - pivot;

        MergeSortResultPartial<TElement> firstHalfSortResult = 
            mergeSortInternalPartial<TElement, Comparator, thresholdForInsertionSort>(first, firstHalfLen, k, draftStorage, cmpIsLess);

        MergeSortResultPartial<TElement> seconHalfSortResult = 
            mergeSortInternalPartial<TElement, Comparator, thresholdForInsertionSort>(first + pivot, secondHalfLen, k, draftStorage + pivot, cmpIsLess);

        bool outFirst_InFirstBuffer = (firstHalfSortResult.output == first);
        bool outFirst_InSecondBuffer = (firstHalfSortResult.output == draftStorage);

        bool outSecond_InFirstBuffer = (seconHalfSortResult.output == first + pivot);
        bool outSecond_InSecondBuffer = (seconHalfSortResult.output == draftStorage + pivot);

        assert(outFirst_InFirstBuffer ^ outFirst_InSecondBuffer);
        assert(outSecond_InFirstBuffer ^ outSecond_InSecondBuffer);

        if (cmpIsLess(*(seconHalfSortResult.output),
                      *(firstHalfSortResult.output + minimum(firstHalfLen, k) - 1)))
        {
            if (outFirst_InSecondBuffer) // firstHalfSortResult.output == draftStorage
                draftStorage = first;

            // outFirst_InFirstBuffer & outSecond_InFirstBuffer => safe to merge
            // outFirst_InSecondBuffer & outSecond_InSecondBuffer => safe to merge
            // outFirst_InFirstBuffer & outSecond_InSecondBuffer => safe to merge because during merge it's ok to have it
            // outFirst_InSecondBuffer & outSecond_InFirstBuffer => safe to merge because during merge it's ok to have it

            mergeInternalPartial(draftStorage, firstHalfSortResult.output, firstHalfSortResult.output + firstHalfLen,
                                               seconHalfSortResult.output, seconHalfSortResult.output + secondHalfLen, k, cmpIsLess);

            // Swap draft storage and result storage
            return MergeSortResultPartial<TElement>(draftStorage);
        }
        else
        {
            // Improvement: Corner case when there is nothing to merge
            // 
            // ... < first[pivot - 2] < first[pivot - 1] <= first[pivot] < first[pivot+1] < ....

            if (outFirst_InFirstBuffer && outSecond_InFirstBuffer)
            {
                // If output is contigious lie inside first
                return MergeSortResultPartial<TElement>(first);
            }
            else if (outFirst_InSecondBuffer && outSecond_InSecondBuffer)
            {
                // If output is contigious lie inside draftStorage
                return MergeSortResultPartial<TElement>(draftStorage);
            }
            else
            {
                // Outputs due to logic and various optimization lie in different arrays
                if (outFirst_InSecondBuffer)
                    draftStorage = first;

                mergeInternalPartial(draftStorage, firstHalfSortResult.output, firstHalfSortResult.output + firstHalfLen,
                                     seconHalfSortResult.output, seconHalfSortResult.output + secondHalfLen, k, cmpIsLess);

                // Swap draft storage and result storage
                return MergeSortResultPartial<TElement>(draftStorage);
            }
        }
    }

    /** Modification of merge sort.
    * About Merge Sort:
    *  - sort is not in place.
    *  - sort is stable. I.e. sort preserves the relative order of equal elements.
    *  - beats insertion sort approx. for n > 30
    *  - Complexity: 6N lg(N) array access and Nlg(N) comparisions
    *
    * Modification:
    *  - back to insertion sort if number of elements to sort is less or equal to 30
    */
    template<class TElement, class Comparator, size_t thresholdForInsertionSort = 32>
    inline void mergeSortPartial(TElement* firstItem, size_t length, size_t k,
                                TElement* draftStorage,
                                Comparator cmpIsLess)
    {
        MergeSortResultPartial<TElement> result = mergeSortInternalPartial
                                                    <TElement, Comparator, thresholdForInsertionSort>
                                                        (firstItem, length, k, draftStorage, cmpIsLess);
        
        if (result.output != firstItem)
        {
            CopyHelpers::copy(firstItem, result.output, minimum(length,k));
        }
    }

    /** More or less correct implementation which works fine with duplicate keys
    * R.Sedjvik: "Up to 1990 all implementations in most textbooks were incorrect"
    */
    template<class TElement, class Comparator>
    TElement* partitionSedjvick(TElement* first, TElement* end, Comparator cmpIsLess)
    {
        TElement x = *first; // assumption: pivot in first position

        TElement* i = first + 1;
        TElement* j = end - 1;

        for (;;)
        {
            for (; i != end && cmpIsLess(*i, x); ++i)
                ;

            for (; j != first && cmpIsLess(x, *j); --j)
                ;

            if (i >= j)
            {
                // Case: 1
                // *j <= x <= *i ==> *j <= *i
                // However this inequality is here when in fact roles of "i" and "j" are swapped

                // Case: 2 - first array has been exhausted
                // 
                // Case: 3 - second array has been exhausted
                // 
                // ............................................................................
                // 
                // Both (2) and (3) are done implicitly.
                
                break;
            }
            else
            {
                dopt::CopyHelpers::swap(i, j);
                i++;
                j--;
            }
        }

        // case-1: *j is less then *pivot (*first) => *pivot is bigger then *j
        // case-2: first and j point to the same place
        
        if (first != j)
            dopt::CopyHelpers::swapDifferentObjects(first, j);

        return j;
    }

    /** Sort elements [first, end)
    */
    template<class TElement, class Comparator, 
             size_t thresholdForSimpleSort = 16,
             bool use_quicksort_for_find_smallest_k = true,
             bool select_pivot_uniformly_at_random = false>
    void quickSortExtended(TElement* firstItem, size_t len, size_t k, Comparator cmpIsLess)
    {
        // Random generator
        //dopt::RandomGenIntegerLinear gen;

        for (;;)
        {
            // 1. Special optimization for top-k
            if constexpr (use_quicksort_for_find_smallest_k)
            {
                if (len <= k)
                {
                    // no need to sort
                    return;
                }

                if constexpr (thresholdForSimpleSort > 0)
                {
                    if (k <= thresholdForSimpleSort)
                    {
                        selectionSortExtended(firstItem, len, k, cmpIsLess);
                        return;
                    }
                }
                else
                {
                    if (len <= 1)
                    {
                        // no need to sort
                        return;
                    }
                }
            }
            else
            {
                if constexpr (thresholdForSimpleSort > 0)
                {
                    if (len <= thresholdForSimpleSort)
                    {
                        insertionSort(firstItem, len, cmpIsLess);
                        return;
                    }
                }
                else
                {
                    if (len <= 1)
                    {
                        // no need to sort
                        return;
                    }
                }
            }

#if 0
            if constexpr (select_pivot_uniformly_at_random)
            {
                dopt::CopyHelpers::swap(firstItem, firstItem + gen.generateInteger() % len);
            }
#endif

            TElement* q = partitionSedjvick(firstItem, firstItem + len, cmpIsLess);

            size_t leftFromPivotWithPivot = q - firstItem + 1;

            if (leftFromPivotWithPivot < k)
            {
                // in case if leftFromPivotWithPivot >= k there is no reason to sort right subarray
                quickSortExtended<TElement, 
                                  Comparator, 
                                  thresholdForSimpleSort,
                                  use_quicksort_for_find_smallest_k>
                              (q + 1, len - leftFromPivotWithPivot, k - leftFromPivotWithPivot, cmpIsLess);
            }

            // TAIL RECURSION
            len = leftFromPivotWithPivot - 1;
        }
    }

    /**
     * Sorts an array using the QuickSort algorithm.
     *
     * @param firstItem A pointer to the first element in the array to be sorted.
     * @param len The number of elements in the array to be sorted.
     * @param cmpIsLess A comparator function that returns true if the first argument is less than the second.
    */
    template<class TElement, class Comparator,
             size_t thresholdForInsertionSort = 32,
             bool select_pivot_uniformly_at_random = true>
    void quickSort(TElement* firstItem, size_t len, Comparator cmpIsLess)
    {
        // Random generator
        dopt::RandomGenIntegerLinear gen;

        for (;;)
        {
            if (len <= 1)
                return;

            if (len <= thresholdForInsertionSort)
            {
                insertionSort(firstItem, len, cmpIsLess);
                return;
            }

            if constexpr (select_pivot_uniformly_at_random)
            {
                dopt::CopyHelpers::swap(firstItem, firstItem + gen.generateInteger() % len);
            }

            TElement* q = partitionSedjvick(firstItem, firstItem + len, cmpIsLess);
            
            size_t leftFromPivotWithPivot = q - firstItem + 1;

            // in case if leftFromPivotWithPivot >= k there is no reason to sort right subarray
            quickSort<TElement, 
                      Comparator, 
                      thresholdForInsertionSort, 
                      select_pivot_uniformly_at_random>
                      (q + 1, len - leftFromPivotWithPivot, cmpIsLess);
            

            // TAIL RECURSION
            len = leftFromPivotWithPivot - 1;
        }
    }

    /**
     * Extracts a byte from a given integral value.
     *
     * @param value The integral value from which to extract the byte.
     * @param byteNumber The index of the byte to extract, starting from 0.
     * @return The byte at the specified index within the given integral value.
    */
    template<class T>
    uint8_t byteFromIntegralType(T value, size_t byteNumber)
    {
        static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");

        uint8_t* ptr = (uint8_t*)&value;
        return ptr[byteNumber];
    }

    /** Sort elements [first, first + length)
    */
    template<class TElement>
    void radixSort(TElement* first, size_t length)
    {
        if (length == 0)
            return;

        TElement* outputBuffer = first;
        TElement* lastBufferUsedAsDestination = nullptr;
        TElement* tempBuffer = (TElement *) alloca( sizeof(TElement) * length);

        // number of round equal to number of bytes in representation of "IndexType"
        size_t rounds = sizeof(TElement);

        // allocate on stack and initialize to zero
        const size_t kCounterLength = 255 + 1;
        size_t counter[kCounterLength];

        for (size_t r = 0; r < rounds; ++r)
        {
            // Make counters equal to zero
            memset(counter, 0, sizeof(counter));

            for (size_t j = 0; j != length; ++j)
            {
                auto data = first[j];
                counter[byteFromIntegralType(data, r)]++;
            }

            for (size_t k = 1; k < kCounterLength; ++k)
            {
                counter[k] = counter[k] + counter[k - 1];
            }

            // counter[i] contains number of elements <= "min + i"

            for (size_t j = length - 1; ; --j)
            {
                auto data = first[j];
                uint8_t curByte = byteFromIntegralType(data, r);
                tempBuffer[counter[curByte] - 1] = data;
                counter[curByte]--;
                if (j == 0)
                    break;
            }

            lastBufferUsedAsDestination = tempBuffer;

            {
                TElement* tmp = first;
                first = tempBuffer;
                tempBuffer = tmp;
            }
            // dopt::CopyHelpers::copy(first, tempBuffer, tempBuffer + length);
        }

        if (outputBuffer != lastBufferUsedAsDestination)
        {
            dopt::CopyHelpers::copy(outputBuffer, lastBufferUsedAsDestination, length);
        }
    }

    /**
     * Retrieves the i-th smallest item from an array.
     *
     * @param firstItem Pointer to the first element of the array.
     * @param len The length of the array.
     * @param cmpIsLess A comparator function to determine the order of elements.
     * @param i The zero-based index of the desired smallest element.
     * @return The i-th smallest element in the array.
    */
    template<class TElement, class Comparator, bool select_pivot_uniformly_at_random = true>
    TElement getIthSmallestItem(TElement* firstItem, size_t len, Comparator cmpIsLess, size_t i)
    {
        // Random generator
        dopt::RandomGenIntegerLinear gen;

        for (;;)
        {
            if (len == 1)
            {
                assert(i == 0);
                return (*firstItem);
            }

            // Base case (to try)

            constexpr size_t insertionSortThreshold = 9;

            if (len <= insertionSortThreshold)
            {
                insertionSort(firstItem, len, cmpIsLess);
                return firstItem[i];
            }

            if constexpr (select_pivot_uniformly_at_random)
            {
                constexpr size_t presSelectN = 3;

                if (len >= presSelectN)
                {
                    std::array<size_t, presSelectN> smallArray;
                    for (size_t i = 0; i < smallArray.size(); ++i)
                        smallArray[i] = gen.generateInteger() % len;

                    insertionSort(smallArray.data(), smallArray.size(), [&firstItem, &cmpIsLess](size_t a, size_t b)
                                                                        {
                                                                            return cmpIsLess(firstItem[a], firstItem[b]);
                                                                        }
                                 );

                    size_t median_index = smallArray[presSelectN/2];
                    dopt::CopyHelpers::swap(firstItem, firstItem + median_index);
                }
                else
                {
                    size_t pivotIndex = gen.generateInteger() % len;
                    dopt::CopyHelpers::swap(firstItem, firstItem + pivotIndex);
                }
            }

            TElement* q = partitionSedjvick(firstItem, firstItem + len, cmpIsLess);

            size_t leftFromPivotWithoutPivot = q - firstItem;
            
            if (leftFromPivotWithoutPivot == i)
            {
                return (*q);
            }
            else if (i < leftFromPivotWithoutPivot)
            {
                len = leftFromPivotWithoutPivot;
            }
            else
            {
                firstItem = q + 1;
                len -= (leftFromPivotWithoutPivot + 1);
                i -= (leftFromPivotWithoutPivot + 1);
            }
        }
    }
}
