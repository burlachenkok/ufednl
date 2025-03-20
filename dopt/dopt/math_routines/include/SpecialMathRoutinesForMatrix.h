#pragma once

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/math_routines/include/Sorting.h"
#include "dopt/math_routines/include/MinHeapIndexed.h"
#include "dopt/math_routines/include/MinHeap.h"

#include "dopt/random/include/Shuffle.h"
#include "dopt/linalg_vectors/include/LightVectorND.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/random/include/RandomGenRealLinear.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

#include <cmath>
#include <set>
#include <vector>
#include <algorithm>

#define SORT dopt::quickSortExtended(&itemsIndicies[0], itemsIndicies.size(), k, comp);

namespace dopt
{
    /**
     * Generates a list of indices for the upper triangular part of a given matrix,
     * including the main diagonal.
     *
     * @param m The input matrix for which upper triangular indices need to be generated.
     * @return A vector containing the indices corresponding to the elements in the upper triangular part of the matrix.
     */
    template<class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> indiciesForUpperTriangularPart(const Matrix& m)
    {        
        size_t mCols = m.columns();

        // fast division by two of a garanteed even number
        std::vector<TIndexType> indicies(((mCols + 1) * mCols) >> 1);
        TIndexType* restrict_ext indiciesRaw = indicies.data();
        
        // write_post points to next write in indiciesRaw
        size_t write_pos = 0;

        // read_pos is index in column order the first element of the matrix
        TIndexType read_pos = 0;

        TIndexType mLda = static_cast<TIndexType>(m.LDA);

        // Loop by columns
        for (size_t j = 0; j < mCols; ++j, read_pos += mLda)
        {
            size_t lenToSetup = j + 1;
            dopt::LightVectorND<dopt::VectorNDRaw<TIndexType>>  rawIndiciesView(indiciesRaw + write_pos, lenToSetup);
            rawIndiciesView.assignIncreasingSequence(read_pos); /// Assign read_pos, read_pos+1,....
            write_pos += lenToSetup;            
        }
        
        assert(write_pos == indicies.size());

        return indicies;
    }

    /**
     * Generates a specified number of random items within the upper triangular part of a matrix.
     *
     * @param matrix The input matrix where items will be selected.
     * @param k The number of random items to generate.
     * @return A list of generated random items from the upper triangular part of the matrix.
    */
    template<class Generator, class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> generateRandKItemsInUpperTriangularPart(Generator& rndGenerator,
                                                                                          size_t k,
                                                                                          const Matrix& m)
    {
        std::vector<TIndexType> indicies = indiciesForUpperTriangularPart<Matrix, TIndexType> (m);
        dopt::shuffle(indicies, k, rndGenerator);
        indicies.resize(k);

        std::sort(indicies.begin(), indicies.end(), [](TIndexType a, TIndexType b) {return a < b; });
        return indicies;
    }

    /**
     * Generates a specified number of random indices within the upper triangular part of a matrix.
     *
     * @param rndGenerator The random generator to be used for selection of indices.
     * @param k The number of random indices to generate.
     * @param m The input matrix whose upper triangular part is considered.
     * @param indiciesOfUpperTriangularPart A vector containing the indices of the upper triangular part of the matrix.
     * @return A vector containing k random indices from the upper triangular part of the matrix.
    */
    template<class Generator, class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> generateRandKItemsInUpperTriangularPart(Generator& rndGenerator,
                                                                                         size_t k, 
                                                                                         const Matrix& m,
                                                                                         const std::vector<uint32_t>& indiciesOfUpperTriangularPart)
    {
        std::vector<TIndexType> indicies(indiciesOfUpperTriangularPart);
        dopt::shuffle(indicies, k, rndGenerator);
        indicies.resize(k);
        
        std::sort(indicies.begin(), indicies.end(), [](TIndexType a, TIndexType b) {return a < b; });
        return indicies;
    }

    /**
     * Generates a random index within the upper triangular part of a given matrix, including the main diagonal.
     *
     * @param rndGenerator The random generator used to generate the random index.
     * @param k The number of random items to generate (not used in the current implementation).
     * @param m The input matrix for which the random index is generated.
     * @return A random index within the upper triangular part of the matrix.
     * @note index does not take into account LDA
    */
    template<class Generator, class Matrix>
    forceinline_ext static uint32_t generateRandSeqKItemsInUpperTriangularPartAsIndex(Generator& rndGenerator,
                                                                                      size_t k,
                                                                                      const Matrix& m)
    {
        size_t sz = (m.rows() * (m.rows() + 1)) / 2;
        uint32_t r = rndGenerator.generateInteger() % sz;
        return r;
    }

    /**
     * Generates a random sequence of `k` items within the upper triangular part of a matrix.
     *
     * @param rndGenerator The random number generator to use for sequence generation.
     * @param k The number of items to select from the upper triangular part.
     * @param m The matrix from which to derive the upper triangular part (currently not used in function body).
     * @param indiciesOfUpperTriangularPart A vector containing indices corresponding to the upper triangular part of the matrix.
     * @return A vector containing `k` randomly selected indices from the upper triangular part of the matrix.
    */
    template<class Generator, class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> generateRandSeqKItemsInUpperTriangularPart(Generator& rndGenerator,
                                                                                            size_t k,
                                                                                            const Matrix& m,
                                                                                            const std::vector<TIndexType>& indiciesOfUpperTriangularPart)
    {
        size_t sz = indiciesOfUpperTriangularPart.size();
        
        uint32_t r = rndGenerator.generateInteger() % sz;
        
        if (r + k <= sz) [[likely]]
        {
            // Copy in one operation
            std::vector<TIndexType> indicies(indiciesOfUpperTriangularPart.data() + r,
                                             indiciesOfUpperTriangularPart.data() + r + k
                                            );
            return indicies;
        }
        else
        {
            std::vector<TIndexType> indicies(k, TIndexType());

            // 
            size_t first_part_len  = sz - r;
            size_t second_part_len = k - first_part_len;

            // Copy in two operations
            dopt::CopyHelpers::copy(indicies.data(), indiciesOfUpperTriangularPart.data() + r, first_part_len);
            dopt::CopyHelpers::copy(indicies.data() + first_part_len, indiciesOfUpperTriangularPart.data(), second_part_len);

            return indicies;
        }
    }

    /**
     * Generates a specified number of random indices within the upper triangular part of a matrix.
     *
     * @param rndGenerator The random number generator to use for index selection.
     * @param k The number of random indices to generate.
     * @param m The input matrix whose upper triangular part is considered.
     * @return A vector containing `k` randomly selected indices from the upper triangular part of the matrix.
    */
    template<class Generator, class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> generateRandSeqKItemsInUpperTriangularPart(Generator& rndGenerator,
                                                                                              size_t k,
                                                                                              const Matrix& m)
    {
        std::vector<TIndexType> indicies = indiciesForUpperTriangularPart<Matrix, TIndexType> (m);
        return generateRandSeqKItemsInUpperTriangularPart<Generator, Matrix, TIndexType> (rndGenerator, k, m, indicies);
    }

    /**
     * Computes the variance factor 'w' for a matrix operator based on the parameters k and d.
     *
     * @param k The number of items to be selected.
     * @param d The dimension size of the matrix.
     * @return The computed weighting factor 'w'.
     */
    forceinline_ext double computeWForRandKMatrixOperator(size_t k, size_t d)
    {
        size_t totalPossibleItems2Send = ((d + 1) * d) / 2;
        double w = (double(totalPossibleItems2Send) / k - 1.0);
        return w;
    }

    /**
     * Computes the contraction factor delta for a matrix operator based on the parameters k and d.
     *
     * @param k The number of items to be selected.
     * @param d The dimension size of the matrix.
     * @return The computed contraction factor delta.
     */
    forceinline_ext double computeDeltaForTopKMatrixOpeator(size_t k, size_t d)
    {
#if DOPT_FIX_TOPK_CONTRACTION_FACTOR
        double delta = double(k) / (double(d) * double(d));
#else
        size_t totalPossibleItems2Send = ((d + 1) * d) / 2;
        double delta = double(k) / (totalPossibleItems2Send);
#endif
        return delta;
    }

    /**
     * Retrieves the top 'k' indices from the upper diagonal part of a given matrix.
     * The selection process is based on the values in the specified 'indiciesOfUpperTriangularPart'.
     *
     * @param matrix The matrix from which to retrieve the indices.
     * @param k The number of top indices to retrieve.
     * @param indiciesOfUpperTriangularPart A vector containing indices of the upper triangular part of the matrix.
     * @return A vector containing the top 'k' indices from the upper diagonal part of the matrix.
    */
    template<bool ignoreSign, class TMat, typename TIndexType = uint32_t>
    forceinline_ext std::vector<TIndexType> getTopKFromUpperDiagonalPart(const TMat& matrix,
                                                                         size_t k,
                                                                         const std::vector<TIndexType>& indiciesOfUpperTriangularPart)
    {
        constexpr size_t kAlgorithm = 2;

        if (kAlgorithm == 1)
        {
            // sorting-based algorithm
            std::vector<TIndexType> itemsIndicies(indiciesOfUpperTriangularPart);

            size_t sz = itemsIndicies.size();

            if (k >= sz)
                return itemsIndicies;

            if constexpr (ignoreSign)
            {
                auto cmp = [&matrix](TIndexType a, TIndexType b) {
                    return dopt::isFirstHigherThenSecondIgnoreSign(matrix.matrixByCols.get(a),
                                                                   matrix.matrixByCols.get(b));
                };

                // Use knowledge about implementation of algorithm which in fact partially sorts
                TIndexType topKthItemIndex = dopt::getIthSmallestItem(itemsIndicies.data(), sz, cmp, k);
                itemsIndicies.resize(k);

                std::sort(itemsIndicies.begin(), itemsIndicies.end(), [](TIndexType a, TIndexType b) {return a < b; });

                return itemsIndicies;
            }
            else
            {
                auto cmp = [&matrix](TIndexType a, TIndexType b) {
                    return (matrix.matrixByCols.get(a)) > (matrix.matrixByCols.get(b));
                };
                
                // Use knowledge about implementation of algorithm which in fact partially sorts
                TIndexType topKthItemIndex = dopt::getIthSmallestItem(itemsIndicies.data(), sz, cmp, k);
                itemsIndicies.resize(k);

                std::sort(itemsIndicies.begin(), itemsIndicies.end(), [](TIndexType a, TIndexType b) {return a < b; });

                return itemsIndicies;
            }
        }
        else
        {
            // min-heap based algorithm
            if (k >= indiciesOfUpperTriangularPart.size())
                return indiciesOfUpperTriangularPart;

            // used factor for a heap, 4 is just a good practical nubmer
            constexpr size_t kHeapBranchFactor = 4;

            if constexpr (ignoreSign)
            {
                auto convert_to_comparable_abs = [&matrix](TIndexType a) {
                    return dopt::abs(matrix.matrixByCols.get(a));
                };

                // construct heap from first k-item
                std::vector<TIndexType> itemsIndiciesMinHeap(indiciesOfUpperTriangularPart.begin(),
                                                             indiciesOfUpperTriangularPart.begin() + k);
                
                dopt::MinHeap<kHeapBranchFactor> minHeap;

                minHeap.buildMinHeap(itemsIndiciesMinHeap, 
                                     convert_to_comparable_abs);

                size_t sz = indiciesOfUpperTriangularPart.size();

                // extract min index and value of min value
                TIndexType minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                typename TMat::TElementType minIndexValue = convert_to_comparable_abs(minIndex);

                // loop through the rest
                for (uint32_t i = k; i < sz; ++i)
                {
                    auto itemToTryIndex = indiciesOfUpperTriangularPart[i];

                    if (convert_to_comparable_abs(itemToTryIndex) > minIndexValue)
                    {
                        minHeap.increaseMinimumInMinHeap(itemsIndiciesMinHeap, itemToTryIndex, convert_to_comparable_abs);

                        minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                        minIndexValue = convert_to_comparable_abs(minIndex);
                    }
                }

                // sorting to make indicies following sequentially
                std::sort(itemsIndiciesMinHeap.begin(), 
                          itemsIndiciesMinHeap.end(), 
                          [](TIndexType a, TIndexType b) {return a < b; }
                         );
                
                return itemsIndiciesMinHeap;
            }
            else
            {
                auto convert_to_comparable = [&matrix](TIndexType a) {
                    return matrix.matrixByCols.get(a);
                };

                std::vector<TIndexType> itemsIndiciesMinHeap(indiciesOfUpperTriangularPart.begin(),
                                                           indiciesOfUpperTriangularPart.begin() + k);
                
                dopt::MinHeap<kHeapBranchFactor> minHeap;

                minHeap.buildMinHeap(itemsIndiciesMinHeap, convert_to_comparable);

                size_t sz = indiciesOfUpperTriangularPart.size();

                TIndexType minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                typename TMat::TElementType minIndexValue = convert_to_comparable(minIndex);

                for (uint32_t i = k; i < sz; ++i)
                {
                    auto itemToTryIndex = indiciesOfUpperTriangularPart[i];

                    if (convert_to_comparable(itemToTryIndex) > minIndexValue)
                    {
                        minHeap.increaseMinimumInMinHeap(itemsIndiciesMinHeap, itemToTryIndex, convert_to_comparable);

                        minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                        minIndexValue = convert_to_comparable(minIndex);
                    }
                }

                std::sort(itemsIndiciesMinHeap.begin(), 
                          itemsIndiciesMinHeap.end(), 
                          [](TIndexType a, TIndexType b) {return a < b; });

                return itemsIndiciesMinHeap;
            }
        }
    }

    /**
    * Retrieves the top less-equal K indices from the upper diagonal part of a matrix based on a specified delta value.
    *
    * @param matrix The input matrix from which to retrieve the top K indices.
    * @param k The number of top indices to retrieve.
    * @param indiciesOfUpperTriangularPart Vector containing the indices of the upper triangular part of the matrix.
    * @param delta A delta value used to derive need criteria
    * @return A vector containing the top K indices from the upper diagonal part of the matrix.
    */
    template<bool ignoreSign, class TMat, typename TIndexType = uint32_t>
    forceinline_ext std::vector<TIndexType> getTopLEKFromUpperDiagonalPart(const TMat& matrix,
                                                                           size_t k,
                                                                           const std::vector<TIndexType>& indiciesOfUpperTriangularPart,
                                                                           double delta)
    {
        thread_local dopt::RandomGenRealLinear generator;

        double one_minus_delta = 1.0 - delta;
        auto mFrobNormSqr = matrix.frobeniusNormSquareForSymmetricMatrixFromUpPart();
        double one_minus_delta_xf = one_minus_delta * mFrobNormSqr;

        // Derived constant to fix numerical problems
        //===================================================
        double kEspForNumerics = 1e-1 * mFrobNormSqr;
        //===================================================

        {
            if (k >= indiciesOfUpperTriangularPart.size())
                return indiciesOfUpperTriangularPart;

            constexpr size_t kHeapBranchFactor = 4;

            if constexpr (ignoreSign)
            {
                auto convert_to_comparable_abs = [&matrix](TIndexType a) {
                    return dopt::abs(matrix.matrixByCols.get(a));
                };

                std::vector<TIndexType> itemsIndiciesMinHeap(indiciesOfUpperTriangularPart.begin(),
                                                             indiciesOfUpperTriangularPart.begin() + k);

                dopt::MinHeap<kHeapBranchFactor> minHeap;

                minHeap.buildMinHeap(itemsIndiciesMinHeap,
                                     convert_to_comparable_abs);

                // Scan through elements in the upper triangular part of the matrix
                {
                    size_t sz = indiciesOfUpperTriangularPart.size();

                    TIndexType minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                    typename TMat::TElementType minIndexValue = convert_to_comparable_abs(minIndex);

                    for (uint32_t i = k; i < sz; ++i)
                    {
                        auto itemToTryIndex = indiciesOfUpperTriangularPart[i];

                        if (convert_to_comparable_abs(itemToTryIndex) > minIndexValue)
                        {
                            minHeap.increaseMinimumInMinHeap(itemsIndiciesMinHeap, itemToTryIndex, convert_to_comparable_abs);

                            minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                            minIndexValue = convert_to_comparable_abs(minIndex);
                        }
                    }
                }

                // Prepare numerator [remove effect from TopK operator]
                auto numerator = mFrobNormSqr;
                for (size_t i = 0; i < itemsIndiciesMinHeap.size(); ++i)
                {
                    uint32_t index = itemsIndiciesMinHeap[i];

                    auto item = matrix.matrixByCols[index];

                    if (index == matrix.getTranspoedIndexFromFlatternIndex(index))
                    {
                        numerator -= item * item;
                    }
                    else
                    {
                        numerator -= 2.0 * item * item;
                    }
                }

                // Scan and find need index for perform rounding
                do
                {
                    auto previous_one_minus_delta_xf = (numerator);  // / (mFrobNormSqr);

                    // Add effect from TopK operator one by one
                    TIndexType minIndex = minHeap.extracFromMinHeap<TIndexType>(itemsIndiciesMinHeap, convert_to_comparable_abs);
                    auto item = matrix.matrixByCols[minIndex];
                    if (minIndex == matrix.getTranspoedIndexFromFlatternIndex(minIndex))
                    {
                        numerator += item * item;
                    }
                    else
                    {
                        numerator += 2.0 * item * item;
                    }

                    auto next_one_minus_delta_xf = (numerator);  // / (mFrobNormSqr);


                    assert(next_one_minus_delta_xf >= previous_one_minus_delta_xf);

                    if (one_minus_delta_xf + kEspForNumerics > previous_one_minus_delta_xf &&
                        one_minus_delta_xf < next_one_minus_delta_xf + kEspForNumerics)
                    {
                        // j -- previous iteration (with minIndex)
                        // i -- next iteration (without minIndex)
                        //    p = (a_j - a) / (a_j - a_i) 
                        //      = (a_j - 1 + 1 - a) / (-1 + a_j + 1 - a_i) 
                        //      = ( (1 - a) - (1 - a_j) ) / ( (1 - a_i) - (1 - a_j) )
                        //     
                        //    and Frob norm square just cancels
                        //
                        auto p = (one_minus_delta_xf - previous_one_minus_delta_xf + kEspForNumerics) /
                                 (next_one_minus_delta_xf - previous_one_minus_delta_xf + kEspForNumerics);


                        if (generator.generateRealInUnitInterval() > p)
                        {
                            // w.p. "1 - p"
                            // select compressor from previous iteration "j". However the difference it include or not "minIndex". In previous iteration we include "minIndex"
                            minHeap.insertInMinHeap(itemsIndiciesMinHeap, minIndex, convert_to_comparable_abs);
                            break;
                        }
                        else
                        {
                            // w.p. "p"
                            break;
                        }
                    }
                } while (!itemsIndiciesMinHeap.empty());

                std::sort(itemsIndiciesMinHeap.begin(),
                          itemsIndiciesMinHeap.end(),
                          [](TIndexType a, TIndexType b) {return a < b; });


                return itemsIndiciesMinHeap;
            }
            else
            {
                auto convert_to_comparable = [&matrix](TIndexType a) {
                    return matrix.matrixByCols.get(a);
                };

                std::vector<TIndexType> itemsIndiciesMinHeap(indiciesOfUpperTriangularPart.begin(),
                                                             indiciesOfUpperTriangularPart.begin() + k);
                dopt::MinHeap<kHeapBranchFactor> minHeap;
                minHeap.buildMinHeap(itemsIndiciesMinHeap, convert_to_comparable);

                // Scan through elements in the upper triangular part of the matrix
                {
                    size_t sz = indiciesOfUpperTriangularPart.size();
                    TIndexType minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);

                    typename TMat::TElementType minIndexValue = convert_to_comparable(minIndex);

                    for (uint32_t i = k; i < sz; ++i)
                    {
                        auto itemToTryIndex = indiciesOfUpperTriangularPart[i];

                        if (convert_to_comparable(itemToTryIndex) > minIndexValue)
                        {
                            minHeap.increaseMinimumInMinHeap(itemsIndiciesMinHeap, itemToTryIndex, convert_to_comparable);

                            minIndex = minHeap.peekFromMinHeap<TIndexType>(itemsIndiciesMinHeap);
                            minIndexValue = convert_to_comparable(minIndex);
                        }
                    }
                }
                
                // Prepare numerator [remove effect from TopK operator]
                auto numerator = mFrobNormSqr;
                for (size_t i = 0; i < itemsIndiciesMinHeap.size(); ++i)
                {
                    TIndexType index = itemsIndiciesMinHeap[i];

                    auto item = matrix.matrixByCols[index];

                    if (index == matrix.getTranspoedIndexFromFlatternIndex(index))
                    {
                        numerator -= item * item;
                    }
                    else
                    {
                        numerator -= 2.0 * item * item;
                    }
                }
                                
                // Scan and find need index for perform rounding
                do
                {
                    auto previous_one_minus_delta_xf = (numerator);// / (mFrobNormSqr);

                    // Add effect from TopK operator one by one
                    TIndexType minIndex = minHeap.extracFromMinHeap<TIndexType>(itemsIndiciesMinHeap, convert_to_comparable);
                    auto item = matrix.matrixByCols[minIndex];
                    if (minIndex == matrix.getTranspoedIndexFromFlatternIndex(minIndex))
                    {
                        numerator += item * item;
                    }
                    else
                    {
                        numerator += 2.0 * item * item;
                    }

                    auto next_one_minus_delta_xf = (numerator); // / (mFrobNormSqr);

                    // Fix numerical problems [start]
                    //if (next_one_minus_delta > 1.0)
                    //    next_one_minus_delta = 1.0;
                    // Fix numerical problems [end]

                    assert(next_one_minus_delta_xf > previous_one_minus_delta_xf);

                    if (one_minus_delta_xf >= previous_one_minus_delta_xf && one_minus_delta_xf <= next_one_minus_delta_xf)
                    {
                        // j -- previous iteration (with minIndex)
                        // i -- next iteration (without minIndex)
                        //    p = (a_j - a) / (a_j - a_i) = (a_j - 1 + 1 - a) / (-1 + a_j + 1 - a_i) = ( (1 - a) - (1 - a_j) ) / ( (1 - a_i) - (1 - a_j) )

                        auto p = (one_minus_delta_xf - previous_one_minus_delta_xf) / (next_one_minus_delta_xf - previous_one_minus_delta_xf);
                        assert(p >= 0.0 && p <= 1.0);

                        if (generator.generateRealInUnitInterval() > p)
                        {
                            // w.p. "1 - p"
                            // select compressor from previous iteration "j". However the difference it include or not "minIndex". In previous iteration we include "minIndex"
                            minHeap.insertInMinHeap(itemsIndiciesMinHeap, minIndex, convert_to_comparable);
                            break;
                        }
                        else
                        {
                            // w.p. "p"
                            break;
                        }
                    }
                } while (!itemsIndiciesMinHeap.empty());
                
                std::sort(itemsIndiciesMinHeap.begin(),
                          itemsIndiciesMinHeap.end(),
                          [](TIndexType a, TIndexType b) {return a < b; });

                return itemsIndiciesMinHeap;
            }
        }
    }

    /**
     * Retrieves the top 'k' indices from the upper diagonal part of a matrix which correspond to top 'k' item
     *
     * @param matrix The input matrix to be processed.
     * @param k The number of top indices to retrieve.
     * @return A vector containing the top 'k' indices from the upper diagonal portion of the matrix.
     * @tparam ignoreSign ignore sign during comparision
     */
    template<bool ignoreSign, class TMat, typename TIndexType = uint32_t>
    forceinline_ext std::vector<TIndexType> getTopKFromUpperDiagonalPart(const TMat& matrix, size_t k)
    {
        std::vector<TIndexType> itemsIndicies = indiciesForUpperTriangularPart<TMat, TIndexType> (matrix);
        return getTopKFromUpperDiagonalPart<ignoreSign, TMat, TIndexType> (matrix, k, itemsIndicies);
    }
}
