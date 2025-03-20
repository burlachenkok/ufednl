#pragma once

#include "dopt/gpu_compute_support/include/linalg_matrices/MatrixNMD_CUDA.h"
#include "dopt/gpu_compute_support/include/linalg_vectors/VectorND_CUDA_Raw.h"

namespace dopt
{
    template<class VectorType, typename TIndexType = uint32_t>
    VectorND_CUDA_Raw<TIndexType> indiciesForUpperTriangularPart(const MatrixNMD_CUDA<VectorType>& m)
    {        
        assert(m.rows() == m.columns());

        size_t mCols = m.columns();
        VectorND_CUDA_Raw<TIndexType> indicies(((mCols + 1) * mCols) >> 1);
        
        applyKernelToFillInIndiciesForUpperTriangPart(indicies.rawDevData(),
                                                      m.rows(),
                                                      indicies.device());
        
        return indicies;
    }
    
#if 0
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

    template<class Generator, class Matrix>
    forceinline_ext static uint32_t generateRandSeqKItemsInUpperTriangularPartAsIndex(Generator& rndGenerator,
                                                                                      size_t k,
                                                                                      const Matrix& m)
    {
        size_t sz = (m.rows() * (m.rows() + 1)) / 2;
        uint32_t r = rndGenerator.generateInteger() % sz;
        return r;
    }
    
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
    
    template<class Generator, class Matrix, typename TIndexType = uint32_t>
    forceinline_ext static std::vector<TIndexType> generateRandSeqKItemsInUpperTriangularPart(Generator& rndGenerator,
                                                                                              size_t k,
                                                                                              const Matrix& m)
    {
        std::vector<TIndexType> indicies = indiciesForUpperTriangularPart<Matrix, TIndexType> (m);
        return generateRandSeqKItemsInUpperTriangularPart<Generator, Matrix, TIndexType> (rndGenerator, k, m, indicies);
    }

    forceinline_ext double computeWForRandKMatrixOperator(size_t k, size_t d)
    {
        size_t totalPossibleItems2Send = ((d + 1) * d) / 2;
        double w = (double(totalPossibleItems2Send) / k - 1.0);
        return w;
    }

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
    
    template<bool ignoreSign, class TMat, typename TIndexType = uint32_t>
    forceinline_ext std::vector<TIndexType> getTopKFromUpperDiagonalPart(const TMat& matrix,
                                                                         size_t k,
                                                                         const std::vector<TIndexType>& indiciesOfUpperTriangularPart)
    {
        constexpr size_t kAlgorithm = 2;

        if (kAlgorithm == 1)
        {
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
    
    template<bool ignoreSign, class TMat, typename TIndexType = uint32_t>
    forceinline_ext std::vector<TIndexType> getTopKFromUpperDiagonalPart(const TMat& matrix, size_t k)
    {
        std::vector<TIndexType> itemsIndicies = indiciesForUpperTriangularPart<TMat, TIndexType> (matrix);
        return getTopKFromUpperDiagonalPart<ignoreSign, TMat, TIndexType> (matrix, k, itemsIndicies);
    }
#endif

}
