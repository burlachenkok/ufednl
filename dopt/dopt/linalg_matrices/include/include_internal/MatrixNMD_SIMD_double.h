#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/system/include/FloatUtils.h"

#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <assert.h>
#include <stddef.h>

namespace dopt
{
    template<>
    inline bool MatrixNMD<dopt::VectorNDRaw_d>::hasSIMDSupport() {
        return true;
    }
    template<>
    inline bool MatrixNMD<dopt::VectorNDStd_d>::hasSIMDSupport() {
        return true;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn MatrixNMD<dopt::VectorNDRaw_d>::matrixVectorMultiply(const MatrixNMD<dopt::VectorNDRaw_d>&m,
                                                                                                                    const MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn& x)
    {
        assert(m.columns() == x.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t r = m.rows();
        size_t c = m.columns();

        MatrixColumn res(r);
        MatrixColumn::TElementType * restrict_ext res_out = res.data();

        // bit-trick to remove remainder. Works only when kVecBatchSize is power of two.
        size_t packed_items_in_column = dopt::roundToNearestMultipleDown<kVecBatchSize>(r);

        VecType vec_aij = VecType();
        VecType vec_res = VecType();

        // Iterate through columns
        constexpr size_t kColumnsToProcess = 4;
        
        size_t j = 0;
        size_t cBound =  dopt::roundToNearestMultipleDown<kColumnsToProcess>(c);
        size_t myLDA = m.LDA;

        size_t jColumnFlatIndex = 0;
        const MatrixColumn::TElementType* restrict_ext matrixByCols = m.matrixByCols.dataConst();

        for (j = 0; j < cBound; j += kColumnsToProcess, jColumnFlatIndex += kColumnsToProcess * myLDA)
        {
            const MatrixColumn::TElementType  xj_1 = x.get(j);
            const MatrixColumn::TElementType* aij_1 = matrixByCols + jColumnFlatIndex /* + 0 * myLDA*/;

            const MatrixColumn::TElementType  xj_2 = x.get(j + 1);
            const MatrixColumn::TElementType* aij_2 = matrixByCols + jColumnFlatIndex + 1 * myLDA;

            const MatrixColumn::TElementType  xj_3 = x.get(j + 2);
            const MatrixColumn::TElementType* aij_3 = matrixByCols + jColumnFlatIndex + 2 * myLDA;

            const MatrixColumn::TElementType  xj_4 = x.get(j + 3);
            const MatrixColumn::TElementType* aij_4 = matrixByCols + jColumnFlatIndex + 3 * myLDA;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_aij.load_a(&(aij_1[i]));
                vec_res = ::mul_add(vec_aij, xj_1, vec_res);                

                vec_aij.load_a(&(aij_2[i]));
                vec_res = ::mul_add(vec_aij, xj_2, vec_res);

                vec_aij.load_a(&(aij_3[i]));
                vec_res = ::mul_add(vec_aij, xj_3, vec_res);

                vec_aij.load_a(&(aij_4[i]));
                vec_res = ::mul_add(vec_aij, xj_4, vec_res);
#else
                vec_aij.load_a(&(aij_1[i]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_2[i]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_3[i]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_4[i]));
                vec_aij *= xj_4;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }
            
#if SUPPORT_CPU_LOAD_STORE_PART
            {
                assert(j == cBound);

                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;
                
                assert(rest < kVecBatchSize);
                
                vec_res.load_partial(int(rest), &res_out[ii]);

                vec_aij.load_partial(int(rest), &(aij_1[ii]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_2[ii]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_3[ii]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_4[ii]));
                vec_aij *= xj_4;
                vec_res += vec_aij;

                vec_res.store_partial(int(rest), &res_out[ii]);
            }
#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij_1[i]) * (xj_1);
                res_out[i] += (aij_2[i]) * (xj_2);
                res_out[i] += (aij_3[i]) * (xj_3);
                res_out[i] += (aij_4[i]) * (xj_4);
            }
#endif
            
        }

        for (;j < c; ++j, jColumnFlatIndex += myLDA)
        {
            const MatrixColumn::TElementType  xj = x.get(j);
            const MatrixColumn::TElementType* aij = matrixByCols + jColumnFlatIndex /* + 0 * myLDA */;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);
                vec_aij.load_a(&(aij[i]));
                
#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_res = ::mul_add(vec_aij, xj, vec_res);
#else
                vec_aij *= xj;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }
            
#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);
                vec_aij.load_partial(int(rest), &(aij[ii]));
                vec_aij *= xj;
                vec_res += vec_aij;
                vec_res.store_partial(int(rest), &res_out[ii]);
            }            
            
#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij[i]) * (xj);
            }
#endif
        }
        return res;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn MatrixNMD<dopt::VectorNDStd_d>::matrixVectorMultiply(const MatrixNMD<dopt::VectorNDStd_d>& m,
                                                                                                             const MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn& x)
    {
        assert(m.columns() == x.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t r = m.rows();
        size_t c = m.columns();

        MatrixColumn res(r);
        MatrixColumn::TElementType* restrict_ext res_out = res.data();

        size_t packed_items_in_column = dopt::roundToNearestMultipleDown<kVecBatchSize>(r);
        
        // bit-trick to remove remainder. Works only when kVecBatchSize is power of two.
        //size_t packed_items_in_column = r & (~kVecBatchSize + 1);

        VecType vec_aij = VecType();
        VecType vec_res = VecType();

        // Iterate through columns
        constexpr size_t kColumnsToProcess = 4;
        size_t j = 0;
        size_t cBound = dopt::roundToNearestMultipleDown<kColumnsToProcess>(c);

        size_t myLDA = m.LDA;

        size_t jColumnFlatIndex = 0;
        const MatrixColumn::TElementType* restrict_ext matrixByCols = m.matrixByCols.dataConst();

        for (j = 0; j < cBound; j += kColumnsToProcess, jColumnFlatIndex += kColumnsToProcess * myLDA)
        {
            const MatrixColumn::TElementType  xj_1 = x.get(j);
            const MatrixColumn::TElementType* aij_1 = matrixByCols + jColumnFlatIndex /* + 0 * myLDA*/;

            const MatrixColumn::TElementType  xj_2 = x.get(j + 1);
            const MatrixColumn::TElementType* aij_2 = matrixByCols + jColumnFlatIndex + 1 * myLDA;

            const MatrixColumn::TElementType  xj_3 = x.get(j + 2);
            const MatrixColumn::TElementType* aij_3 = matrixByCols + jColumnFlatIndex + 2 * myLDA;

            const MatrixColumn::TElementType  xj_4 = x.get(j + 3);
            const MatrixColumn::TElementType* aij_4 = matrixByCols + jColumnFlatIndex + 3 * myLDA;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd

                vec_aij.load_a(&(aij_1[i]));
                vec_res = ::mul_add(vec_aij, xj_1, vec_res);

                vec_aij.load_a(&(aij_2[i]));
                vec_res = ::mul_add(vec_aij, xj_2, vec_res);

                vec_aij.load_a(&(aij_3[i]));
                vec_res = ::mul_add(vec_aij, xj_3, vec_res);

                vec_aij.load_a(&(aij_4[i]));
                vec_res = ::mul_add(vec_aij, xj_4, vec_res);
#else
                vec_aij.load_a(&(aij_1[i]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_2[i]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_3[i]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_4[i]));
                vec_aij *= xj_4;
                vec_res += vec_aij;
#endif
                vec_res.store_a(&res_out[i]);
            }

            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij_1[i]) * (xj_1);
                res_out[i] += (aij_2[i]) * (xj_2);
                res_out[i] += (aij_3[i]) * (xj_3);
                res_out[i] += (aij_4[i]) * (xj_4);
            }
        }

        for (; j < c; ++j, jColumnFlatIndex += myLDA)
        {
            const MatrixColumn::TElementType  xj = x.get(j);
            const MatrixColumn::TElementType* aij = matrixByCols + jColumnFlatIndex /* + 0 * myLDA */;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);
                vec_aij.load_a(&(aij[i]));

#if SUPPORT_CPU_FMA_EXT
                vec_res = ::mul_add(vec_aij, xj, vec_res);
#else
                vec_aij *= xj;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);
                vec_aij.load_partial(int(rest), &(aij[ii]));
                vec_aij *= xj;
                vec_res += vec_aij;
                vec_res.store_partial(int(rest), &res_out[ii]);
            }

#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij[i]) * (xj);
            }
#endif

        }
        return res;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn MatrixNMD<dopt::VectorNDRaw_d>::matrixVectorMultiply(const MatrixNMD<dopt::VectorNDRaw_d>& m,
                                                                                                             const MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn& x, 
                                                                                                             typename MatrixColumn::TElementType beta, 
                                                                                                             const MatrixColumn& v)
    {
        assert(m.columns() == x.size());
        assert(m.rows() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t r = m.rows();
        size_t c = m.columns();

        MatrixColumn res = v * beta;
        
        MatrixColumn::TElementType* restrict_ext res_out = res.data();

        size_t packed_items_in_column = dopt::roundToNearestMultipleDown<kVecBatchSize>(r);
        VecType vec_aij = VecType();
        VecType vec_res = VecType();

        // Iterate through columns
        constexpr size_t kColumnsToProcess = 4;
        size_t j = 0;
        size_t cBound =  dopt::roundToNearestMultipleDown<kColumnsToProcess>(c);
        size_t myLDA = m.LDA;

        size_t jColumnFlatIndex = 0;
        const MatrixColumn::TElementType* restrict_ext matrixByCols = m.matrixByCols.dataConst();

        for (j = 0; j < cBound; j += kColumnsToProcess, jColumnFlatIndex += kColumnsToProcess * myLDA)
        {
            const MatrixColumn::TElementType  xj_1 = x.get(j);
            const MatrixColumn::TElementType* aij_1 = matrixByCols + jColumnFlatIndex /* + 0 * myLDA*/;

            const MatrixColumn::TElementType  xj_2 = x.get(j + 1);
            const MatrixColumn::TElementType* aij_2 = matrixByCols + jColumnFlatIndex + 1 * myLDA;

            const MatrixColumn::TElementType  xj_3 = x.get(j + 2);
            const MatrixColumn::TElementType* aij_3 = matrixByCols + jColumnFlatIndex + 2 * myLDA;

            const MatrixColumn::TElementType  xj_4 = x.get(j + 3);
            const MatrixColumn::TElementType* aij_4 = matrixByCols + jColumnFlatIndex + 3 * myLDA;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_aij.load_a(&(aij_1[i]));
                vec_res = ::mul_add(vec_aij, xj_1, vec_res);

                vec_aij.load_a(&(aij_2[i]));
                vec_res = ::mul_add(vec_aij, xj_2, vec_res);

                vec_aij.load_a(&(aij_3[i]));
                vec_res = ::mul_add(vec_aij, xj_3, vec_res);

                vec_aij.load_a(&(aij_4[i]));
                vec_res = ::mul_add(vec_aij, xj_4, vec_res);
#else
                vec_aij.load_a(&(aij_1[i]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_2[i]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_3[i]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_4[i]));
                vec_aij *= xj_4;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);

                vec_aij.load_partial(int(rest), &(aij_1[ii]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_2[ii]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_3[ii]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_4[ii]));
                vec_aij *= xj_4;
                vec_res += vec_aij;

                vec_res.store_partial(int(rest), &res_out[ii]);
            }
#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij_1[i]) * (xj_1);
                res_out[i] += (aij_2[i]) * (xj_2);
                res_out[i] += (aij_3[i]) * (xj_3);
                res_out[i] += (aij_4[i]) * (xj_4);
            }
#endif
            
        }

        for (; j < c; ++j, jColumnFlatIndex += myLDA)
        {
            const MatrixColumn::TElementType  xj = x.get(j);
            const MatrixColumn::TElementType* aij = matrixByCols + jColumnFlatIndex /* + 0 * myLDA */;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);
                vec_aij.load_a(&(aij[i]));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_res = ::mul_add(vec_aij, xj, vec_res);
#else
                vec_aij *= xj;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);
                vec_aij.load_partial(int(rest), &(aij[ii]));
                vec_aij *= xj;
                vec_res += vec_aij;
                vec_res.store_partial(int(rest), &res_out[ii]);
            }

#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij[i]) * (xj);
            }
#endif
            
        }
        return res;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn MatrixNMD<dopt::VectorNDStd_d>::matrixVectorMultiply(const MatrixNMD<dopt::VectorNDStd_d>& m, 
                                                                                                             const MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn& x,
                                                                                                             typename MatrixColumn::TElementType beta, 
                                                                                                             const MatrixColumn& v)
    {
        assert(m.columns() == x.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t r = m.rows();
        size_t c = m.columns();

        MatrixColumn res = v * beta;
        MatrixColumn::TElementType* restrict_ext res_out = res.data();

        // bit-trick to remove remainder. Works only when kVecBatchSize is power of two.
        size_t packed_items_in_column = dopt::roundToNearestMultipleDown<kVecBatchSize>(r);

        VecType vec_aij = VecType();
        VecType vec_res = VecType();

        // Iterate through columns
        constexpr size_t kColumnsToProcess = 4;
        size_t j = 0;
        size_t cBound =  dopt::roundToNearestMultipleDown<kColumnsToProcess>(c);
        size_t myLDA = m.LDA;

        size_t jColumnFlatIndex = 0;
        const MatrixColumn::TElementType* restrict_ext matrixByCols = m.matrixByCols.dataConst();

        for (j = 0; j < cBound; j += kColumnsToProcess, jColumnFlatIndex += kColumnsToProcess * myLDA)
        {
            const MatrixColumn::TElementType  xj_1 = x.get(j);
            const MatrixColumn::TElementType* aij_1 = matrixByCols + jColumnFlatIndex /* + 0 * myLDA*/;

            const MatrixColumn::TElementType  xj_2 = x.get(j + 1);
            const MatrixColumn::TElementType* aij_2 = matrixByCols + jColumnFlatIndex + 1 * myLDA;

            const MatrixColumn::TElementType  xj_3 = x.get(j + 2);
            const MatrixColumn::TElementType* aij_3 = matrixByCols + jColumnFlatIndex + 2 * myLDA;

            const MatrixColumn::TElementType  xj_4 = x.get(j + 3);
            const MatrixColumn::TElementType* aij_4 = matrixByCols + jColumnFlatIndex + 3 * myLDA;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_aij.load_a(&(aij_1[i]));
                vec_res = ::mul_add(vec_aij, xj_1, vec_res);

                vec_aij.load_a(&(aij_2[i]));
                vec_res = ::mul_add(vec_aij, xj_2, vec_res);

                vec_aij.load_a(&(aij_3[i]));
                vec_res = ::mul_add(vec_aij, xj_3, vec_res);

                vec_aij.load_a(&(aij_4[i]));
                vec_res = ::mul_add(vec_aij, xj_4, vec_res);
#else
                vec_aij.load_a(&(aij_1[i]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_2[i]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_3[i]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_a(&(aij_4[i]));
                vec_aij *= xj_4;
                vec_res += vec_aij;
#endif
                vec_res.store_a(&res_out[i]);
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);

                vec_aij.load_partial(int(rest), &(aij_1[ii]));
                vec_aij *= xj_1;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_2[ii]));
                vec_aij *= xj_2;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_3[ii]));
                vec_aij *= xj_3;
                vec_res += vec_aij;

                vec_aij.load_partial(int(rest), &(aij_4[ii]));
                vec_aij *= xj_4;
                vec_res += vec_aij;

                vec_res.store_partial(int(rest), &res_out[ii]);
            }
#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij_1[i]) * (xj_1);
                res_out[i] += (aij_2[i]) * (xj_2);
                res_out[i] += (aij_3[i]) * (xj_3);
                res_out[i] += (aij_4[i]) * (xj_4);
            }
#endif

        }

        for (; j < c; ++j, jColumnFlatIndex += myLDA)
        {
            const MatrixColumn::TElementType  xj = x.get(j);
            const MatrixColumn::TElementType* aij = matrixByCols + jColumnFlatIndex /* + 0 * myLDA */;

            for (size_t i = 0; i < packed_items_in_column; i += kVecBatchSize)
            {
                vec_res.load_a(&res_out[i]);
                vec_aij.load_a(&(aij[i]));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                vec_res = ::mul_add(vec_aij, xj, vec_res);
#else
                vec_aij *= xj;
                vec_res += vec_aij;
#endif

                vec_res.store_a(&res_out[i]);
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t ii = packed_items_in_column;
                size_t rest = r - packed_items_in_column;

                assert(rest < kVecBatchSize);

                vec_res.load_partial(int(rest), &res_out[ii]);
                vec_aij.load_partial(int(rest), &(aij[ii]));
                vec_aij *= xj;
                vec_res += vec_aij;
                vec_res.store_partial(int(rest), &res_out[ii]);
            }

#else
            for (size_t i = packed_items_in_column; i < r; ++i)
            {
                res_out[i] += (aij[i]) * (xj);
            }
#endif

        }
        return res;
    }

//=============================================================================================================//

#if SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits

    inline void transpose4x4_SSE_variant_a(double* restrict_ext B, const size_t ldb, const double* restrict_ext A, const size_t lda)
    {
        // To support this code AVX_256_bits CPU extension should be supported

        __m256d row0 = _mm256_loadu_pd(&A[0 * lda]);
        __m256d row1 = _mm256_loadu_pd(&A[1 * lda]);
        __m256d row2 = _mm256_loadu_pd(&A[2 * lda]);
        __m256d row3 = _mm256_loadu_pd(&A[3 * lda]);

        {
            // https://stackoverflow.com/questions/36167517/m256d-transpose4-equivalent

            __m256d tmp3, tmp2, tmp1, tmp0;

            tmp0 = _mm256_shuffle_pd((row0), (row1), 0x0);
            tmp2 = _mm256_shuffle_pd((row0), (row1), 0xF);
            tmp1 = _mm256_shuffle_pd((row2), (row3), 0x0);
            tmp3 = _mm256_shuffle_pd((row2), (row3), 0xF);

            (row0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            (row1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            (row2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
            (row3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
        }

        _mm256_storeu_pd(&B[0 * ldb], row0);
        _mm256_storeu_pd(&B[1 * ldb], row1);
        _mm256_storeu_pd(&B[2 * ldb], row2);
        _mm256_storeu_pd(&B[3 * ldb], row3);
    }
#endif

    template<>
    inline void MatrixNMD<dopt::VectorNDRaw_d>::internal_execute_transpose_blockwise(double* restrict_ext B, const size_t ldb,
                                                                                     const double* restrict_ext A, const size_t lda,
                                                                                     const size_t r, const size_t c)
    {
        // B matrix contains "B" by columns. Stride between columns is LDB
        // A matrix contains "A" by columns. Stride between columns is LDA
        // A has input has r rows and c columns
        constexpr size_t i_tile_s_level = 16;
        constexpr size_t j_tile_s_level = 16;
        
        static_assert(i_tile_s_level % 4 == 0);
        static_assert(j_tile_s_level % 4 == 0);

        // Iterating through input matrix A
        for (size_t is = 0; is < r; is += i_tile_s_level)
        {
            // Next (is + i_tile_s_level) does bound current matrix in ROW dimension
            if (is + i_tile_s_level <= r)
            {
                // Bound for next tile level is valid
                size_t it_bound = is + i_tile_s_level;

                for (size_t js = 0; js < c; js += j_tile_s_level)
                {
                    if (js + j_tile_s_level <= c)
                    {
                        // Case which we can optimize
                        size_t jt_bound = js + j_tile_s_level;

#if DOPT_INCLUDE_VECTORIZED_CPU_TRANSPOSE_MATS && (SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits)
                        for (size_t it = is; it < it_bound; it += 4)
                        {
                            for (size_t jt = js; jt < jt_bound; jt += 4)
                            {
                                transpose4x4_SSE_variant_a(&B[jt + it * ldb], ldb, &A[it + jt * lda], lda);
                            }
                        }
#else
                        for (size_t it = is; it < it_bound; it++)
                            for (size_t jt = js; jt < jt_bound; jt++)
                                B[jt + it * ldb] = A[it + jt * lda];

#endif
                        
                    }
                    else
                    {
                        // Case where we we can optimize
                        size_t jt_bound = c;
                        for (size_t it = is; it < it_bound; it++)
                            for (size_t jt = js; jt < jt_bound; jt++)
                                B[jt + it * ldb] = A[it + jt * lda];
                    }
                }
            }
            // Next (is + i_tile_s_level) does not bound current matrix in ROW dimension
            else
            {
                size_t it_bound = r;

                for (size_t js = 0; js < c; js += j_tile_s_level)
                {
                    // Bound in column dimension
                    size_t jt_bound = minimum(js + j_tile_s_level, c);
                    
                    for (size_t it = is; it < it_bound; it++)
                        for (size_t jt = js; jt < jt_bound; jt++)
                            B[jt + it * ldb] = A[it + jt * lda];
                }
            }
        }

        return;
    }

    template<>
    inline void MatrixNMD<dopt::VectorNDStd_d>::internal_execute_transpose_blockwise(double* restrict_ext B, const size_t ldb,
                                                                                     const double* restrict_ext A, const size_t lda,
                                                                                     const size_t r, const size_t c)
    {
        // B matrix contains "B" by columns. Stride between columns is LDB
        // A matrix contains "A" by columns. Stride between columns is LDA
        // A has input has r rows and c columns
        constexpr size_t i_tile_s_level = 16;
        constexpr size_t j_tile_s_level = 16;

        static_assert(i_tile_s_level % 4 == 0);
        static_assert(j_tile_s_level % 4 == 0);

        // Iterating through input matrix A
        for (size_t is = 0; is < r; is += i_tile_s_level)
        {
            // Next (is + i_tile_s_level) does bound current matrix in ROW dimension
            if (is + i_tile_s_level <= r)
            {
                // Bound for next tile level is valid
                size_t it_bound = is + i_tile_s_level;

                for (size_t js = 0; js < c; js += j_tile_s_level)
                {
                    if (js + j_tile_s_level <= c)
                    {
                        // Case which we can optimize
                        size_t jt_bound = js + j_tile_s_level;

#if DOPT_INCLUDE_VECTORIZED_CPU_TRANSPOSE_MATS && (SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits)
                        for (size_t it = is; it < it_bound; it += 4)
                        {
                            for (size_t jt = js; jt < jt_bound; jt += 4)
                            {
                                transpose4x4_SSE_variant_a(&B[jt + it * ldb], ldb, &A[it + jt * lda], lda);
                            }
                        }
#else
                        for (size_t it = is; it < it_bound; it++)
                            for (size_t jt = js; jt < jt_bound; jt++)
                                B[jt + it * ldb] = A[it + jt * lda];

#endif

                    }
                    else
                    {
                        // Case where we we can optimize
                        size_t jt_bound = c;
                        for (size_t it = is; it < it_bound; it++)
                            for (size_t jt = js; jt < jt_bound; jt++)
                                B[jt + it * ldb] = A[it + jt * lda];
                    }
                }
            }
            // Next (is + i_tile_s_level) does not bound current matrix in ROW dimension
            else
            {
                size_t it_bound = r;

                for (size_t js = 0; js < c; js += j_tile_s_level)
                {
                    // Bound in column dimension
                    size_t jt_bound = minimum(js + j_tile_s_level, c);

                    for (size_t it = is; it < it_bound; it++)
                        for (size_t jt = js; jt < jt_bound; jt++)
                            B[jt + it * ldb] = A[it + jt * lda];
                }
            }
        }

        return;
    }


    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d> MatrixNMD<dopt::VectorNDRaw_d>::getTranspose() const
    {
        size_t r = rows();
        size_t c = columns();
        MatrixNMD res(c, r);
        MatrixNMD<dopt::VectorNDRaw_d>::internal_execute_transpose_blockwise(res.matrixByCols.data(), res.LDA, matrixByCols.dataConst(), LDA, r, c);
        return res;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDStd_d> MatrixNMD<dopt::VectorNDStd_d>::getTranspose() const
    {
        size_t r = rows();
        size_t c = columns();
        MatrixNMD res(c, r);
        MatrixNMD<dopt::VectorNDRaw_d>::internal_execute_transpose_blockwise(res.matrixByCols.data(), res.LDA, matrixByCols.dataConst(), LDA, r, c);
        return res;
    }

    template<>
    inline MatrixNMD<dopt::VectorNDStd_d>::TElementType MatrixNMD<dopt::VectorNDStd_d>::frobeniusNormSquareForSymmetricMatrixFromUpPart() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;

        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        const TElementType* restrict_ext thisData = matrixByCols.dataConst();
        size_t d = rows();

        TElementType resDiagonal = TElementType();
        TElementType resRest = TElementType();
        
        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        for (size_t j = 0; j < d; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);

            // compute first j elements
            size_t sz = j;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    cvec[k] += ::square(avec[k]);
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;

                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load_a(thisData + items + read_pos);
                    resRest += ::horizontal_add(::square(avec));

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                assert(resLen <= kVecBatchSize);

                avec.load_partial(int(resLen), thisData + items + read_pos);
                resRest += ::horizontal_add(::square(avec));
            }
#else
            for (; i < sz; ++i)
            {
                TElementType value = thisData[read_pos + i];
                resRest += value * value;
            }
            
#endif

            TElementType diagItem = thisData[read_pos + j];
            resDiagonal += diagItem * diagItem;
        }

#if 1
        TElementType resOutDiagonal = TElementType();
        TElementType res[kUnrollFactor] = {};

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_add(cvec[k]);
                resOutDiagonal += res[k];
            }
        }
#else
        for (size_t k = 1; k < kUnrollFactor; ++k)
            cvec[0] += cvec[k];
        TElementType resOutDiagonal = ::horizontal_add(cvec[0]);
#endif

        resOutDiagonal += resRest;

        return (resOutDiagonal + resOutDiagonal + resDiagonal);
    }

    template<>
    inline MatrixNMD<dopt::VectorNDStd_d>::TElementType MatrixNMD<dopt::VectorNDStd_d>::frobeniusNormForSymmetricMatrixFromUpPart() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;

        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        const TElementType* restrict_ext thisData = matrixByCols.dataConst();
        size_t d = rows();

        TElementType resDiagonal = TElementType();
        TElementType resRest = TElementType();
        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        for (size_t j = 0; j < d; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);

            // compute first j elements
            size_t sz = j;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    cvec[k] += ::square(avec[k]);
                }
            }
            

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load_a(thisData + items + read_pos);
                    resRest += ::horizontal_add(::square(avec));

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                assert(resLen <= kVecBatchSize);
                avec.load_partial(int(resLen), thisData + items + read_pos);
                resRest += ::horizontal_add(::square(avec));
            }
#else
            for (; i < sz; ++i)
            {
                TElementType value = thisData[read_pos + i];
                resRest += value * value;
            }

#endif

            TElementType diagItem = thisData[read_pos + j];
            resDiagonal += diagItem * diagItem;
        }

#if 1
        TElementType resOutDiagonal = TElementType();
        TElementType res[kUnrollFactor] = {};

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_add(cvec[k]);
                resOutDiagonal += res[k];
            }
        }
#else
        for (size_t k = 1; k < kUnrollFactor; ++k)
            cvec[0] += cvec[k];
        TElementType resOutDiagonal = ::horizontal_add(cvec[0]);
#endif

        resOutDiagonal += resRest;

        return ::sqrt(resOutDiagonal +
                          resOutDiagonal +
                          resDiagonal);
    }

    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d>::TElementType MatrixNMD<dopt::VectorNDRaw_d>::frobeniusNormSquareForSymmetricMatrixFromUpPart() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;

        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        const TElementType* restrict_ext thisData = matrixByCols.dataConst();
        size_t d = rows();

        TElementType resDiagonal = TElementType();
        TElementType resRest = TElementType();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        for (size_t j = 0; j < d; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);

            // compute first j elements
            size_t sz = j;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    cvec[k] += ::square(avec[k]);
                }
            }


#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load_a(thisData + items + read_pos);
                    resRest += ::horizontal_add(::square(avec));

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                assert(resLen <= kVecBatchSize);

                avec.load_partial(int(resLen), thisData + items + read_pos);
                resRest += ::horizontal_add(::square(avec));
            }
#else
            for (; i < sz; ++i)
            {
                TElementType value = thisData[read_pos + i];
                resRest += value * value;
            }

#endif

            TElementType diagItem = thisData[read_pos + j];
            resDiagonal += diagItem * diagItem;
        }

#if 1
        TElementType resOutDiagonal = TElementType();
        TElementType res[kUnrollFactor] = {};

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_add(cvec[k]);
                resOutDiagonal += res[k];
            }
        }
#else
        for (size_t k = 1; k < kUnrollFactor; ++k)
            cvec[0] += cvec[k];
        TElementType resOutDiagonal = ::horizontal_add(cvec[0]);
#endif
        resOutDiagonal += resRest;

        return (resOutDiagonal + resOutDiagonal + resDiagonal);
    }


    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d>::TElementType MatrixNMD<dopt::VectorNDRaw_d>::frobeniusNormForSymmetricMatrixFromUpPart() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;

        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        const TElementType* restrict_ext thisData = matrixByCols.dataConst();
        size_t d = rows();

        TElementType resDiagonal = TElementType();
        TElementType resRest = TElementType();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        for (size_t j = 0; j < d; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);

            // compute first j elements
            size_t sz = j;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    cvec[k] += ::square(avec[k]);
                }
            }

            assert(i == items);

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load_a(thisData + items + read_pos);
                    resRest += ::horizontal_add(::square(avec));

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                assert(resLen <= kVecBatchSize);

                avec.load_partial(int(resLen), thisData + items + read_pos);
                resRest += ::horizontal_add(::square(avec));
            }
#else
            for (; i < sz; ++i)
            {
                TElementType value = thisData[read_pos + i];
                resRest += value * value;
            }

#endif

            TElementType diagItem = thisData[read_pos + j];
            resDiagonal += diagItem * diagItem;
        }

#if 1
        TElementType resOutDiagonal = TElementType();
        TElementType res[kUnrollFactor] = {};

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_add(cvec[k]);
                resOutDiagonal += res[k];
            }
        }
#else
        for (size_t k = 1; k < kUnrollFactor; ++k)
            cvec[0] += cvec[k];
        TElementType resOutDiagonal = ::horizontal_add(cvec[0]);
#endif
        resOutDiagonal += resRest;

        return ::sqrt(resOutDiagonal +
                          resOutDiagonal +
                          resDiagonal);
    }

    template<>
    inline MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn MatrixNMD<dopt::VectorNDRaw_d>::matrixVectorMultiplyWithPreTranspose(const MatrixNMD<dopt::VectorNDRaw_d>& mTranspose,
                                                                                                                             const MatrixNMD<dopt::VectorNDRaw_d>::MatrixColumn& x)
    {
        assert(mTranspose.rows() == x.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        // Result shape
        size_t r = mTranspose.columns();
        size_t c = mTranspose.rows();
        MatrixColumn res(r);
        MatrixColumn::TElementType* restrict_ext res_out = res.data();

        VecType vec_aij_1 = VecType();
        VecType vec_aij_2 = VecType();
        VecType vec_aij_3 = VecType();
        VecType vec_aij_4 = VecType();
        VecType vec_x = VecType();

        // Iterate through rows
        constexpr size_t kRowsToProcess = 4;                                                // number of rows to process in one iteration
        size_t packed_items_in_rows = dopt::roundToNearestMultipleDown<kRowsToProcess>(r);  // number of packed rows in matrix

        // Bound for columns processing 
        size_t cBound = dopt::roundToNearestMultipleDown<kVecBatchSize>(c);                 // bound for column index to process in SIMD way

        const MatrixColumn::TElementType* restrict_ext matrixByRows = mTranspose.matrixByCols.dataConst();
        size_t myLDA = mTranspose.LDA;

        size_t i = 0;

        for (i = 0; i < packed_items_in_rows; i += kRowsToProcess,
                                              matrixByRows += myLDA * kRowsToProcess)
        {
            // Rows of matrix "A"
            const MatrixColumn::TElementType* restrict_ext aij_1 = matrixByRows /* + 0 * myLDA*/;
            const MatrixColumn::TElementType* restrict_ext aij_2 = matrixByRows + 1 * myLDA;
            const MatrixColumn::TElementType* restrict_ext aij_3 = matrixByRows + 2 * myLDA;
            const MatrixColumn::TElementType* restrict_ext aij_4 = matrixByRows + 3 * myLDA;

            const MatrixColumn::TElementType* restrict_ext XRaw = x.dataConst();

            VecType accum_vec_aij_1 = VecType(0);
            VecType accum_vec_aij_2 = VecType(0);
            VecType accum_vec_aij_3 = VecType(0);
            VecType accum_vec_aij_4 = VecType(0);

            size_t j = 0;
            for (j = 0; j < cBound; j += kVecBatchSize,
                                    aij_1 += kVecBatchSize,
                                    aij_2 += kVecBatchSize,
                                    aij_3 += kVecBatchSize,
                                    aij_4 += kVecBatchSize,
                                    XRaw += kVecBatchSize)
            {
                vec_aij_1.load_a(aij_1);
                vec_aij_2.load_a(aij_2);
                vec_aij_3.load_a(aij_3);
                vec_aij_4.load_a(aij_4);
                vec_x.load_a(XRaw);

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                accum_vec_aij_1 = ::mul_add(vec_aij_1, vec_x, accum_vec_aij_1);
                accum_vec_aij_2 = ::mul_add(vec_aij_2, vec_x, accum_vec_aij_2);
                accum_vec_aij_3 = ::mul_add(vec_aij_3, vec_x, accum_vec_aij_3);
                accum_vec_aij_4 = ::mul_add(vec_aij_4, vec_x, accum_vec_aij_4);
#else
                accum_vec_aij_1 += vec_aij_1 * vec_x;
                accum_vec_aij_2 += vec_aij_2 * vec_x;
                accum_vec_aij_3 += vec_aij_3 * vec_x;
                accum_vec_aij_4 += vec_aij_4 * vec_x;
#endif
            }

            assert(j == cBound);

#if SUPPORT_CPU_LOAD_STORE_PART
            
            {

                size_t rest = c - cBound;
                assert(rest < kVecBatchSize);

                vec_aij_1.load_partial(int(rest), aij_1);
                vec_aij_2.load_partial(int(rest), aij_2);
                vec_aij_3.load_partial(int(rest), aij_3);
                vec_aij_4.load_partial(int(rest), aij_4);
                vec_x.load_partial(int(rest), XRaw);

                accum_vec_aij_1 += vec_aij_1 * vec_x;
                accum_vec_aij_2 += vec_aij_2 * vec_x;
                accum_vec_aij_3 += vec_aij_3 * vec_x;
                accum_vec_aij_4 += vec_aij_4 * vec_x;
            }
            
            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            MatrixColumn::TElementType accum_scalar_aij_2 = ::horizontal_add(accum_vec_aij_2);
            MatrixColumn::TElementType accum_scalar_aij_3 = ::horizontal_add(accum_vec_aij_3);
            MatrixColumn::TElementType accum_scalar_aij_4 = ::horizontal_add(accum_vec_aij_4);
#else

            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            MatrixColumn::TElementType accum_scalar_aij_2 = ::horizontal_add(accum_vec_aij_2);
            MatrixColumn::TElementType accum_scalar_aij_3 = ::horizontal_add(accum_vec_aij_3);
            MatrixColumn::TElementType accum_scalar_aij_4 = ::horizontal_add(accum_vec_aij_4);

            for (; j < c; ++j, ++aij_1, ++aij_2, ++aij_3, ++aij_4, ++XRaw)
            {
                const MatrixColumn::TElementType xMultiply = *XRaw;
                accum_scalar_aij_1 += (*aij_1) * xMultiply;
                accum_scalar_aij_2 += (*aij_2) * xMultiply;
                accum_scalar_aij_3 += (*aij_3) * xMultiply;
                accum_scalar_aij_4 += (*aij_4) * xMultiply;
            }
#endif

            res_out[i + 0] = accum_scalar_aij_1;
            res_out[i + 1] = accum_scalar_aij_2;
            res_out[i + 2] = accum_scalar_aij_3;
            res_out[i + 3] = accum_scalar_aij_4;
        }

        for (; i < r; ++i, matrixByRows += myLDA)
        {
            const MatrixColumn::TElementType* restrict_ext aij_1 = matrixByRows /* + 0 * myLDA*/;
            const MatrixColumn::TElementType* restrict_ext XRaw = x.dataConst();

            VecType accum_vec_aij_1 = VecType(0);
            size_t j = 0;

            for (j = 0; j < cBound; j += kVecBatchSize,
                                    aij_1 += kVecBatchSize,
                                    XRaw += kVecBatchSize)
            {
                vec_aij_1.load_a(aij_1);
                vec_x.load_a(XRaw);
                
#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                accum_vec_aij_1 = ::mul_add(vec_aij_1, vec_x, accum_vec_aij_1);
#else
                accum_vec_aij_1 += vec_aij_1 * vec_x;
#endif
                
            }

            assert(j == cBound);

#if SUPPORT_CPU_LOAD_STORE_PART
            size_t rest = c - cBound;
            assert(rest < kVecBatchSize);

            vec_aij_1.load_partial(int(rest), aij_1);
            vec_x.load_partial(int(rest), XRaw);
            accum_vec_aij_1 += vec_aij_1 * vec_x;

            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            
#else

            
            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            for (; j < c; ++j, ++aij_1, ++XRaw)
            {
                const MatrixColumn::TElementType xMultiply = *XRaw;
                accum_scalar_aij_1 += (*aij_1) * xMultiply;
            }
#endif

            res_out[i] = accum_scalar_aij_1;
        }

        return res;
    }


    template<>
    inline MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn MatrixNMD<dopt::VectorNDStd_d>::matrixVectorMultiplyWithPreTranspose(const MatrixNMD<dopt::VectorNDStd_d>& mTranspose,
                                                                                                                             const MatrixNMD<dopt::VectorNDStd_d>::MatrixColumn& x)
    {
        assert(mTranspose.rows() == x.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        // Result shape
        size_t r = mTranspose.columns();
        size_t c = mTranspose.rows();
        MatrixColumn res(r);
        MatrixColumn::TElementType* restrict_ext res_out = res.data();

        VecType vec_aij_1 = VecType();
        VecType vec_aij_2 = VecType();
        VecType vec_aij_3 = VecType();
        VecType vec_aij_4 = VecType();
        VecType vec_x = VecType();

        // Iterate through rows
        constexpr size_t kRowsToProcess = 4;                     // number of rows to process in one iteration
        size_t packed_items_in_rows = dopt::roundToNearestMultipleDown<kRowsToProcess>(r);  // number of packed rows in matrix

        // Bound for columns processing 
        size_t cBound = dopt::roundToNearestMultipleDown<kVecBatchSize>(c);                 // bound for column index to process in SIMD way

        const MatrixColumn::TElementType* restrict_ext matrixByRows = mTranspose.matrixByCols.dataConst();
        size_t myLDA = mTranspose.LDA;

        size_t i = 0;

        for (i = 0; i < packed_items_in_rows; i += kRowsToProcess,
            matrixByRows += myLDA * kRowsToProcess)
        {
            // Rows of matrix "A"
            const MatrixColumn::TElementType* restrict_ext aij_1 = matrixByRows /* + 0 * myLDA*/;
            const MatrixColumn::TElementType* restrict_ext aij_2 = matrixByRows + 1 * myLDA;
            const MatrixColumn::TElementType* restrict_ext aij_3 = matrixByRows + 2 * myLDA;
            const MatrixColumn::TElementType* restrict_ext aij_4 = matrixByRows + 3 * myLDA;

            const MatrixColumn::TElementType* restrict_ext XRaw = x.dataConst();

            VecType accum_vec_aij_1 = VecType(0);
            VecType accum_vec_aij_2 = VecType(0);
            VecType accum_vec_aij_3 = VecType(0);
            VecType accum_vec_aij_4 = VecType(0);

            size_t j = 0;
            for (j = 0; j < cBound; j += kVecBatchSize,
                                    aij_1 += kVecBatchSize,
                                    aij_2 += kVecBatchSize,
                                    aij_3 += kVecBatchSize,
                                    aij_4 += kVecBatchSize,
                                    XRaw += kVecBatchSize)
            {
                vec_aij_1.load_a(aij_1);
                vec_aij_2.load_a(aij_2);
                vec_aij_3.load_a(aij_3);
                vec_aij_4.load_a(aij_4);
                vec_x.load_a(XRaw);

#if SUPPORT_CPU_FMA_EXT
                
                // (1st * 2nd) + 3rd
                accum_vec_aij_1 = ::mul_add(vec_aij_1, vec_x, accum_vec_aij_1);
                accum_vec_aij_2 = ::mul_add(vec_aij_2, vec_x, accum_vec_aij_2);
                accum_vec_aij_3 = ::mul_add(vec_aij_3, vec_x, accum_vec_aij_3);
                accum_vec_aij_4 = ::mul_add(vec_aij_4, vec_x, accum_vec_aij_4);

#else
                accum_vec_aij_1 += vec_aij_1 * vec_x;
                accum_vec_aij_2 += vec_aij_2 * vec_x;
                accum_vec_aij_3 += vec_aij_3 * vec_x;
                accum_vec_aij_4 += vec_aij_4 * vec_x;
#endif
            }

            assert(j == cBound);

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t rest = c - cBound;
                assert(rest < kVecBatchSize);
                
                vec_aij_1.load_partial(int(rest), aij_1);
                vec_aij_2.load_partial(int(rest), aij_2);
                vec_aij_3.load_partial(int(rest), aij_3);
                vec_aij_4.load_partial(int(rest), aij_4);
                vec_x.load_partial(int(rest), XRaw);

                accum_vec_aij_1 += vec_aij_1 * vec_x;
                accum_vec_aij_2 += vec_aij_2 * vec_x;
                accum_vec_aij_3 += vec_aij_3 * vec_x;
                accum_vec_aij_4 += vec_aij_4 * vec_x;
            }

            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            MatrixColumn::TElementType accum_scalar_aij_2 = ::horizontal_add(accum_vec_aij_2);
            MatrixColumn::TElementType accum_scalar_aij_3 = ::horizontal_add(accum_vec_aij_3);
            MatrixColumn::TElementType accum_scalar_aij_4 = ::horizontal_add(accum_vec_aij_4);
#else
            
            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            MatrixColumn::TElementType accum_scalar_aij_2 = ::horizontal_add(accum_vec_aij_2);
            MatrixColumn::TElementType accum_scalar_aij_3 = ::horizontal_add(accum_vec_aij_3);
            MatrixColumn::TElementType accum_scalar_aij_4 = ::horizontal_add(accum_vec_aij_4);

            for (; j < c; ++j, ++aij_1, ++aij_2, ++aij_3, ++aij_4, ++XRaw)
            {
                const MatrixColumn::TElementType xMultiply = *XRaw;
                accum_scalar_aij_1 += (*aij_1) * xMultiply;
                accum_scalar_aij_2 += (*aij_2) * xMultiply;
                accum_scalar_aij_3 += (*aij_3) * xMultiply;
                accum_scalar_aij_4 += (*aij_4) * xMultiply;
            }
#endif
            
            res_out[i + 0] = accum_scalar_aij_1;
            res_out[i + 1] = accum_scalar_aij_2;
            res_out[i + 2] = accum_scalar_aij_3;
            res_out[i + 3] = accum_scalar_aij_4;
        }

        for (; i < r; ++i, matrixByRows += myLDA)
        {
            const MatrixColumn::TElementType* restrict_ext aij_1 = matrixByRows /* + 0 * myLDA*/;
            const MatrixColumn::TElementType* restrict_ext XRaw = x.dataConst();

            VecType accum_vec_aij_1 = VecType(0);
            size_t j = 0;

            for (j = 0; j < cBound; j += kVecBatchSize,
                                    aij_1 += kVecBatchSize,
                                    XRaw += kVecBatchSize)
            {
                vec_aij_1.load_a(aij_1);
                vec_x.load_a(XRaw);
                
#if SUPPORT_CPU_FMA_EXT                
                accum_vec_aij_1 = ::mul_add(vec_aij_1, vec_x, accum_vec_aij_1);
#else
                accum_vec_aij_1 += vec_aij_1 * vec_x;
#endif
            }

            assert(j == cBound);

#if SUPPORT_CPU_LOAD_STORE_PART

            {
                size_t rest = c - cBound;
                assert(rest < kVecBatchSize);

                vec_aij_1.load_partial(int(rest), aij_1);
                vec_x.load_partial(int(rest), XRaw);
                
                accum_vec_aij_1 += vec_aij_1 * vec_x;
            }
            
            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
#else

            MatrixColumn::TElementType accum_scalar_aij_1 = ::horizontal_add(accum_vec_aij_1);
            for (; j < c; ++j, ++aij_1, ++XRaw)
            {
                const MatrixColumn::TElementType xMultiply = *XRaw;
                accum_scalar_aij_1 += (*aij_1) * xMultiply;
            }

#endif

            res_out[i] = accum_scalar_aij_1;
        }

        return res;
    }

    template<>
    inline void MatrixNMD<dopt::VectorNDRaw_d>::applyNaturalCompressor()
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();


        const FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, TElementType> maskItem = getFloatPointMask2RemoveMantissa<DOPT_ARCH_LITTLE_ENDIAN, TElementType>();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType mask = VecType(maskItem.real_value_repr);

        size_t myCols = columns_;
        size_t myRows = rows_;
        
        TElementType* restrict_ext thisData = matrixByCols.data();

        for (size_t j = 0; j < myCols; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);
            size_t sz = myRows;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k) {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    avec[k] &= mask;
                    avec[k].store_a(thisData + read_pos + (i + k * kVecBatchSize));
                }
            }

            for (;i < sz; ++i)
            {
                *reinterpret_cast<uint64_t*>(thisData + read_pos + i) &= maskItem.integer_value_repr;
            }
        }

        return;
    }

    template<>
    inline void MatrixNMD<dopt::VectorNDStd_d>::applyNaturalCompressor()
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        const FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, TElementType> maskItem = getFloatPointMask2RemoveMantissa<DOPT_ARCH_LITTLE_ENDIAN, TElementType>();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType mask = VecType(maskItem.real_value_repr);

        size_t myCols = columns_;
        size_t myRows = rows_;
        
        TElementType* restrict_ext thisData = matrixByCols.data();

        for (size_t j = 0; j < myCols; ++j)
        {
            size_t read_pos = this->template getFlattenIndexFromColumn</*i*/0>(j);
            size_t sz = myRows;
            size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

            size_t i = 0;
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k) {
                    avec[k].load_a(thisData + read_pos + (i + k * kVecBatchSize));
                    avec[k] &= mask;
                    avec[k].store_a(thisData + read_pos + (i + k * kVecBatchSize));
                }
            }

            for (; i < sz; ++i)
            {
                *reinterpret_cast<uint64_t*>(thisData + read_pos + i) &= maskItem.integer_value_repr;
            }
        }

        return;
    }
}


#endif
