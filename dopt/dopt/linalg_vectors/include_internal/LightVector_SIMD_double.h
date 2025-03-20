#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "dopt/linalg_vectors/include/LightVectorND.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/linalg_vectors/include/VectorND_Std.h"

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <limits>

#include <assert.h>
#include <math.h>
#include <stddef.h>

namespace dopt
{
    template <>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::addInPlaceVectorWithMultiple(double multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {};
        VecType bvec[kUnrollFactor] = {};
        VecType multiple_vec(multiple);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // 1st * 2nd + 3rd
                avec[k] = ::mul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] += (bvec[k] * multiple_vec);
#endif
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += (bvec * multiple_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] += (vData[i] * multiple);
        }

#endif

        return *this;
    }

    template <>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::subInPlaceVectorWithMultiple(double multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType multiple_vec(multiple);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));
                
#if SUPPORT_CPU_FMA_EXT
                // -(1st * 2nd) + 3rd
                avec[k] = ::nmul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] -= (bvec[k] * multiple_vec);
#endif
                
                
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec -= (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec -= (bvec * multiple_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] -= (vData[i] * multiple);
        }

#endif

        return *this;
    }
    
    template <>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::operator += (const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized


        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));
                avec[k] += bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += bvec;
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] += vData[i];
        }
#endif

        return *this;
    }

    template <>
    inline double LightVectorND<VectorNDRaw_d>::operator & (const LightVectorND<VectorNDRaw_d>& rhs) const
    {
        assert(size() == rhs.size());
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();
        
        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                bvec[k].load(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;

            while (resLen > kVecBatchSize)
            {
                VecType avec, bvec;
                avec.load(thisData + items);
                bvec.load(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            VecType avec, bvec;
            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);    
        }
#else

        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }


    template <>
    inline double LightVectorND<VectorNDRaw_d>::dotProductForAlignedMemory(const LightVectorND<VectorNDRaw_d>& rhs) const
    {
        assert(size() == rhs.size());
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor) {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k].load_a(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;

            while (resLen > kVecBatchSize)
            {
                VecType avec, bvec;
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            VecType avec, bvec;
            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }

    template <>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::addInPlaceVectorWithMultiple(double multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        VecType multiple_vec(multiple);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                avec[k] = ::mul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] += (bvec[k] * multiple_vec);
#endif
                
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += (bvec * multiple_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] += (vData[i] * multiple);
        }
#endif

        return *this;
    }
    
    template <>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::operator += (const LightVectorND<VectorNDStd_d>& v)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));
                avec[k] += bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += (bvec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += (bvec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] += vData[i];
        }
#endif

        return *this;
    }

    template <>
    inline double LightVectorND<VectorNDStd_d>::operator & (const LightVectorND<VectorNDStd_d>& rhs) const
    {
        assert(size() == rhs.size());
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                bvec[k].load(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
                
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(thisData + items);
                bvec.load(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }


    template <>
    inline double LightVectorND<VectorNDStd_d>::dotProductForAlignedMemory(const LightVectorND<VectorNDStd_d>& rhs) const
    {
        assert(size() == rhs.size());
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();

        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k].load_a(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }

    template <>
    inline double LightVectorND<VectorNDRaw_d>::vectorL2NormSquare() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;
            
            while (resLen > kVecBatchSize)
            {
                avec.load(thisData + items);
                resRest += ::horizontal_add(::square(avec));

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline double LightVectorND<VectorNDRaw_d>::vectorL2NormSquareForAlignedMemory() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();
        
#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                resRest += ::horizontal_add(::square(avec));

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::addWithVectorMultiple(const LightVectorND<VectorNDRaw_d>& v,
                                                                                             const TElementType multiplier)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        VecType multiple_vec(multiplier);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(inputData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                bvec[k].load(resRaw + (i + k * kVecBatchSize));

                // (1st * 2nd) + 3rd
                bvec[k] = ::mul_add(avec[k], multiple_vec, bvec[k]);
#else
                avec[k] *= multiple_vec;
                bvec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k] += avec[k];
#endif
                
                bvec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(inputData + items);
                avec *= multiple_vec;
                
                bvec.load(resRaw + items);
                bvec += avec;
                
                bvec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), inputData + items);
            avec *= multiple_vec;
            
            bvec.load_partial(int(resLen), resRaw + items);
            bvec += avec;

            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] += inputData[i] * multiplier;
        }
#endif
        
        return *this;
    }

    template<>
    // template <double multiplier>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::assignWithVectorMultiple(const LightVectorND<VectorNDRaw_d>& v, double multiplier)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();
        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        if (multiplier == -1.0)
        {
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] = -avec[k];
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec = -avec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }
                
                avec.load_partial(int(resLen), inputData + items);
                avec = -avec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = -inputData[i];
            }
#endif
            
        }
        else
        {
            VecType multiple_vec(multiplier);

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] *= multiple_vec;
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec *= multiple_vec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                avec.load_partial(int(resLen), inputData + items);
                avec *= multiple_vec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = inputData[i] * multiplier;
            }
#endif
            
        }
        
        return *this;
    }

    template<>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::assignWithVectorDifference(const LightVectorND<VectorNDRaw_d>& a, const LightVectorND<VectorNDRaw_d>& b)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(aInputData + (i + k * kVecBatchSize));
                bvec[k].load(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(aInputData + items);
                bvec.load(bInputData + items);
                avec -= bvec;
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }

    template<>
    inline LightVectorND<VectorNDRaw_d>& LightVectorND<VectorNDRaw_d>::assignWithVectorDifferenceAligned(const LightVectorND<VectorNDRaw_d>& a, const LightVectorND<VectorNDRaw_d>& b)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(aInputData + (i + k * kVecBatchSize));
                bvec[k].load_a(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(aInputData + items);
                bvec.load_a(bInputData + items);
                avec -= bvec;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            

            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }
    //========================================================================================================//

    template <>
    inline double LightVectorND<VectorNDStd_d>::vectorL2NormSquare() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load(thisData + items);
                resRest += ::horizontal_add(::square(avec));
    
                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline double LightVectorND<VectorNDStd_d>::vectorL2NormSquareForAlignedMemory() const
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                resRest += ::horizontal_add(::square(avec));
                
                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::addWithVectorMultiple(const LightVectorND<VectorNDStd_d>& v,
                                                                                             const TElementType multiplier)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        VecType multiple_vec(multiplier);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(inputData + (i + k * kVecBatchSize));
                
#if SUPPORT_CPU_FMA_EXT
                bvec[k].load(resRaw + (i + k * kVecBatchSize));

                // (1st * 2nd) + 3rd
                bvec[k] = ::mul_add(avec[k], multiple_vec, bvec[k]);
#else
                avec[k] *= multiple_vec;
                bvec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k] += avec[k];
#endif
                
                bvec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(inputData + items);
                avec *= multiple_vec;

                bvec.load(resRaw + items);
                bvec += avec;

                bvec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
                
            avec.load_partial(int(resLen), inputData + items);
            bvec.load_partial(int(resLen), resRaw + items);
            avec *= multiple_vec;
            bvec += avec;
            bvec.store_partial(int(resLen), resRaw + items);            
        }
#else
        for (; i < sz; ++i)
            resRaw[i] += inputData[i] * multiplier;
#endif
        
        return *this;
    }

    template<>
    //template<double multiplier>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::assignWithVectorMultiple(const LightVectorND<VectorNDStd_d>& v, double multiplier)
    {
        assert(size() == v.size());

        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        if (multiplier == -1.0)
        {
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] = -avec[k];
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec = -avec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                avec.load_partial(int(resLen), inputData + items);
                avec = -avec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = -inputData[i];
            }
#endif

        }
        else
        {
            VecType multiple_vec(multiplier);

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] *= multiple_vec;
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }
            
#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec *= multiple_vec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }
                
                avec.load_partial(int(resLen), inputData + items);
                avec *= multiple_vec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = inputData[i] * multiplier;
            }
#endif
        }
        
        return *this;
    }

    template<>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::assignWithVectorDifference(const LightVectorND<VectorNDStd_d>& a, const LightVectorND<VectorNDStd_d>& b)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(aInputData + (i + k * kVecBatchSize));
                bvec[k].load(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(aInputData + items);
                bvec.load(bInputData + items);
                avec -= bvec;
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }

    template<>
    inline LightVectorND<VectorNDStd_d>& LightVectorND<VectorNDStd_d>::assignWithVectorDifferenceAligned(const LightVectorND<VectorNDStd_d>& a, const LightVectorND<VectorNDStd_d>& b)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = dopt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(aInputData + (i + k * kVecBatchSize));
                bvec[k].load_a(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(aInputData + items);
                bvec.load_a(bInputData + items);
                avec -= bvec;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);

        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }
}

#endif
