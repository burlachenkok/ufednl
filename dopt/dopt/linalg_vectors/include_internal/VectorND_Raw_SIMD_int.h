/** @file
* C++ cross-platform extension of math vector, elements of which are stored in std::vector.
* If there are some problem with compilation for Visual Studio specify Properties->Project->Code Generation->Enable Enhancement Instruction Set
*/

#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <limits>

#include <assert.h>
#include <math.h>
#include <stddef.h>

namespace dopt
{
    template <>
    template<int start>
    inline VectorNDRaw<int64_t> VectorNDRaw<int64_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};
        
        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = (start + static_cast<TElementType>(i));
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif
        
        return res;
    }

    template <>
    template<int start>
    inline VectorNDRaw<uint64_t> VectorNDRaw<uint64_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }

    template <>
    template<int start>
    inline VectorNDRaw<int32_t> VectorNDRaw<int32_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }

    template <>
    template<int start>
    inline VectorNDRaw<uint32_t> VectorNDRaw<uint32_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }
        
#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }


    template <>
    template<int start>
    inline VectorNDRaw<int16_t> VectorNDRaw<int16_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }

    template <>
    template<int start>
    inline VectorNDRaw<uint16_t> VectorNDRaw<uint16_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }


    template <>
    template<int start>
    inline VectorNDRaw<int8_t> VectorNDRaw<int8_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }


#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }

    template <>
    template<int start>
    inline VectorNDRaw<uint8_t> VectorNDRaw<uint8_t>::sequence(size_t dimension)
    {
        typedef dopt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = dopt::getVecBatchSize<VecType>();

        size_t sz = dimension;
        VectorNDRaw res(eAllocNotInit, sz);
        TElementType* restrict_ext resRaw = res.data();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};

        for (size_t i = 0; i < kVecBatchSize; ++i) {
            indicies[i] = start + static_cast<TElementType>(i);
        }

        VecType add_value(kVecBatchSize);
        VecType indicies_reg = VecType();
        indicies_reg.load_a(indicies);

        size_t items = dopt::roundToNearestMultipleDown<kVecBatchSize>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store_a(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
            resRaw[i] = start + i;
#endif

        return res;
    }
}

#endif
