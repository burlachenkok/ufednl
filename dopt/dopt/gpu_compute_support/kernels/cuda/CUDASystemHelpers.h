#pragma once

#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"
#include "dopt/gpu_compute_support/kernels/cuda/linalg_vectors/LinalgComputePreprocessing.h"

#include <limits>
#include <assert.h>

namespace dopt
{
    template <class T>
    KR_DEV_FN inline bool myCAS(volatile T* dest, T expected, T desired)
    {
#if 0
        if (*dest == expected)
        {
            *dest = desired;
            return true;
        }
        else
        {
            return false;
        }
#endif
        static_assert(sizeof(T) * 8 == 8 || sizeof(T) * 8 == 16 || sizeof(T) * 8 == 32 || sizeof(T) * 8 == 64 , "Only 8, 16, 32, 64 bit types are supported");
        
        if constexpr (sizeof(T) * 8 == 8)
        {
            // A Bit hacky, but CUDA Driver API does not support atomicCAS for bytes. Implementation below is independent of Endiadness of CPU.
            assert(intptr_t(dest) % 4 == 0);
            typedef int TUsedType;
            static_assert(sizeof(TUsedType) == 4, "Size of int should be 4 byte for make this byte tricks");

            // temp storage
            TUsedType desiredUInt, expectedUInt;

            // get raw byte view to temp storage
            unsigned char* b_desired = reinterpret_cast<unsigned char*>(&desiredUInt);
            unsigned char* b_expected = reinterpret_cast<unsigned char*>(&expectedUInt);

            // setup first byte
            b_desired[0] = reinterpret_cast<const unsigned char&>(desired);
            b_expected[0] = reinterpret_cast<const unsigned char&>(expected);
            
            // setup next byte from reading look ahead
            b_desired[1] = b_expected[1] = (reinterpret_cast<volatile unsigned char*>(dest))[1];
            b_desired[2] = b_expected[2] = (reinterpret_cast<volatile unsigned char*>(dest))[2];
            b_desired[3] = b_expected[3] = (reinterpret_cast<volatile unsigned char*>(dest))[3];

            TUsedType oldDestination = ::atomicCAS((TUsedType*)dest, expectedUInt, desiredUInt);
            
            return oldDestination == expectedUInt;
        }
        else if constexpr (sizeof(T) * 8 == 16)
        {
            // A Bit hacky, but CUDA Driver API does not support atomicCAS for bytes. Implementation below is independent of Endiadness of CPU.
            assert(intptr_t(dest) % 4 == 0);
            typedef int TUsedType;
            static_assert(sizeof(TUsedType) == 4, "Size of int should be 4 byte for make this byte tricks");
            static_assert(sizeof(unsigned short) == 2, "Size of unsigned short should be 2 byte for make this byte tricks");

            // temp storage
            TUsedType desiredUInt, expectedUInt;

            // get raw byte view to temp storage
            unsigned char* b_desired = reinterpret_cast<unsigned char*>(&desiredUInt);
            unsigned char* b_expected = reinterpret_cast<unsigned char*>(&expectedUInt);

            // setup first byte and second byte
            *(reinterpret_cast<unsigned short*>(b_desired)) = reinterpret_cast<const short&>(desired);
            *(reinterpret_cast<unsigned short*>(b_expected)) = reinterpret_cast<const short&>(expected);

            // setup next byte from reading look ahead
            b_desired[2] = b_expected[2] = (reinterpret_cast<volatile unsigned char*>(dest))[2];
            b_desired[3] = b_expected[3] = (reinterpret_cast<volatile unsigned char*>(dest))[3];

            TUsedType oldDestination = ::atomicCAS((TUsedType*)dest, expectedUInt, desiredUInt);

            return oldDestination == expectedUInt;
        }
        else if constexpr (sizeof(T) * 8 == 32)
        {
            assert(intptr_t(dest) % 4 == 0);
            typedef unsigned int TUsedType;
            static_assert(sizeof(TUsedType) == 4, "Size of TUsedType should be 2 bytes");

            const TUsedType& expectedUInt = reinterpret_cast<const TUsedType&>(expected);
            const TUsedType& desiredUInt = reinterpret_cast<const TUsedType&>(desired);

            TUsedType oldDestination = ::atomicCAS((TUsedType*)dest, expectedUInt, desiredUInt);
            
            return oldDestination == expectedUInt;
        }
        else if constexpr (sizeof(T) * 8 == 64)
        {
            assert(intptr_t(dest) % 8 == 0);
            typedef unsigned long long TUsedType;
            static_assert(sizeof(TUsedType) == 8, "Size of TUsedType should be 2 bytes");

            const TUsedType& expectedUInt = reinterpret_cast<const TUsedType&>(expected);
            const TUsedType& desiredUInt = reinterpret_cast<const TUsedType&>(desired);

            TUsedType oldDestination = ::atomicCAS((TUsedType*)dest, expectedUInt, desiredUInt);
            return oldDestination == expectedUInt;
        }
        else
        {
            assert(!"UNLIKELY CASE. BACKUP IMPL. PLEASE CHECK.");
            return false;
        }
    }
    
    template<BinaryReductionOperation func, class T, bool cfgTurnDoubleReadToEliminateContention = true>
    KR_DEV_FN void myAtomicApplyOperation(volatile T* pDst, T src)
    {
        T old_value_expected, new_desired_val;

        // compile-time if (C++17)
        if constexpr (cfgTurnDoubleReadToEliminateContention)
        {
            do
            {
                old_value_expected = *pDst;
                new_desired_val = binaryOp<func>(old_value_expected, src);
                // For CPU: Compare and Swap acquire cache line in exclusive mode, invalidating another caches. Leads to high contention if all processors doing CAS at the same place
                // For GPU: The same effect, but with memory transactions.
                // To reduce this effect maybe it's better to waste time on reading value from memory and do CAS only if it's necessary [TRY OFF]
            } while (old_value_expected != *pDst || !myCAS(pDst, old_value_expected, new_desired_val));
        }
        else
        {
            do
            {
                old_value_expected = *pDst;
                new_desired_val = binaryOp<func>(old_value_expected, src);
            } while (!myCAS(pDst, old_value_expected, new_desired_val));
        }
    }
}
