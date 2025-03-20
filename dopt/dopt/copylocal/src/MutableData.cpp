﻿#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/Data.h"

#include <assert.h>
#include <stdarg.h>
#include <string>

namespace dopt
{
    MutableData& MutableData::operator = (const MutableData& rhs)
    {
        if (this == &rhs)
            return *this;

        length = rhs.length;
        pos = rhs.pos;

        if (bits)
        {
            deallocateBytes(bits);
            bits = NULL;
        }

        if (rhs.bits)
        {
            bits = (uint8_t*)allocateBytes<true>(length);
            memcpy(bits, rhs.bits, length);
        }

        return *this;
    }

    bool MutableData::realloc_internal_with_increase(size_t newSize)
    {
        assert(newSize > 0);
        assert(newSize > length);

        newSize = dopt::roundToNearestMultipleUp<kChunkSizeIncrease>(newSize);

        uint8_t* ptr = (uint8_t*)allocateBytes<true/*newSize not zero*/>(newSize);
        
        if (ptr == nullptr)
            return false;

        if (bits != nullptr)
        {
            memcpy(ptr, bits, length);
            deallocateBytes(bits);
            bits = nullptr;
        }
        
        bits = ptr;
        length = newSize;
        return true;
    }

    MutableData* MutableData::getMutableDataFromData(const Data& d)
    {
        MutableData* mdata = new MutableData();
        mdata->putBytes(d.getPtrToResidual(), d.getResidualLength());
        return mdata;
    }

    MutableData* MutableData::getMutableDataFromData(const Data& d, size_t bytesLength)
    {
        size_t readBytes = bytesLength < d.getResidualLength() ? bytesLength : d.getResidualLength();
        MutableData* mdata = new MutableData();
        mdata->putBytes(d.getPtrToResidual(), readBytes);
        return mdata;
    }
}
