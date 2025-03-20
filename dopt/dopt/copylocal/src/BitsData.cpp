#include "dopt/copylocal/include/BitsData.h"

#if 0
namespace dopt
{
    BitsData* BitsData::getDataFromMutableData(const MutableData* m, bool shareMDataPtr)
    {
        if (shareMDataPtr)
        {
            BitsData* result = new BitsData(m->getPtr(), m->getFilledSize(), Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            return result;
        }
        else
        {
            BitsData* result = new BitsData(m->getPtr(), m->getFilledSize(), Data::MemInitializedType::eAllocAndCopy);
            return result;
        }
    }
}
#endif
