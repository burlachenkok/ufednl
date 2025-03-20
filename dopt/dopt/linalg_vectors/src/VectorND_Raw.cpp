#include "VectorND_Raw.h"

#if DOPT_USE_CUSTOM_HEAPS

namespace dopt
{
    thread_local TMemPoolsForVectorsMap memPoolsForVectors = TMemPoolsForVectorsMap();
}

#endif
