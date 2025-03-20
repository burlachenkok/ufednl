#include "dopt/gpu_compute_support/include/linalg_vectors/LightVectorND_CUDA.h"
#include "dopt/gpu_compute_support/include/linalg_vectors/VectorND_CUDA_Raw.h"

template class dopt::LightVectorND_CUDA<dopt::VectorND_CUDA_Raw<double>>;
