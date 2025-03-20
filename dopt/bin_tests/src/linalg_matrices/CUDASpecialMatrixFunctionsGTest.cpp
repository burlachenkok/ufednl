#if DOPT_CUDA_SUPPORT

#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/gpu_compute_support/include/CUDASpecialMathRoutinesForMatrix.h"

#include "gtest/gtest.h"
#include <stdint.h>
#include <iostream>

TEST(dopt, CUDAMatrixIndexingFillIn)
{
    std::vector<size_t> shapes = { 3, 5, 10, 20, 100, 200 };
    
    for (size_t rowsAndCols : shapes)
    {
        dopt::MatrixNMD_CUDA<dopt::VectorND_CUDA_Raw<float>> matrix(rowsAndCols, rowsAndCols);
        dopt::VectorND_CUDA_Raw<uint32_t> indDev = dopt::indiciesForUpperTriangularPart(matrix);
        dopt::VectorNDRaw<uint32_t> indHost;
        indDev.store(indHost);

        EXPECT_TRUE(matrix.columns() == matrix.rows());
        EXPECT_TRUE(indHost.size() == (matrix.columns() * (matrix.columns() + 1) / 2));
        EXPECT_TRUE(indDev.size() == (matrix.columns() * (matrix.columns() + 1) / 2));

        size_t pos = 0;
        for (size_t j = 0; j < matrix.columns(); j++)
        {
            for (size_t i = 0; i <= j; i++, pos++)
            {
                size_t posInMatrix = matrix.getFlattenIndexFromPosition(i, j);
                EXPECT_EQ(indHost[pos], posInMatrix);
            }
        }
    }
}
    
#endif
