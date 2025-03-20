#pragma once

#include <limits>
#include "dopt/gpu_compute_support/include/CudaRuntimeHelpers.h"

namespace dopt
{   
    enum class UnaryOperation
    {
        eAbsEw    = 0,
        eNegEw    = 1,
        eAppendEw = 2,
        eSubEw    = 3,
        eMulEw    = 4,
        eDivEw    = 5,

        eExpEw       = 6,
        eLogEw       = 7,
        eInvEw       = 8,
        eSquareEw    = 9,
        eSqrtEw      = 10,
        eInvSquareEw = 11,
        eSigmoidEw   = 12,
        eAssignEw    = 13        
    };
    
    enum class SingleArgUnaryOperation
    {
        eMultByValue  = 0,
        eDivByValue   = 1,
        eSetToValue   = 2,
        eZeroOutItems = 3,
        eAddValue     = 4
    };

    enum class TwoArgUnaryOperation
    {
        eClamp = 0,      ///< Clamp ew to [lower(arg1), upper(arg2)]
        eScaledDiff = 1, ///< Special ew operation to evaluate elementwise difference with multiple: (arg1 - item[i]) * arg2
    };

    enum class BinaryReductionOperation
    {
        eMax      = 0,
        eMin      = 1,
        eSum      = 2,
        eMultiply = 3
    };

    enum class PreprocessForBinaryReductionOperation
    {
        eIdentity = 0,
        eCard = 1,
        eLogisticLossFromMargin = 2,
        eLogisticLossFromSigmoid = 3
    };
    
    template<BinaryReductionOperation bOp, class T>
    KR_DEV_AND_HOST_FN inline T neutralForOperation()
    {
        switch (bOp)
        {
        case BinaryReductionOperation::eMax:
            return std::numeric_limits<T>::min();

        case BinaryReductionOperation::eMin:
            return std::numeric_limits<T>::max();

        case BinaryReductionOperation::eSum:
            return T();

        case BinaryReductionOperation::eMultiply:
            return T(1);
        }

        return T();
    }


    template<PreprocessForBinaryReductionOperation selPreprocess, class T>
    KR_DEV_AND_HOST_FN inline T preprocessItem(T arg)
    {
        switch (selPreprocess)
        {
        case PreprocessForBinaryReductionOperation::eIdentity:
            return arg;
        case PreprocessForBinaryReductionOperation::eCard:
            return arg > std::numeric_limits<T>::epsilon() || arg < -std::numeric_limits<T>::epsilon() ? T(1) : T(0);
        case PreprocessForBinaryReductionOperation::eLogisticLossFromMargin:
            return log(1.0 + ::exp(-double(arg)));
        case PreprocessForBinaryReductionOperation::eLogisticLossFromSigmoid:
            return -(log(double(arg)));
        }

        return T();
    }
    template<BinaryReductionOperation bOp, class T>
    KR_DEV_AND_HOST_FN inline T binaryOp(const T a, const T b)
    {
        switch (bOp)
        {
        case BinaryReductionOperation::eMax:
            return (a > b ? a : b);
        case BinaryReductionOperation::eMin:
            return (a < b ? a : b);
        case BinaryReductionOperation::eSum:
            return a + b;
        case BinaryReductionOperation::eMultiply:
            return a * b;
        }

        return T();
    }
}
