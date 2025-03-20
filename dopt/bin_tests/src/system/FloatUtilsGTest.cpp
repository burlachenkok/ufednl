#include "dopt/system/include/FloatUtils.h"
#include "gtest/gtest.h"

#include <math.h>
#include <float.h>

TEST(dopt, FloatUtilsGTest)
{
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(1.0f).components.sign == 0);
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-1.0f).components.sign == 1);

    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(10.0).components.sign == 0);
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-12.0).components.sign == 1);

    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.0f).components.mantissa == 0);
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-64.0f).components.mantissa == 0);

    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.0).components.mantissa == 0);
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-64.0).components.mantissa == 0);

    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(0.123456).components.mantissa != 0);
    EXPECT_TRUE(dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(0.123456f).components.mantissa != 0);

    // C runtime does not inclide hidden bit
    EXPECT_TRUE( (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, double>::bits_mantissa) == DBL_MANT_DIG - 1);
    EXPECT_TRUE( (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, float>::bits_mantissa) == FLT_MANT_DIG - 1);

    EXPECT_TRUE( (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, double>::bits_mantissa) + (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, double>::bits_exponent) + 1 == 64);
    EXPECT_TRUE( (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, float>::bits_mantissa) + (dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, float>::bits_exponent) + 1 == 32);

    // Reconstruction tests [double]
    for (double i = -1000.0; i < -1.0; i += 0.123)
    {
        auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(i);
        double reconstruct = (pack.components.sign ? -1.0 : 1.0);
        reconstruct *= (1.0 + pack.components.mantissa * pack.one_div_2_bits_mantissa);
        reconstruct *= pow(2.0, pack.components.exponent - pack.exponent_pow2_shift);
        EXPECT_TRUE(fabs(reconstruct - i) < 1e-9);
    }    
    for (double i = 1000.0; i > 1.0; i -= 0.123)
    {
        auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(i);
        double reconstruct = (pack.components.sign ? -1.0 : 1.0);
        reconstruct *= (1.0 + pack.components.mantissa * pack.one_div_2_bits_mantissa);
        reconstruct *= pow(2.0, pack.components.exponent - pack.exponent_pow2_shift);
        EXPECT_TRUE(fabs(reconstruct - i) < 1e-9);
    }

    // Reconstruction tests [float]
    for (float i = -1000.0; i < -1.0; i += 0.123)
    {
        auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(i);
        float reconstruct = (pack.components.sign ? -1.0 : 1.0);
        reconstruct *= (1.0 + pack.components.mantissa * pack.one_div_2_bits_mantissa);
        reconstruct *= pow(2.0, pack.components.exponent - pack.exponent_pow2_shift);
        EXPECT_TRUE(fabs(reconstruct - i) < 1e-9);
    }
    for (float i = 1000.0; i > 1.0; i -= 0.123)
    {
        auto pack = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(i);
        float reconstruct = (pack.components.sign ? -1.0 : 1.0);
        reconstruct *= (1.0 + pack.components.mantissa * pack.one_div_2_bits_mantissa);
        reconstruct *= pow(2.0, pack.components.exponent - pack.exponent_pow2_shift);
        EXPECT_TRUE(fabs(reconstruct - i) < 1e-9);
    }

    // Test unpack functionality
    for (double i = 1000.0; i > 1.0; i -= 1.123)
    {
        for (double j = -1000.0; j < -1.0; j += 2.123)
        {
            auto packA = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(i);
            auto packB = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(j);

            uint32_t pack_one = dopt::pack2FP64NoMantissa(packA, packB);

            packA.components.mantissa = 0;
            packB.components.mantissa = 0;
            uint32_t pack_two = dopt::pack2FP64NoMantissa(packA, packB);
            
            EXPECT_TRUE(pack_one == pack_two);

            #if DOPT_ARCH_LITTLE_ENDIAN
                EXPECT_TRUE(((char*)&pack_two)[3] == 0);
            #else
                EXPECT_TRUE(((char*)&pack_two)[0] == 0);
            #endif

            dopt::FPComponentsPack<DOPT_ARCH_LITTLE_ENDIAN, double> unpack[2];
            dopt::unpack2FP64NoMantissa(unpack, pack_two);

            EXPECT_TRUE(unpack[0].components.mantissa == 0);
            EXPECT_TRUE(unpack[1].components.mantissa == 0);

            EXPECT_TRUE(packA.integer_value_repr == unpack[0].integer_value_repr);
            EXPECT_TRUE(packB.integer_value_repr == unpack[1].integer_value_repr);
        }
    }
}

TEST(dopt, FloatUtilsMaskingGTest)
{
    {
        auto pack_fp64_pos = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.123);
        EXPECT_TRUE(pack_fp64_pos.real_value_repr == 32.123);
        EXPECT_TRUE(pack_fp64_pos.components.sign == 0);

        auto pack_fp32_pos = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.123f);
        EXPECT_TRUE(pack_fp32_pos.real_value_repr == 32.123f);
        EXPECT_TRUE(pack_fp32_pos.components.sign == 0);

        auto pack_fp64_neg = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-32.123);
        EXPECT_TRUE(pack_fp64_neg.real_value_repr == -32.123);
        EXPECT_TRUE(pack_fp64_neg.components.sign == 1);

        auto pack_fp32_neg = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(-32.123f);
        EXPECT_TRUE(pack_fp32_neg.real_value_repr == -32.123f);
        EXPECT_TRUE(pack_fp32_neg.components.sign == 1);
    }

    {
        auto pack_fp64 = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(1.0/(256.0));
        EXPECT_TRUE(pack_fp64.real_value_repr == 1.0 / (256.0));
        EXPECT_TRUE(pack_fp64.components.sign == 0);
        EXPECT_TRUE(pack_fp64.components.mantissa == 0) << "Mantisssa should be zero because provided FP64 number is power of 2";

        auto pack_fp32 = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(1.0 / (256.0f));
        EXPECT_TRUE(pack_fp64.real_value_repr == 1.0 / (256.0f));
        EXPECT_TRUE(pack_fp64.components.sign == 0);
        EXPECT_TRUE(pack_fp64.components.mantissa == 0) << "Mantisssa should be zero because provided FP32 number is power of 2";
    }

    {
        auto pack_fp64 = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.123);
        EXPECT_TRUE(pack_fp64.real_value_repr == 32.123);
        EXPECT_TRUE(pack_fp64.components.sign == 0);
        EXPECT_TRUE(pack_fp64.components.mantissa != 0);

        auto mask = dopt::getFloatPointMask2RemoveMantissa<DOPT_ARCH_LITTLE_ENDIAN, decltype(pack_fp64)::TScalar>();
        auto pack_fp64_masked = pack_fp64;
        pack_fp64_masked.integer_value_repr &= mask.integer_value_repr;
        
        EXPECT_TRUE(pack_fp64.components.exponent == pack_fp64_masked.components.exponent);
        EXPECT_TRUE(pack_fp64.components.sign == pack_fp64_masked.components.sign);
        EXPECT_TRUE(pack_fp64_masked.components.mantissa == 0);
    }

    {
        auto pack_fp32 = dopt::getFloatPointPack<DOPT_ARCH_LITTLE_ENDIAN>(32.123f);
        EXPECT_TRUE(pack_fp32.real_value_repr == 32.123f);
        EXPECT_TRUE(pack_fp32.components.sign == 0);
        EXPECT_TRUE(pack_fp32.components.mantissa != 0);

        auto mask = dopt::getFloatPointMask2RemoveMantissa<DOPT_ARCH_LITTLE_ENDIAN, decltype(pack_fp32)::TScalar>();
        auto pack_fp32_masked = pack_fp32;
        pack_fp32_masked.integer_value_repr &= mask.integer_value_repr;

        EXPECT_TRUE(pack_fp32.components.exponent == pack_fp32_masked.components.exponent);
        EXPECT_TRUE(pack_fp32.components.sign == pack_fp32_masked.components.sign);
        EXPECT_TRUE(pack_fp32_masked.components.mantissa == 0);
    }
}
