/** @file
* Float number utils
*/

#pragma once

#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include <limits>

namespace dopt
{
    template<bool isArchLitteEndian>
    struct FP64Components{};

    /**  Components of a floating point number in FP64 format for little-endian systems in IEEE 754 format.
    */
    template<>
    struct FP64Components<true /*isArchLitteEndian*/>
    {
        uint64_t mantissa : 52;  ///< 52 bits for the mantissa
        uint64_t exponent : 11;  ///< 11 bits for the exponent
        uint64_t sign : 1;       ///< 1 bit for the sign [MSB on little endian systems]
    };

    /**  Components of a floating point number in FP64 format for big-endian systems in IEEE 754 format.
    */
    template<>
    struct FP64Components<false /*isArchLitteEndian*/>
    {
        uint64_t sign : 1;       ///< 1 bit for the sign [MSB on little endian systems]
        uint64_t exponent : 11;  ///< 11 bits for the exponent
        uint64_t mantissa : 52;  ///< 52 bits for the mantissa
    };

    template<bool isArchLitteEndian>
    struct FP32Components {};

    /**  Components of a floating point number in FP32 format for little-endian systems in IEEE 754 format.
    */
    template<>
    struct FP32Components <true /*isArchLitteEndian*/> {
        uint32_t mantissa : 23;  ///< 23 bits for the mantissa
        uint32_t exponent : 8;   ///< 8 bits for the exponent
        uint32_t sign : 1;       ///< 1 bit for the sign
    };

    /**  Components of a floating point number in FP32 format for big-endian systems in IEEE 754 format.
    */
    template<>
    struct FP32Components <false /*isArchLitteEndian*/> {
        uint32_t sign : 1;       ///< 1 bit for the sign
        uint32_t exponent : 8;   ///< 8 bits for the exponent
        uint32_t mantissa : 23;  ///< 23 bits for the mantissa
    };

    template<bool isArchLitteEndian>
    struct FP16Components {};
    
    /**  Components of a floating point number in FP16 format for little-endian systems in IEEE 754 format.
    */
    template<>
    struct FP16Components<true /*isArchLitteEndian*/> {
        uint16_t mantissa : 10;  ///< 10 bits for the mantissa
        uint16_t exponent : 5;   ///< 5 bits for the exponent
        uint16_t sign : 1;       ///< 1 bit for the sign
    };

    /**  Components of a floating point number in FP16 format for big-endian systems in IEEE 754 format.
    */
    template<>
    struct FP16Components<false /*isArchLitteEndian*/> {
        uint16_t sign : 1;       ///< 1 bit for the sign
        uint16_t exponent : 5;   ///< 5 bits for the exponent
        uint16_t mantissa : 10;  ///< 10 bits for the mantissa
    };
    //============================================================================
    template <bool isArchLitteEndian, class T>
    union FPComponentsPack {
        using TScalar = T;

        T real_value_repr; ///< Real value representation
    };
    
    template <bool isArchLitteEndian>
    union FPComponentsPack<isArchLitteEndian, double>
    {
        using TScalar = double;
        
        double real_value_repr;                          ///< Real value representation of scalar
        uint64_t integer_value_repr;                     ///< Unigned integer representation to work with bits
        FP64Components<isArchLitteEndian> components;    ///< Bitwise components

        static constexpr size_t bits_sign = 1;
        static constexpr size_t bits_exponent = 11;
        static constexpr size_t bits_mantissa = 52;
        
        static constexpr double one_div_2_bits_mantissa = 2.220446049250313e-16; ///< 2^{-52}
        static constexpr int exponent_pow2_shift = 1023;
    };

    template <bool isArchLitteEndian>
    union FPComponentsPack <isArchLitteEndian, float>
    {
        using TScalar = float;
        
        float real_value_repr;                          ///< Real value representation of scalar
        uint32_t integer_value_repr;                    ///< Unigned integer representation to work with bits
        FP32Components<isArchLitteEndian> components;   ///< Bitwise components

        static constexpr size_t bits_sign = 1;
        static constexpr size_t bits_exponent = 8;
        static constexpr size_t bits_mantissa = 23;

        static constexpr double one_div_2_bits_mantissa = 1.1920928955078125e-07; ///< 2^{-23}
        static constexpr int exponent_pow2_shift = 127;
    };

    //============================================================================
    static_assert(sizeof(double) == sizeof(FP64Components<true>));
    static_assert(sizeof(float) == sizeof(FP32Components<true>));
    static_assert(sizeof(uint16_t) == sizeof(FP16Components<true>));

    static_assert(sizeof(double) == sizeof(FP64Components<false>));
    static_assert(sizeof(float) == sizeof(FP32Components<false>));
    static_assert(sizeof(uint16_t) == sizeof(FP16Components<false>));

    static_assert(sizeof(double) == sizeof(FPComponentsPack<true, double>));
    static_assert(sizeof(float) == sizeof(FPComponentsPack<true, float>));
    
    //============================================================================

    /** Get mask to remove mantissa part of the floating point number.
    * @return Mask to remove mantissa part of the floating point number with AND operation.
    * @tparam T Type of the floating point number.
    * @tparam isArchLitteEndian Flag of the system architecture.
    */
    template <bool isArchLitteEndian, class T>
    inline FPComponentsPack<isArchLitteEndian, T> getFloatPointMask2RemoveMantissa() {
        FPComponentsPack<isArchLitteEndian, T> pack;
        pack.integer_value_repr = (decltype(pack.integer_value_repr)) (-1);
        pack.components.mantissa = 0;
        return pack;
    }

    /** Get float point scalar packed into the structure for comfortable work in bitwise representation too.
    * @param value real value of the floating point number.
    * @return Packed structure with the real value and bitwise representation.
    */
    template <bool isArchLitteEndian, class T>
    inline FPComponentsPack<isArchLitteEndian, T> getFloatPointPack(T value) {
        FPComponentsPack<isArchLitteEndian, T> pack;
        pack.real_value_repr = value;
        return pack;
    }

    /** Get matissa part of the floating point number, but without leading one
    * @param pack the packed structure with the real value and bitwise representation.
    * @return scalar which represents mantissa part of the floating point number, but without leading one.
    * @note FP64 has representation (-1)^sign x 1.b51_b50_...._b1_b0 x 2^{exponent - 1023}
    * @note If ignore exponent, sign, and leading one then still there a need to multiply mantissa by 2^{-52}
    */
    template <bool isArchLitteEndian, class T>
    inline double getMantissaPartNoLeadingOne(FPComponentsPack<isArchLitteEndian, T> pack) {
        double value = double(pack.components.mantissa);
        value *= FPComponentsPack<isArchLitteEndian, T>::one_div_2_bits_mantissa;
        return value;
    }

    /** Get matissa part of the floating point number, without leading one
    * @param pack the packed structure with the real value and bitwise representation.
    * @return scalar which represents mantissa part of the floating point number, but without leading one.
    * @note FP64 has representation (-1)^sign x 1.b51_b50_...._b1_b0 x 2^{exponent - 1023}
    * @note If ignore exponent, sign, and leading one then still there a need to multiply matissa w/o leading 1 2^{-52}, and add "1"
    */
    template <class T, bool isArchLitteEndian>
    inline double getMantissaPartAndLeadingOne(FPComponentsPack<isArchLitteEndian, T> pack) {
        double value = double(pack.components.mantissa);
        value *= FPComponentsPack<isArchLitteEndian, T>::one_div_2_bits_mantissa;
        return 1.0 + value;
    }

    /** Pack two floating point numbers into one 32-bit integer (actually 24 bits part)
    * @param itemA the packed structure with the real value and bitwise representation of the first floating point number.
    * @param itemB the packed structure with the real value and bitwise representation of the second floating point number.
    * @return 32-bit integer which contains packed two floating point numbers in the first 24 bits (3 bytes)
    * @note the bitwise representation for Little-Endian: [signA(1b) expPart(11) signB(1) expPart(11)]
    */    
    template <bool isArchLitteEndian>
    inline uint32_t pack2FP64NoMantissa(FPComponentsPack<isArchLitteEndian, double> itemA,
                                        FPComponentsPack<isArchLitteEndian, double> itemB)
    {
        constexpr size_t sign_a_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian, double>::bits_sign;
        constexpr size_t mant_a_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian, double>::bits_sign;
        constexpr size_t sign_b_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent;
        constexpr size_t mant_b_shift = 0;

        // CHECK THAT EVERYTHIN IS FIT INTO 3 BYTES
        static_assert(sign_a_shift < 24);
        static_assert(mant_a_shift < 24);
        static_assert(sign_b_shift < 24);
        static_assert(mant_b_shift < 24);

        uint32_t result = (itemA.components.sign     << sign_a_shift) |
                          (itemA.components.exponent << mant_a_shift) |
                          (itemB.components.sign     << sign_b_shift) |
                          (itemB.components.exponent << mant_b_shift);

        assert(((char*)&result)[3] == 0);
        
        return result;
    }

    /** Unpack two floating point numbers from one 32-bit integer (actually 24 bits part).
    * @param itemA the packed structure with the real value and bitwise representation of the first floating point number.
    * @param itemB the packed structure with the real value and bitwise representation of the second floating point number.
    * @return 32-bit integer which contains packed two floating point numbers in the first 24 bits (3 bytes)
    * @note the bitwise representation for Little-Endian: [signA(1b) expPart(11) signB(1) expPart(11)]
    */
    template <bool isArchLitteEndian>
    inline void unpack2FP64NoMantissa(FPComponentsPack<isArchLitteEndian, double> unpack[2], uint32_t buffer)
    {
        constexpr size_t sign_a_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian,double>::bits_sign;
        constexpr size_t mant_a_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent + FPComponentsPack<isArchLitteEndian, double>::bits_sign;
        constexpr size_t sign_b_shift = FPComponentsPack<isArchLitteEndian, double>::bits_exponent;
        constexpr size_t mant_b_shift = 0;
        static_assert(sign_a_shift <= 31);
        static_assert(mant_a_shift <= 31);
        static_assert(sign_b_shift <= 31);
        static_assert(mant_b_shift <= 31);

        FPComponentsPack<isArchLitteEndian, double>& itemA = unpack[0];
        itemA.components.sign = (buffer >> sign_a_shift) & 0x1;

        static_assert(FPComponentsPack<isArchLitteEndian, double>::bits_exponent == 11, "Next line of code is valid only for 11 bits exponent");
        itemA.components.exponent = (buffer >> mant_a_shift) & 0b1111'1111'111;
        itemA.components.mantissa = 0;

        FPComponentsPack<isArchLitteEndian, double>& itemB = unpack[1];
        itemB.components.sign = (buffer >> sign_b_shift) & 0x1;
        
        static_assert(FPComponentsPack<isArchLitteEndian, double>::bits_exponent == 11, "Next line of code is valid only for 11 bits exponent");
        itemB.components.exponent = (buffer >> mant_b_shift) & 0b1111'1111'111;
        itemB.components.mantissa = 0;
    }
}
