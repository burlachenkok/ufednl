#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/BitsData.h"
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/MutableBitsData.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(dopt, BitsDataGTest)
{
    {
        uint8_t buffer[] = {dopt::createByte(0,1,1,1, 0,0,0,0),
                            dopt::createByte(0,1,1,1, 0,0,0,1)
                           };

        dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::BitsData mdata(&bufferData);
        
        EXPECT_EQ(16, mdata.getResidualLengthInBits());
        EXPECT_EQ(buffer[0], mdata.getByte());
        EXPECT_EQ(8, mdata.getResidualLengthInBits());
        EXPECT_FALSE(mdata.isEmpty());
        EXPECT_EQ(8, mdata.getPosInBits());
        EXPECT_EQ(buffer[1], mdata.getByte());
        EXPECT_TRUE(mdata.isEmpty());
        EXPECT_EQ(16, mdata.getPosInBits());
    }

    {
        uint8_t buffer[] = { dopt::createByte(1/*8-th bit*/, 1 /*7th*/, 1, 1, 0, 0, 0, 0 /*1st bit*/) };
        
        dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::BitsData mdata(&bufferData);
        
        EXPECT_EQ(8, mdata.getResidualLengthInBits());
        EXPECT_EQ(0, mdata.getPosInBits());
        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_FALSE(mdata.isEmpty());
            EXPECT_EQ(0, mdata.getBit());
        }
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(mdata.isEmpty());
            EXPECT_EQ(1, mdata.getBit());
        }
        EXPECT_EQ(7, mdata.getPosInBits());
        mdata.getBit();
        EXPECT_TRUE(mdata.isEmpty());
    }
    {
        uint8_t buffer[] = { dopt::createByte(1, 1, 1, 1, 0, 0, 0, 0) };
        
        dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::BitsData extract(&bufferData);

        EXPECT_EQ(dopt::createByte(1, 1, 1, 1, 0, 0, 0, 0), extract.getByte());
    }

    {
        float buffer[] = {0.1f, 0.2f, 0.3f};
        
        dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::BitsData mdata(&bufferData);

        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_EQ(8 * sizeof(buffer) - i, mdata.getResidualLengthInBits());
            mdata.getBit();
        }
        EXPECT_FLOAT_EQ(buffer[1], mdata.getFloat());
        EXPECT_FLOAT_EQ(buffer[2], mdata.getFloat());
        EXPECT_TRUE(mdata.isEmpty());
    }

    {
        uint8_t bitsByte1[] = {1, 1, 1, 1, 0, 0, 0, 0}; /* lsb -- 1, msb -- 0*/
        uint8_t bitsByte2[] = {1, 0, 1, 1, 0, 0, 1, 0}; /* lsb -- 1, msb -- 0*/
        uint8_t buffer[] = { dopt::createByteFromLsbList(bitsByte1), dopt::createByteFromLsbList(bitsByte2) };

        dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        dopt::BitsData mdata(&bufferData);

        EXPECT_EQ(16, mdata.getResidualLengthInBits());
        EXPECT_EQ(0, mdata.getPosInBits());
        EXPECT_FALSE(mdata.isEmpty());

        for (size_t i = 0; i < std::size(bitsByte1); ++i)
        {
            EXPECT_EQ(i, mdata.getPosInBits());
            EXPECT_EQ(16 - i, mdata.getResidualLengthInBits());
            EXPECT_EQ(bitsByte1[i], mdata.getBit()); // take one bit
        }
        EXPECT_EQ(8, mdata.getResidualLengthInBits());

        for (size_t i = 0; i < std::size(bitsByte1); ++i)
        {
            EXPECT_EQ(8 + i, mdata.getPosInBits());
            EXPECT_EQ(8 - i, mdata.getResidualLengthInBits());
            EXPECT_EQ(bitsByte2[i], mdata.getBit()); // take one bit
        }
        EXPECT_EQ(0, mdata.getResidualLengthInBits());
        EXPECT_TRUE(mdata.isEmpty());
    }

    {
        uint8_t bitsByte1[] = {1, 1, 1, 1, 0, 0, 0, 0}; /* lsb -- 1, msb -- 0*/
        uint8_t bitsByte2[] = {1, 0, 1, 1, 0, 0, 1, 0}; /* lsb -- 1, msb -- 0*/
        uint8_t buffer[] = { dopt::createByteFromLsbList(bitsByte1), dopt::createByteFromLsbList(bitsByte2) };

        {
            dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            dopt::BitsData mdata(&bufferData);

            for (size_t i = 0; i < std::size(bitsByte1); ++i)
                EXPECT_EQ(bitsByte1[i], mdata.getBit());
            for (size_t i = 0; i < std::size(bitsByte2); ++i)
                EXPECT_EQ(bitsByte2[i], mdata.getBit());
        }
        
        {
            dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            dopt::BitsData mdata(&bufferData);

            EXPECT_EQ(buffer[0], mdata.getByte());
            EXPECT_EQ(buffer[1], mdata.getByte());
        }

        {
            dopt::Data bufferData(buffer, sizeof(buffer), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            dopt::BitsData mdata(&bufferData);
            
            for (size_t i = 0; i < std::size(bitsByte1); ++i)
                EXPECT_EQ(bitsByte1[i], mdata.getBit());
            for (size_t i = 0; i < std::size(bitsByte2); ++i)
                EXPECT_EQ(bitsByte2[i], mdata.getBit());
        }
    }
}

TEST(dopt, BitsDataFillingGTest)
{
    uint64_t value_1 = 0x2734;
    uint32_t value_2 = 0x12345678;
    uint16_t value_3 = 0x2234;
    float    value_4 = 0.123f;
    double   value_5 = -0.778;
    uint8_t  value_6 = 120;   
    
    dopt::MutableData mdataBuffer;
    dopt::MutableBitsData mdata(&mdataBuffer);
    mdata.putBit(0);
    mdata.putBits(&value_1, sizeof(value_1) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + 1);
    
    mdata.putBits(&value_2, sizeof(value_2) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + 1);
    
    mdata.putBits(&value_3, sizeof(value_3) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + sizeof(value_3) * 8  +  1);

    mdata.putBits(&value_4, sizeof(value_4) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + sizeof(value_3) * 8 + sizeof(value_4) * 8 + 1);

    mdata.putBits(&value_5, sizeof(value_5) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + sizeof(value_3) * 8 + sizeof(value_4) * 8  + sizeof(value_5) * 8 + 1);

    mdata.putBits(&value_6, sizeof(value_6) * 8);
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + sizeof(value_3) * 8 + sizeof(value_4) * 8 + sizeof(value_5) * 8 + sizeof(value_6) * 8 + 1);
        
    while (!mdata.isByteBufferComplete())
    {
        mdata.putBit(0);
    }
    EXPECT_TRUE(mdata.getFilledSizeInBits() == sizeof(value_1) * 8 + sizeof(value_2) * 8 + sizeof(value_3) * 8 + sizeof(value_4) * 8 + sizeof(value_5) * 8 + sizeof(value_6) * 8 + 8);
    EXPECT_TRUE(mdata.isByteBufferComplete());

    dopt::Data bufferData(mdataBuffer.getPtr(), mdataBuffer.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
    dopt::BitsData toread(&bufferData);

    {
        EXPECT_TRUE(0 == toread.getBit());

        uint64_t value_1_2read = 0;
        uint32_t value_2_2read = 0;
        uint16_t value_3_2read = 0;
        float    value_4_2read = 0;
        double   value_5_2read = 0;
        uint8_t  value_6_2read = 0;
        
        toread.getBits(&value_1_2read, sizeof(value_1_2read) * 8);
        EXPECT_TRUE(value_1 == value_1_2read);

        toread.getBits(&value_2_2read, sizeof(value_2_2read) * 8);
        EXPECT_TRUE(value_2 == value_2_2read);

        toread.getBits(&value_3_2read, sizeof(value_3_2read) * 8);
        EXPECT_TRUE(value_3 == value_3_2read);

        toread.getBits(&value_4_2read, sizeof(value_4_2read) * 8);
        EXPECT_TRUE(value_4 == value_4_2read);

        toread.getBits(&value_5_2read, sizeof(value_5_2read) * 8);
        EXPECT_TRUE(value_5 == value_5_2read);

        toread.getBits(&value_6_2read, sizeof(value_6_2read) * 8);
        EXPECT_TRUE(value_6 == value_6_2read);
    }

    toread.rewindToStart();
    
    {
        EXPECT_TRUE(0 == toread.getBit());

        uint64_t value_1_2read = 0;
        uint32_t value_2_2read = 0;
        uint16_t value_3_2read = 0;
        float    value_4_2read = 0;
        double   value_5_2read = 0;
        uint8_t  value_6_2read = 0;

        toread.getBits(&value_1_2read, sizeof(value_1_2read) * 8);
        EXPECT_TRUE(value_1 == value_1_2read);

        toread.getBits(&value_2_2read, sizeof(value_2_2read) * 8);
        EXPECT_TRUE(value_2 == value_2_2read);
    }
}
