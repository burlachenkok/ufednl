#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/MutableBitsData.h"
#include "dopt/math_routines/include/SimpleMathRoutines.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

TEST(dopt, MutableBitsDataFillingGTest)
{
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);
        EXPECT_TRUE(mdata.isEmpty());
        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 0);
        mdata.putBit(1);
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 1);
        EXPECT_FALSE(mdata.isEmpty());
        EXPECT_FALSE(mdata.isByteBufferComplete());
        for (size_t i = 0; i < 7; ++i)
            mdata.putBit(1);
        EXPECT_FALSE(mdata.isEmpty());
        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.getFilledSizeInBits() == mdataBuffer.getFilledSize() * 8);
    }

    {
        uint32_t value = 0x12345678;
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);
        
        mdata.putBits(&value, 0);
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 0);
        
        mdata.putBits(&value, 1);
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 1);

        mdata.putBits(&value, 1);
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 2);

        mdata.putBits(&value, 32);
        EXPECT_TRUE(mdata.getFilledSizeInBits() == 2+32);
    }
}

TEST(dopt, MutableBitsDataGTest)
{
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);
        EXPECT_TRUE(mdata.isEmpty());
        EXPECT_TRUE(mdata.putBit(0xff)); // put bit with value "1"
        EXPECT_FALSE(mdata.isEmpty());

        dopt::MutableBitsData mdataCopy(mdata);
        EXPECT_FALSE(mdataCopy.isEmpty());
    }
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        for (uint64_t i = 0; i < 100; ++i)
        {
            EXPECT_EQ(i, mdata.getFilledSizeInBits());
            EXPECT_TRUE(mdata.putBit(1));
        }
        EXPECT_EQ(100, mdata.getFilledSizeInBits());
        mdata.putByte(0);
        EXPECT_EQ(108, mdata.getFilledSizeInBits());

        unsigned char buff[10] = {};
        EXPECT_TRUE(mdata.putBits(buff, 42));
        EXPECT_EQ(108+42, mdata.getFilledSizeInBits());
        EXPECT_TRUE(mdata.putFloat(0.0f));
        EXPECT_EQ(108+42+32, mdata.getFilledSizeInBits());
        EXPECT_TRUE(mdata.putDouble(0.0f));
        EXPECT_EQ(108+42+32+64, mdata.getFilledSizeInBits());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        mdata.putBit(1);
        mdata.putInt32(0);
        EXPECT_EQ(1+32, mdata.getFilledSizeInBits());
        mdata.putUint32(0);
        EXPECT_EQ(1+32+32, mdata.getFilledSizeInBits());
        mdata.putUint64(0);
        EXPECT_EQ(1+32+32+64, mdata.getFilledSizeInBits());
        mdata.putInt64(0);
        EXPECT_EQ(1+32+32+64+64, mdata.getFilledSizeInBits());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.putBit(1));
        unsigned char tmp = 0xFF;
        EXPECT_TRUE(mdata.putBits(&tmp, 6));
        EXPECT_FALSE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.putFloat(0.0f));
        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.putBits(&tmp, 4));
        EXPECT_FALSE(mdata.isByteBufferComplete());
        EXPECT_TRUE(mdata.putBits(&tmp, 4));
        EXPECT_TRUE(mdata.isByteBufferComplete());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.putBit(0));
        EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.putBit(0));
        EXPECT_TRUE(mdata.putBit(0));
        EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.putBit(0));
        EXPECT_EQ(dopt::createByte(0,1,0,0, 1,0,1,1), mdata.getData()->getByte());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        uint8_t b1 = dopt::createByte(0,1,0,0, 1,0,1,1);
        EXPECT_TRUE(mdata.putBits(&b1, 8));
        EXPECT_EQ(b1, mdata.getData()->getByte());

        uint8_t b2 = dopt::createByte(0,1,1,0, 1,1,1,1);
        EXPECT_TRUE(mdata.putBits(&b2, 8));
        EXPECT_TRUE(mdata.isByteBufferComplete());
        
        EXPECT_EQ(2, mdata.getData()->getResidualLength());
        EXPECT_EQ(2, mdata.getData()->getTotalLength());

        dopt::DataUniquePtr d = mdata.getData();
        EXPECT_EQ(b1, d->getByte());
        EXPECT_EQ(b2, d->getByte());
        EXPECT_TRUE(d->isEmpty());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        uint32_t testValue = 0xA1B2C3E4;

        #if DOPT_ARCH_LITTLE_ENDIAN
            uint32_t testValueInLSB = testValue;
        #else
            uint32_t testValueInLSB = dopt::lsbToMsb32(testValue);
        #endif

        uint8_t bits[32] = {};
        EXPECT_EQ(32, dopt::getLsbBits(bits, testValueInLSB));

        for (size_t i = 0; i < std::size(bits); ++i)
        {
            bool isBitSetup  = bits[i] ? true : false;
            bool isBitSetupInTestValue  = (testValue & (0x1 << i)) ? true : false;
            
            EXPECT_EQ(isBitSetup, isBitSetupInTestValue);
            
            EXPECT_EQ(i, mdata.getFilledSizeInBits());

            EXPECT_TRUE(mdata.putBit(bits[i]));

            if ( (i + 1) % 8 == 0)
                EXPECT_TRUE(mdata.isByteBufferComplete());
            else
                EXPECT_FALSE(mdata.isByteBufferComplete());
        }

        EXPECT_TRUE(mdata.isByteBufferComplete());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        mdata.putBit(1);
        mdata.putBit(1);
        EXPECT_TRUE(mdata.putTuple2(uint16_t(90), 90.0f));
        EXPECT_EQ(2+16+32, mdata.getFilledSizeInBits());
        EXPECT_TRUE(mdata.putTuple3(12.0f, 90.0f, 90.0f));
        EXPECT_EQ(2+16+32 + 32*3, mdata.getFilledSizeInBits());
        EXPECT_FALSE(mdata.isByteBufferComplete());
        for (int i = 0; i < 6; ++i)
            EXPECT_TRUE(mdata.putBit(1));
        EXPECT_TRUE(mdata.isByteBufferComplete());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        EXPECT_TRUE(mdata.putTuple4(1.0, 2.0, 3.0, 4.0));
        EXPECT_EQ(4*64, mdata.getFilledSizeInBits());
        EXPECT_TRUE(mdata.isByteBufferComplete());

        dopt::DataUniquePtr d = mdata.getData();
        EXPECT_TRUE(d.get() != NULL);

        EXPECT_DOUBLE_EQ(1.0, d->getDouble());
        EXPECT_DOUBLE_EQ(2.0, d->getDouble());
        EXPECT_DOUBLE_EQ(3.0, d->getDouble());
        EXPECT_DOUBLE_EQ(4.0, d->getDouble());
        EXPECT_TRUE(d->isEmpty());
    }

    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        uint8_t bits[8] = {1, 0, 1, 0, 0, 1, 1, 1};
        
        for (size_t i = 0; i < std::size(bits); ++i)
        {
            EXPECT_TRUE(mdata.putBit(bits[7-i]));
        }
        EXPECT_TRUE(mdata.isByteBufferComplete());
        EXPECT_EQ(dopt::createByte(1, 0, 1, 0, 0, 1, 1, 1), mdata.getData()->getByte());
    }
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdata(&mdataBuffer);

        uint8_t tmp[] = {10, 11, 14};
        
        EXPECT_TRUE(mdata.putBits(tmp, 8 * sizeof(tmp)));
        dopt::DataUniquePtr d = mdata.getData();

        EXPECT_TRUE(d.get()!= NULL);
        EXPECT_EQ(sizeof(tmp), d->getTotalLength());

        for (size_t i = 0; i < std::size(tmp); ++i)
        {
            EXPECT_EQ(i, d->seekStart(i));
            EXPECT_EQ(tmp[i], d->getByte());
        }
    }
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdataLsb(&mdataBuffer);

        //               LSB                   MSB
        uint8_t bits[] = {1, 1, 1, 1, 0, 0, 0, 0};
        EXPECT_EQ(8, std::size(bits));
        for (int i = 0; i < 8; ++i)
        {
            mdataLsb.putBit(bits[i]);
        }

        EXPECT_EQ(dopt::createByteFromLsbList(bits), mdataLsb.getData()->getByte());
        EXPECT_TRUE(mdataLsb.isByteBufferComplete());
    }
    {
        dopt::MutableData mdataBuffer;
        dopt::MutableBitsData mdataLsb(&mdataBuffer);

        uint8_t testByte = dopt::createByte(1, 1, 1, 1, 0, 0, 0, 0);
        mdataLsb.putBits(&testByte, sizeof(testByte) * 8);
        EXPECT_EQ(testByte, mdataLsb.getData()->getByte());

        uint16_t t = 65432;
        mdataLsb.putBits(&t, 8 * sizeof(t));
        
        dopt::DataUniquePtr dd = mdataLsb.getData();
        dd->seekCur(1);
        EXPECT_EQ(t, dd->getUint16());
     }
}
