#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/linalg_matrices/include/MatrixNMD.h"
#include "dopt/linalg_vectors/include/VectorND_Raw.h"
#include "dopt/random/include/RandomGenRealLinear.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace
{
    struct TestPod
    {
        int a;
        int b;
        float c;
    };
}

TEST(dopt, MutableDataGTest)
{
	dopt::MutableData mdata;
	EXPECT_EQ(0, mdata.getFilledSize());
	EXPECT_EQ(mdata.getFilledSize(), 0);
	EXPECT_TRUE(mdata.isEmpty());

	EXPECT_TRUE(mdata.putByte(23));
	EXPECT_EQ(1, mdata.getFilledSize());
	EXPECT_TRUE(mdata.putByte(23));
	EXPECT_EQ(2, mdata.getFilledSize());
	EXPECT_TRUE(mdata.putInt16(1));
	EXPECT_EQ(4, mdata.getFilledSize());
	EXPECT_FALSE(mdata.isEmpty());

	dopt::MutableData mdataCopy(mdata);
	EXPECT_FALSE(mdata.isEmpty());
	EXPECT_FALSE(mdataCopy.isEmpty());

	EXPECT_EQ(mdataCopy.getFilledSize(), mdata.getFilledSize());
	EXPECT_NE(mdataCopy.getPtr(), mdata.getPtr());

    EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0);
    EXPECT_TRUE(mdataCopy.getAllocBytesToStoreData() > 0);

    EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);
    EXPECT_TRUE(mdataCopy.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);
	
    size_t offsetCurrent = mdataCopy.seekCur(0);
    EXPECT_EQ(mdataCopy.seekCur(0), mdata.seekCur(0));
	EXPECT_TRUE(mdataCopy.putFloat(12.0f));
	EXPECT_TRUE(mdataCopy.putDouble(12.0));

    EXPECT_EQ(mdataCopy.seekCur(0), offsetCurrent + sizeof(float) + sizeof(double));

	EXPECT_NE(mdataCopy.getFilledSize(), mdata.getFilledSize());
	mdataCopy = mdata;
	EXPECT_EQ(mdataCopy.getFilledSize(), mdata.getFilledSize());

	mdata.seekStart(0);
    EXPECT_NE(mdataCopy.seekCur(0), mdata.seekCur(0));
	mdataCopy = mdata;
    EXPECT_EQ(mdataCopy.seekCur(0), mdata.seekCur(0));
}

TEST(Data, MutableDataOverflowGTest)
{
    {
	    dopt::MutableData mdata; // reallocate memory in case of overflow memory, memory was allocated with sizes (n*10 bytes)

	    EXPECT_EQ(mdata.getAllocBytesToStoreData(), 0) << "Mutable data after ctor should consumed 0 bytes from heap";
	    EXPECT_TRUE(mdata.putInt32(1)) << "Just try put 32-bit signed integer";
	    EXPECT_TRUE(mdata.putInt32(1));
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);

	    EXPECT_EQ(mdata.getFilledSize(), 2*4);

	    EXPECT_TRUE(mdata.putUint32(1));
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);

	    EXPECT_EQ(mdata.getFilledSize(), 3*4);
	    EXPECT_EQ(mdata.putBytes(nullptr, 8), 0);
        EXPECT_EQ(mdata.putBytes('0', 8), 8);

	    EXPECT_EQ(mdata.getFilledSize(), 3*4 + 8);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);

	    EXPECT_TRUE(mdata.putString("123", dopt::MutableData::PutStringFlags::ePutNoTerminator));
	    EXPECT_EQ(mdata.getFilledSize(), 3*4+8+3);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0);
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);

	    EXPECT_TRUE(mdata.putString("123", dopt::MutableData::PutStringFlags::ePutZeroTerminator));
	    EXPECT_EQ(mdata.getFilledSize(), 3*4+8+3+3+1);
    }

    {
        dopt::MutableData mdataFilling;
        mdataFilling.putTuple3(1.0, 2.0, 3.0);
        EXPECT_EQ(3*sizeof(double), mdataFilling.getFilledSize());

        mdataFilling.seekCur(-8);
        EXPECT_EQ(2*sizeof(double), mdataFilling.getFilledSize());
        EXPECT_TRUE(mdataFilling.putTuple2(int64_t(1), int(1)));
        EXPECT_EQ(2*sizeof(double) + sizeof(int64_t) + sizeof(int), mdataFilling.getFilledSize());

        EXPECT_EQ(0, mdataFilling.seekStart(0));
        EXPECT_TRUE(mdataFilling.putTuple2(uint16_t(1), uint32_t(1)));
        EXPECT_EQ(2 + 4, mdataFilling.getFilledSize());
        EXPECT_EQ(0, mdataFilling.seekStart(0));
        EXPECT_TRUE(mdataFilling.putTuple3(uint16_t(1), uint32_t(1), double(1)));
        EXPECT_EQ(2 + 4 + 8, mdataFilling.getFilledSize());

        EXPECT_EQ(mdataFilling.seekCur(0), mdataFilling.getCurWritePos());
        EXPECT_EQ(mdataFilling.seekStart(0), 0);

        float arr1[3] = {};
        EXPECT_EQ(mdataFilling.putBytes<false> (arr1, sizeof(arr1)), sizeof(arr1));
        EXPECT_EQ(mdataFilling.getFilledSize(), 0);

        EXPECT_TRUE(mdataFilling.putTuple3(arr1[0], arr1[1], arr1[2]));
        EXPECT_EQ(mdataFilling.getFilledSize(), 12);
        EXPECT_EQ(mdataFilling.seekStart(0), mdataFilling.seekStart(0));
    }

    {
        dopt::MutableData mdataFilling;
        float arr2[3] = {1.0f, 2.0f, 3.0f};
        EXPECT_EQ(mdataFilling.putBytes<true> (arr2, sizeof(arr2)), 12);
        EXPECT_EQ(mdataFilling.getFilledSize(), 12);
        TestPod arr3[3] = {};
        {
            size_t res = mdataFilling.putPODs<TestPod, true>(arr3, 3);
            EXPECT_EQ(res, 3);
        }
        EXPECT_EQ(mdataFilling.getFilledSize(), 12+3*sizeof(arr3[0]));


        dopt::DataUniquePtr testData = dopt::DataUniquePtr(dopt::Data::getDataFromMutableData(&mdataFilling, true));
        EXPECT_EQ(testData->getTotalLength(), mdataFilling.getFilledSize());
        EXPECT_EQ(testData->getPtr(), mdataFilling.getPtr());
        EXPECT_EQ(testData->getResidualLength(), mdataFilling.getFilledSize());
        EXPECT_FALSE(testData->isRawMemoryBeenAllocated());

        dopt::DataUniquePtr testDataCopy = dopt::DataUniquePtr(dopt::Data::getDataFromMutableData(&mdataFilling, false));
        EXPECT_EQ(testDataCopy->getTotalLength(), mdataFilling.getFilledSize());
        EXPECT_NE(testDataCopy->getPtr(), mdataFilling.getPtr());
        EXPECT_EQ(testDataCopy->getResidualLength(), mdataFilling.getFilledSize());
        EXPECT_TRUE(testDataCopy->isRawMemoryBeenAllocated());
    }

    {
        dopt::MutableData mdata;
        char s[35] = {};
        EXPECT_TRUE(mdata.putByte(1));
        EXPECT_EQ(1, mdata.getFilledSize());            
        EXPECT_EQ(mdata.putBytes(s, sizeof(s)), sizeof(s));
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() > 0) << "Check correct chunk allocations";
        EXPECT_TRUE(mdata.getAllocBytesToStoreData() % dopt::MutableData::kChunkSizeIncrease == 0);

        EXPECT_EQ(36, mdata.getFilledSize());
    }
}


TEST(Data, MutableDataSpecialFillingGTest)
{
    dopt::MutableData mdata;
    mdata.putCharacter('H');
    mdata.putCharacter('e');
    mdata.putCharacter('l');
    mdata.putCharacter('l');
    mdata.putCharacter('o');
    mdata.putCharacter('\0');

    EXPECT_EQ(mdata.getFilledSize(), 6);
    EXPECT_TRUE( std::string((char*)mdata.getPtr()) == std::string("Hello") );
    mdata.seekCur(-1);
    
    EXPECT_TRUE(mdata.getCurWritePos() == 5);
    mdata.putIntegerAsAString(123);
    mdata.putIntegerAsAString(-456);
    mdata.putUnsignedIntegerAsAString(789);
    mdata.putUnsignedIntegerAsAString(0);
    mdata.putUnsignedIntegerAsAString(0);
    mdata.putIntegerAsAString(-0);
    mdata.putIntegerAsAString(-0);
    mdata.putIntegerAsAString(1);
    mdata.putIntegerAsAString(123456);
    mdata.putCharacter('\0');

    EXPECT_TRUE(std::string((char*)mdata.getPtr()) == std::string("Hello123-45678900001123456"));
}

size_t getNumberOfOnes(uint64_t value)
{
    size_t res = 0;

    for (;value > 0; value >>= 1)
    {
        if (value & 0x1)
        {
            res++;
        }
    }

    return res;
}

TEST(Data, MutableDataVaryingIntegersGTest)
{
    dopt::MutableData mdata;
    mdata.putCharacter('a');
    mdata.putCharacter('b');
    mdata.putCharacter('c');
    EXPECT_EQ(mdata.getFilledSize(), 3);

    mdata.seekStart(0);
    EXPECT_EQ(mdata.getFilledSize(), 0);

    for (uint64_t i = 0; i <= 127; ++i)
    {
        mdata.putUnsignedVaryingInteger(i);
        EXPECT_EQ(mdata.getFilledSize(), 1) << "Check that 7-bit values takes 1 bytes in the buffer";
        dopt::Data data(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        uint64_t ii = data.getUnsignedVaryingInteger();

        EXPECT_EQ(ii, i) << "Check that we correctly save and restore 7-bit values";
        mdata.seekStart(0);
    }

    for (uint64_t i = 128; i <= 16383 /*2**14 -1*/; ++i)
    {
        mdata.putUnsignedVaryingInteger(i);
        EXPECT_EQ(mdata.getFilledSize(), 2) << "Check that 14-bit values takes 2 bytes in the buffer";
        dopt::Data data(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        uint64_t ii = data.getUnsignedVaryingInteger();
        EXPECT_EQ(ii, i) << "Check that we correctly save and restore 14-bit values";
        mdata.seekStart(0);
    }

    for (uint64_t i = 16384; i <= 2097151 /*2**21 -1*/; ++i)
    {
        mdata.putUnsignedVaryingInteger(i);
        EXPECT_EQ(mdata.getFilledSize(), 3) << "Check that 21-bit values takes 3 bytes in the buffer";
        dopt::Data data(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        uint64_t ii = data.getUnsignedVaryingInteger();
        EXPECT_EQ(ii, i) << "Check that we correctly save and restore 21-bit values";
        mdata.seekStart(0);
    }

    for (uint64_t i = 2097152; i <= 268435455 /*2**28 -1*/; i += 13)
    {
        mdata.putUnsignedVaryingInteger(i);
        EXPECT_EQ(mdata.getFilledSize(), 4) << "Check that 28-bit values takes 4 bytes in the buffer";
        dopt::Data data(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        uint64_t ii = data.getUnsignedVaryingInteger();
        EXPECT_EQ(ii, i) << "Check that we correctly save and restore 28-bit values";
        mdata.seekStart(0);
    }
    
    std::vector<uint64_t> testItems = { 0x0ULL,
                                        0x0FULL,
                                        0xFFULL, 
                                        0xFFFULL,
                                        0xFFFFULL,
                                        0xFFFFFULL,
                                        0xFFFFFFULL,
                                        0xFFFFFFFULL,
                                        0xFFFFFFFFULL,
                                        0xFFFFFFFFFULL,
                                        0xFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFFFFFULL,
                                        0xFFFFFFFFFFFFFFFFULL};
    
    for (uint64_t value : testItems)    
    {
        size_t lenInBytesPrev = mdata.getFilledSize();
        mdata.putUnsignedVaryingInteger(value);
        size_t lenInBytesNew = mdata.getFilledSize();

        size_t putBytes = lenInBytesNew - lenInBytesPrev;
        size_t valueBits = getNumberOfOnes(value);
        
        EXPECT_TRUE(putBytes + 1e-6 > valueBits/7.0);
        EXPECT_TRUE(putBytes - 1e-6 < 1.0 + valueBits / 7.0);
    }

    dopt::Data dataTmp(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

    for (uint64_t value : testItems)
    {
        uint64_t valueReconstructed = dataTmp.getUnsignedVaryingInteger();
        EXPECT_EQ(valueReconstructed, value);
    }
    
    EXPECT_TRUE(dataTmp.isEmpty());

    {
        dopt::MutableData mdataCompileTime;
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(0)>();
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(1)>();
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(2)>();
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(125)>();
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(200)>();
        mdataCompileTime.putUnsignedVaryingIntegerKnowAtCompileTime<uint64_t, uint64_t(300)>();

        dopt::MutableData mdataRuntime;
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(0));
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(1));
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(2));
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(125));
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(200));
        mdataRuntime.putUnsignedVaryingInteger<uint64_t>(uint64_t(300));

        EXPECT_TRUE(mdataCompileTime.getFilledSize() == mdataRuntime.getFilledSize());
        for (size_t i = 0; i < mdataCompileTime.getFilledSize(); ++i)
        {
            EXPECT_TRUE(mdataCompileTime.getPtr()[i] == mdataRuntime.getPtr()[i]);
        }
    }
}

TEST(Data, MutableDataMatrixGTest)
{
    dopt::RandomGenRealLinear g(1.0, 10.0);
    std::vector<size_t> rows = { 1, 3, 5, 10, 20 };
    std::vector<size_t> cols = { 1, 5, 10, 20, 50 };

    for (size_t r : rows)
    {
        for (size_t c : cols)
        {
            dopt::MutableData mdata;
            dopt::MatrixNMD<dopt::VectorNDRaw_d> matrix(r, c);
            dopt::MatrixNMD<dopt::VectorNDRaw_d> matrixReconstruct = matrix;
            matrix.setAllRandomly(g);
            EXPECT_TRUE(mdata.putMatrixItems(matrix));
            dopt::Data data(mdata.getPtr(), mdata.getFilledSize(), dopt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            EXPECT_TRUE(data.getMatrixItems(matrixReconstruct));
            EXPECT_TRUE((matrix - matrixReconstruct).frobeniusNorm() < 1e-10);
            EXPECT_TRUE(data.getResidualLength() == 0);
        }
    }
}
