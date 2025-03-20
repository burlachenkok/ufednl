/** @file
* cross-platform memory bit stream
*/
#pragma once

#include "dopt/copylocal/include/MutableData.h"
#include "dopt/copylocal/include/Data.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stddef.h>
#include <memory.h>

namespace dopt
{
    /**@class MutableBitsData
    * @brief A container representing the ability to add bits into a buffer with a dynamically changing size
    * @details Bits are accumulated in the byte accumulator starting from LSB to MSB. When the byte is full, it is flushed into the byte buffer.
    */
    class MutableBitsData
    {
    private:
        MutableData* mdata;   ///< Buffer with completed bytes
        uint8_t rest;         ///< Rest byte with varying bit-size equal to restBitsSize
        uint8_t restBitsSize; ///< Number of filled bits in rest byte

    protected:
        /**
         * @brief Rewinds and adds bytes to the bit stream from the given source buffer.
         * @details This function rewinds the bit stream to its start position and adds bytes to it from the provided source buffer. It processes each byte in the source buffer, handling bits accumulation and flushing bytes when the internal buffer is full.
         * @param srcBuffer Pointer to the source buffer containing bytes to add to the bit stream.
         * @param srcBufferLenInBits Length of the source buffer in bits.
         * @return true if the operation is successful, false otherwise.
         */
        bool rewindAndPutBytesToBitStream(const void* restrict_ext srcBuffer, size_t srcBufferLenInBits)
        {
            rewindToStart();
            
            const uint8_t* restrict_ext srcBufferBytes = static_cast<const uint8_t*>(srcBuffer);

            if (srcBuffer == nullptr)
                return 0;

            size_t sizeInBytes = srcBufferLenInBits / 8;
            size_t bitsInLastByte = srcBufferLenInBits % 8;

            if (bitsInLastByte != 0)
            {
                sizeInBytes += 1;
            }
            else
            {
                bitsInLastByte = 8;
            }

            for (size_t i = 0; i < sizeInBytes; ++i)
            {
                uint8_t curByte = srcBufferBytes[i];
                size_t bitsNumberInByte = 8;

                if (i == sizeInBytes - 1)
                {
                    bitsNumberInByte = bitsInLastByte;
                }

                for (size_t j = 0; j < bitsNumberInByte; ++j)
                {
                    if (curByte & (0x1 << j))
                    {
                        rest |= (0x1 << restBitsSize);
                        ++restBitsSize;
                    }
                    else
                    {
                        restBitsSize++;
                    }

                    // Flush Byte
                    if (restBitsSize == 8)
                    {
                        mdata->putByte(rest);
                        rest = 0;
                        restBitsSize = 0;
                    }
                }
            }

            return true;
        }

        /**
         * @brief Puts bytes into a bit stream.
         * @details This method converts the provided source buffer of bits into a bit stream and stores it in the object's internal buffer.
         *
         * @param srcBuffer Pointer to the source buffer containing data to be added to the bit stream.
         * @param srcBufferLenInBits Length of the source buffer in bits.
         * @return true if the operation was successful, false otherwise.
         * @sa rewindAndPutBytesToBitStream
         */
        bool putBytesToBitStream(const void* restrict_ext srcBuffer, size_t srcBufferLenInBits)
        {
            const uint8_t* restrict_ext srcBufferBytes = static_cast<const uint8_t*>(srcBuffer);

            if (srcBuffer == nullptr)
                return 0;

            size_t sizeInBytes    = srcBufferLenInBits / 8;
            size_t bitsInLastByte = srcBufferLenInBits % 8;

            if (bitsInLastByte != 0)
            {
                sizeInBytes += 1;
            }
            else
            {
                bitsInLastByte = 8;
            }

            for (size_t i = 0; i < sizeInBytes; ++i)
            {
                uint8_t curByte = srcBufferBytes[i];
                size_t bitsNumberInByte = 8;
                
                if (i == sizeInBytes - 1)
                {
                    bitsNumberInByte = bitsInLastByte;
                }
                
                for (size_t j = 0; j < bitsNumberInByte; ++j)
                {
                    if (curByte & (0x1 << j))
                    {
                        rest |= (0x1 << restBitsSize);
                        ++restBitsSize;
                    }
                    else
                    {
                        restBitsSize++;
                    }

                    // Flush Byte
                    if (restBitsSize == 8)
                    {
                        mdata->putByte(rest);
                        rest = 0;
                        restBitsSize = 0;
                    }
                }
            }

            return true;
        }


        /**
        * @brief Inserts a bit pattern into the bit stream.
        * @details This function accumulates bits from the given bit pattern into a buffer, managing buffer size dynamically.
        *          Each byte is flushed into the data storage once it is filled.
        *
        * @param bitPattern The bit pattern to be inserted into the bit stream.
        * @param srcBufferLenInBits The length of the source buffer in bits.
        *
        * @return Returns true on successful insertion of the bit pattern.
        */
        bool putBitToBitStream(bool bitPattern, size_t srcBufferLenInBits)
        {
            size_t sizeInBytes = srcBufferLenInBits / 8;
            size_t bitsInLastByte = srcBufferLenInBits % 8;

            if (bitsInLastByte != 0)
            {
                sizeInBytes += 1;
            }
            else
            {
                bitsInLastByte = 8;
            }

            for (size_t i = 0; i < sizeInBytes; ++i)
            {
                size_t bitsNumberInByte = 8;

                if (i == sizeInBytes - 1)
                {
                    bitsNumberInByte = bitsInLastByte;
                }

                for (size_t j = 0; j < bitsNumberInByte; ++j)
                {
                    if (bitPattern)
                    {
                        rest |= (0x1 << restBitsSize);
                        ++restBitsSize;
                    }
                    else
                    {
                        restBitsSize++;
                    }

                    // Flush Byte
                    if (restBitsSize == 8)
                    {
                        mdata->putByte(rest);
                        rest = 0;
                        restBitsSize = 0;
                    }
                }
            }

            return true;
        }
        
    public:
        /**
        * @brief Retrieves a unique pointer to a Data object containing accumulated bits.
        * @details This function checks if the byte buffer is complete and all bits are properly padded.
        * If the buffer is empty, it returns a DataUniquePtr pointing to an empty Data object.
        * Otherwise, it allocates a new Data object copying the contents of the buffer and returns it as a DataUniquePtr.
        *
        * @return dopt::DataUniquePtr A unique pointer to the Data object containing the accumulated and padded bits.
        * @remark This method allocated memory storate and copy content into it.
        * @warning This method asserts if the byte buffer is incomplete or has unpadded bits.
        */
        dopt::DataUniquePtr getData()
        {
            if (!isByteBufferComplete())
            {
                assert(!"Can not receive DataAutoPtr from MutableBitsData due to the fact that exist not padded bits in end of buffer!");
                return dopt::DataUniquePtr();
            }

            if (mdata->isEmpty())
                return dopt::DataUniquePtr(new dopt::Data(nullptr, 0));

            size_t srcLen = mdata->getFilledSize();
            uint8_t* srcData = mdata->getPtr();

            return  dopt::DataUniquePtr(new dopt::Data(srcData,
                                                       srcLen,
                                                       dopt::Data::MemInitializedType::eAllocAndCopy));
        }


        /**
         * @brief Constructs a MutableBitsData object with a given MutableData instance.
         * @param theMData A pointer to the MutableData instance used for bit accumulation.
         */
        MutableBitsData(MutableData* theMData)
        : mdata(theMData)
        , rest(0)
        , restBitsSize(0)
        {}

        /**
         * @brief Copy constructor for MutableBitsData.
         * @details Initializes a new instance of the MutableBitsData class by copying the data from an existing instance.
         * @param rhs A constant reference to the MutableBitsData object to be copied.
         * @return A new instance of MutableBitsData with the same data as the provided object.
         */
        MutableBitsData(const MutableBitsData& rhs)
        : mdata(rhs.mdata)
        , rest(rhs.rest)
        , restBitsSize(rhs.restBitsSize)
        {
        }

        MutableBitsData& operator = (const MutableBitsData& rhs)
        {
            mdata = rhs.mdata;
            rest = rhs.rest;
            restBitsSize = rhs.restBitsSize;
            return *this;
        }

        /**
         * @brief Reserves a specified amount of memory.
         * @details This method requests a memory reservation of the given size in bytes.
         * @param sizeInBytes The amount of memory to reserve, in bytes.
         * @return Returns true if the memory was successfully reserved, otherwise false.
         */
        bool reserveMemory(size_t sizeInBytes) {
            return mdata->reserveMemory(sizeInBytes);
        }
        
        /** Is container empty. I.e. nobody puh nothing into it.
        * @return true if container is empty.
        */
        bool isEmpty() const
        {
            return mdata->isEmpty() && restBitsSize == 0;
        }

        /** Get current buffer size, allocated from the heap to store data
        * @return number of bytes in heap allocated to store buffer
        * @remark runnng byte is not included into this capacity
        */
        size_t getAllocBytesToStoreData() const {
            return mdata->getAllocBytesToStoreData();
        }

        /** Put byte into used buffer stream
        * @param val byte, which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putByte(uint8_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put single character into used buffer stream
        * @param val character (one byte size) which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putCharacter(char val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put single float into used buffer stream
        * @param single float, which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putFloat(float val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put a bit into the buffer
        * @param val bit value
        * @return the result of the operation
        */
        bool putBit(uint8_t val) {
            return putBytesToBitStream(&val, 1);
        }

        /** Put some bits into the buffer. If there are bits that do not cover a byte, then the least significant bits from this byte are added.
        * @param buffer byte buffer
        * @param bitsNumber buffer length in bits
        * @return the result of the operation
        */
        bool putBits(const void* buffer, size_t bitsNumber) {
            return putBytesToBitStream(buffer, bitsNumber);
        }

        template <class Tx, class Ty>
        /**
         * @brief Inserts two trivially copyable objects into a bit stream.
         * @param x The first trivially copyable object to be inserted.
         * @param y The second trivially copyable object to be inserted.
         * @return True if the objects are successfully inserted into the bit stream.
         */
        bool putTuple2(const Tx& x, const Ty& y)
        {
            static_assert(std::is_trivially_copyable<Tx>::value, "For such low level manipulation only POD types are allowed as type for x");
            static_assert(std::is_trivially_copyable<Ty>::value, "For such low level manipulation only POD types are allowed as type for y");

            putBytesToBitStream(&x, sizeof(x) * 8);
            putBytesToBitStream(&y, sizeof(y) * 8);
            return true;
        }
        
        template <class Tx, class Ty, class Tz>
        /**
         * @brief Inserts three elements into a bit stream.
         * @details This function accepts three trivially copyable elements and inserts their byte representations into a bit stream.
         * @param x The first element to be inserted into the bit stream.
         * @param y The second element to be inserted into the bit stream.
         * @param z The third element to be inserted into the bit stream.
         * @return True if the operation was successful.
         */
        bool putTuple3(const Tx& x, const Ty& y, const Tz& z)
        {
            static_assert(std::is_trivially_copyable<Tx>::value, "For such low level manipulation only POD types are allowed as type for x");
            static_assert(std::is_trivially_copyable<Ty>::value, "For such low level manipulation only POD types are allowed as type for y");
            static_assert(std::is_trivially_copyable<Tz>::value, "For such low level manipulation only POD types are allowed as type for z");

            putBytesToBitStream(&x, sizeof(x) * 8);
            putBytesToBitStream(&y, sizeof(y) * 8);
            putBytesToBitStream(&z, sizeof(z) * 8);

            return true;
        }

        /**
         * @brief Inserts four POD (plain old data) type elements into a bit stream.
         * @details This method takes four elements of possibly different trivially copyable types and inserts their byte representations into a bit stream.
         * The method ensures that each type is trivially copyable to maintain low-level manipulation restrictions.
         *
         * @param x Element of type Tx to be added to the bit stream.
         * @param y Element of type Ty to be added to the bit stream.
         * @param z Element of type Tz to be added to the bit stream.
         * @param w Element of type Tw to be added to the bit stream.
         * @return Returns true if all elements are successfully added to the bit stream.
         */
        template <class Tx, class Ty, class Tz, class Tw>
        bool putTuple4(const Tx& x, const Ty& y, const Tz& z, const Tw& w)
        {
            static_assert(std::is_trivially_copyable<Tx>::value, "For such low level manipulation only POD types are allowed as type for x");
            static_assert(std::is_trivially_copyable<Ty>::value, "For such low level manipulation only POD types are allowed as type for y");
            static_assert(std::is_trivially_copyable<Tz>::value, "For such low level manipulation only POD types are allowed as type for z");
            static_assert(std::is_trivially_copyable<Tw>::value, "For such low level manipulation only POD types are allowed as type for w");

            putBytesToBitStream(&x, sizeof(x) * 8);
            putBytesToBitStream(&y, sizeof(y) * 8);
            putBytesToBitStream(&z, sizeof(z) * 8);
            putBytesToBitStream(&w, sizeof(w) * 8);

            return true;
        }

        /** Put double into used buffer stream
        * @param val double value which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putDouble(double val) 
        {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Rewind write position and put double into used buffer stream
        * @param val double value which will be pushed into buffer
        * @return true - pushing completed successfully
        * @remark Please use if you understand what you're doing. If you don't understand please use putDouble()
        */
        bool rewindAndPutDouble(double val) {
            return rewindAndPutBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put int8_t value into used buffer stream
       * @param val item which will be pushed into buffer
       * @return true - pushing completed successfully
       */
        bool putInt8(int8_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put int16_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt16(int16_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put uint8_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint8(uint8_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put uint16_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint16(uint16_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put int32_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt32(int32_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put uint32_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint32(uint32_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put uint64_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint64(uint64_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }

        /** Put int64_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt64(int64_t val) {
            return putBytesToBitStream(&val, sizeof(val) * 8);
        }
        
        /** Put uint64_t value into used buffer stream as varying size integer. Therefor actual size in bytes can be 1-9 bytes long.
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        template<class TIntType>
        /**
         * @brief Encodes an unsigned integer in varying length format and writes it into the bit stream.
         * @details The method extracts 7 bits at a time from the integer and writes them,
         *          along with a continuation flag if (1) in high order bit
         *          more bits remain, into the bit stream until the entire integer is encoded.
         *
         * @param val The unsigned integer to encode and write into the bit stream.
         * @return true if the encoding and writing process is completed successfully.
         */
        bool putUnsignedVaryingInteger(TIntType val)
        {
            for (;;)
            {
                // extract 7 LSB bits
                uint8_t value2Write = (uint8_t)(val & 0b0111'1111);
                val >>= 7;

                if (val == 0)
                {
                    // last byte is value2Write
                    putBytesToBitStream(&value2Write, 8);
                    break;
                }
                else
                {
                    // flush 1bit flag and 7 payload bits
                    value2Write |= 0b1000'0000;
                    putBytesToBitStream(&value2Write, 8);
                }
            }
            return true;
        }

        /**
        * @brief Encodes an unsigned integer known at compile time into a varying-length format.
        * @details This method optimizes the encoding of unsigned integers by determining the appropriate encoding strategy at compile time. If the input value is less than 128, it is encoded as a single byte. Otherwise, a variable-length encoding technique is used.
        * @return true if the encoding is successful, false otherwise.
         */
        template<class TIntType, TIntType value>
        bool putUnsignedVaryingIntegerKnowAtCompileTime()
        {
            if constexpr (value < 128)
                return putUint8((uint8_t)value);
            else
                return putUnsignedVaryingInteger(value);
        }

        
        /** Size of the filled area in bits. Or the offset of the current record position relative to the beginning of the buffer.
        * @return the requested value
        */
        size_t getFilledSizeInBits() const
        {
            size_t total = 0;
            total += restBitsSize;
            total += mdata->getFilledSize() * 8;
            return total;
        }

        /** Whether the byte buffer is formed wihtout any residual
        * @return true if the byte buffer is formed 
        */
        bool isByteBufferComplete() const {
            return restBitsSize == 0;
        }

        /** Setup write position to the start of byte stream
        */
        void rewindToStart() 
        {
            mdata->rewindToStart();
            rest = 0;
            restBitsSize = 0;
        }
        
        /** Flags which controls how exactly string is placed into output buffer
        */
        enum class PutStringFlags : std::int8_t
        {
            ePutNoTerminator,
            ePutNewLine,
            ePutZeroTerminator
        };

        /** Put string content into mutable data object.
        * @param str string to put. Only content with any leading termination
        * @param flag information about terminator that you want to put with the conent.
        * @return operation result
        * @sa PutStringFlags
        */
        bool putString(std::string_view str, PutStringFlags flag = PutStringFlags::ePutZeroTerminator)
        {
            putBits(str.data(), str.size() * sizeof(std::string_view::value_type));

            switch (flag)
            {
            case PutStringFlags::ePutNoTerminator:
                break;
            case PutStringFlags::ePutNewLine:
                putByte('\n');
                break;
            case PutStringFlags::ePutZeroTerminator:
                putByte('\0');
                break;
            default:
                assert(!"This case of 'putString' have not been processed by the current implementation");
                break;
            }

            return true;
        }
    };

    typedef std::unique_ptr<MutableBitsData> MutableBitsDataUniquePtr;
}
