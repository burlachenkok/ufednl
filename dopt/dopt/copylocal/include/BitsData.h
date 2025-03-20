/** @file
 * C++2003 cross-platform raw data access at level of bits
 */

#pragma once

#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/MutableData.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <stddef.h>
#include <memory.h>
#include <assert.h>

#include <utility>
#include <sstream>

namespace dopt
{
    /**@class BitsData
    * @brief A container representing the ability to receive bits
    * @details Bytes are pumped out of the mdata byte store in the order in which they are stored there. Bits from a byte are pumped out of the accumulator starting from LSB to MSB.
    */
    class BitsData
    {
    private:
        Data* mdata;
        uint8_t lastExtractedByte; ///< Last extracted byte from byte storage
        uint8_t nextBitToExtract;  ///< Next bit to extract
        size_t posInBits;          ///< Current bit position in the stream
        size_t lenInBits;          ///< Length of the available bit buffer in bits

    protected:
        /**
        * @brief Extracts a specified number of bits from a byte stream and stores them in the output buffer.
        * @details The function extracts bits from a byte stream one by one, stores them in an accumulator,
        * and writes to the output buffer when the accumulator is full or when the specified number of bits has been processed.
        * @param output A pointer to the buffer where the extracted bits will be stored.
        * @param bitsNumber The number of bits to extract from the byte stream.
        * @return Returns true if the bits are successfully extracted and stored in the output buffer, false otherwise.
        */
        bool getValueFromStream(uint8_t* restrict_ext output, size_t bitsNumber)
        {
            uint8_t outputAccum = 0;
            uint8_t outputAccumNextPos = 0;

            for (size_t i = 0; i < bitsNumber; ++i)
            {
                // It's time to receive next byte from byte storage
                if (nextBitToExtract == 8)
                {
                    if (mdata->isEmpty())
                    {
                        assert(!"No more bytes in byte storage!");
                        return false;
                    }
                    else
                    {
                        lastExtractedByte = mdata->getByte();
                        nextBitToExtract = 0;
                    }
                }

                if (lastExtractedByte & (0x1 << (nextBitToExtract)))
                {
                    outputAccum |= (0x1 << outputAccumNextPos);
                }
                else
                {
                    /* do nothing with output accum */
                }

                nextBitToExtract++;
                outputAccumNextPos++;
                posInBits++;

                // It is time to flush byte to output buffer because 
                //   (1) We filled accumulator
                //      OR
                //   (2) It's a last iteration and we need flush some result, even with not full complete accumulator
                if (outputAccumNextPos == 8 || i == bitsNumber - 1)
                {
                    *(output++) = outputAccum;
                    outputAccum = 0;
                    outputAccumNextPos = 0;
                }
            }

            return true;
        }


    public:
        /**
        * @brief Constructs a BitsData object with the provided data.
        * @param theData The data to initialize the BitsData object with.
        * Initializes various internal states for bit extraction.
        */
        BitsData(Data* theData)
        : mdata(theData)
        , lastExtractedByte(0)
        , nextBitToExtract(8)
        , posInBits(0)
        , lenInBits(theData->getTotalLength() * 8)
        {
        }

        /**
        * @brief Copy constructor for BitsData
        * @details Initializes a new BitsData object by copying the values from an existing BitsData object.
        * @param rhs The BitsData object to copy from.
        */
        BitsData(const BitsData& rhs)
        : mdata(rhs.mdata)
        , lastExtractedByte(rhs.lastExtractedByte)
        , nextBitToExtract(rhs.nextBitToExtract)
        , posInBits(rhs.posInBits)
        , lenInBits(rhs.lenInBits)
        {
        }

        /**
        * @brief Move constructor for the BitsData class
        * @param rhs The BitsData object to be moved
        * @return This method does not return a value as it is a constructor
        */
        BitsData(BitsData&& rhs) noexcept
        : mdata(std::move(rhs.mdata))
        , lastExtractedByte(rhs.lastExtractedByte)
        , nextBitToExtract(rhs.nextBitToExtract)
        , posInBits(rhs.posInBits)
        , lenInBits(rhs.lenInBits)
        {
        }

        BitsData& operator = (const BitsData& rhs)
        {
            mdata = rhs.mdata;
            lastExtractedByte = rhs.lastExtractedByte;
            nextBitToExtract = rhs.nextBitToExtract;
            posInBits = rhs.posInBits;
            lenInBits = rhs.lenInBits;
            return *this;
        }

        BitsData& operator = (BitsData&& rhs) noexcept
        {
            mdata = std::move(rhs.mdata);
            lastExtractedByte = rhs.lastExtractedByte;
            nextBitToExtract = rhs.nextBitToExtract;
            posInBits = rhs.posInBits;
            lenInBits = rhs.lenInBits;
            return *this;
        }

        /** Check that container is empty, i.e. no bits arer available
        * @return true if no bits are available
        */
        bool isEmpty() const
        {
            return posInBits == lenInBits;
        }

        /** Total length in bits of container without take into account seeked offset
        * @return number of bits
        */
        size_t getTotalLengthIbBits() const {
            return lenInBits;
        }

        /** Residual number of bits in stream
        * @return length in bits
        */
        size_t getResidualLengthInBits() const {
            return lenInBits - posInBits;
        }

        /** Get the current offset (starting from zero) in the container where the next beat will come from
        * @return buffer length
        */
        size_t getPosInBits() const {
            return posInBits;
        }
        
        /** Get the next "character" from the buffer
        * @return obtained value
        */
        char getCharacter() {
            char returnedSymbol;
            getValueFromStream((uint8_t*) &returnedSymbol, sizeof(returnedSymbol) * 8);
            return returnedSymbol;
        }

        /** Get the next "byte" from the buffer
        * @return obtained value
        */
        uint8_t getByte() {
            uint8_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "float" from the buffer
        * @return obtained value
        */
        float getFloat() {
            float output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "double" from the buffer
        * @return obtained value
        */
        double getDouble() {
            double output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "signed short" from the buffer
        * @return obtained value
        */
        int16_t getInt16()
        {
            int16_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }


        /** Get the next "unsigned short" from the buffer
        * @return obtained value
        */
        uint16_t getUint16()
        {
            uint16_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "signed integer 32 bit" from the buffer
        * @return obtained value
        */
        int32_t getInt32() {
            int32_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "unsigned integer 32 bit" from the buffer
        * @return obtained value
        */
        uint32_t getUint32() {
            uint32_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "signed integer 64 bit" from the buffer
        * @return obtained value
        */
        int64_t getInt64() {
            int64_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get the next "unsigned integer 64 bit" from the buffer
        * @return obtained value
        */
        uint64_t getUint64() {
            uint64_t output;
            getValueFromStream((uint8_t*)&output, sizeof(output) * 8);
            return output;
        }

        /** Get 2 elements with type which is standard layout type from data stream.
        * @param x reference to container which will obtain 1st value
        * @param y reference to container which will obtain 2nd value
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTuple2(T& x, T& y)
        {
            if (getResidualLengthInBits() < sizeof(T) * 2 * 8)
                return false;
            getValueFromStream((uint8_t*)&x, sizeof(T) * 8);
            getValueFromStream((uint8_t*)&y, sizeof(T) * 8);
            return true;
        }

        /** Get 3 elements with type which is standard layout type from data stream.
        * @param x reference to container which will obtain 1st value
        * @param y reference to container which will obtain 2nd value
        * @param z reference to container which will obtain 3rd value
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTuple3(T& x, T& y, T& z)
        {
            if (getResidualLengthInBits() < sizeof(T) * 3 * 8)
                return false;
            getValueFromStream((uint8_t*)&x, sizeof(T) * 8);
            getValueFromStream((uint8_t*)&y, sizeof(T) * 8);
            getValueFromStream((uint8_t*)&z, sizeof(T) * 8);

            return true;
        }
        
        /** Get n elements with type which is standard layout type from data stream.
        * @param container pointer to data storage
        * @param n number of elements of type T which you want to obtain
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTupleN(T* restrict_ext container, size_t n)
        {
            if (getResidualLengthInBits() < sizeof(T) * n * 8)
                return false;

            for (size_t i = 0; i < n; ++i)
                getValueFromStream((uint8_t*)(& container[i]), sizeof(T) * 8);

            return true;
        }

        /**
        * @brief Extracts a single bit from the byte stream.
        * @details Retrieves one byte from the stream, extracts a single bit from it, and returns the bit.
        * @return A single bit from the byte stream as an 8-bit unsigned integer.
        */
        uint8_t getBit()
        {
            uint8_t output;
            getValueFromStream(&output, 1);
            return output;
        }

        /**
        * @brief Checks if the buffer contains only bytes, and there are no residual bits.
        * @details This method determines if the next bit to extract from the buffer
        * is at the start of a new byte.
        * @return true if a next bit is at the start of the next byte), otherwise false.
        */
        bool isOnlyByteBuffer() const {
            return nextBitToExtract == 8;
        }

        /** Setup read position to the start of byte stream
        */
        void rewindToStart() 
        {
            nextBitToExtract = 8;
            lastExtractedByte = 0;
            posInBits = 0;
            mdata->rewindToStart();
        }

        /**
        * @brief Extract bits from the bit stream and place them into the destination buffer.
        * @param dstByteBuffer Pointer to the destination byte buffer where the extracted bits will be stored.
        * @param dstByteBufferLenInBits Number of bits to extract from the bit stream and store in the destination buffer.
        */
        void getBits(void* dstByteBuffer, size_t dstByteBufferLenInBits)
        {
            getValueFromStream((uint8_t*)dstByteBuffer, dstByteBufferLenInBits);
        }
        
        /** Obtain len bytes from stream starting from current position relative to current Data::pos
        * @param dstBuffer buffer receiver
        * @param dstBufferLen size of dst buffer in bytes
        * @param moveDataWindow if true then after obtaining dstBufferLen the current read position is moving. if false - it not moves.
        * @return number of real obtained bytes which are less or equal to dstBufferLen
        */
        size_t getBytes(void* restrict_ext dstBuffer, size_t dstBufferLen)
        {
            getValueFromStream((uint8_t*)dstBuffer, dstBufferLen * 8);
            return dstBufferLen;
        }

        /** Obtain objectsNumber "POD" or "Standard Layout" objects from byte stream.
        * @param dstBuffer buffer receiver.
        * @param objectsNumber number of objects you want to obtain.
        * @param moveDataWindow if true then after obtaining dstBufferLen the current read position is moving. if false - it not moves.
        * @return number of real obtained objects which are less or equal to objectsNumber
        */
        template <class T>
        size_t getObjects(T* restrict_ext dstBuffer, size_t objectsNumber = 1)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            const size_t bytesNumber = getBytes(dstBuffer, sizeof(T) * objectsNumber);
            return bytesNumber / sizeof(T);
        }

        /** Obtain ASCII string from byte stream. String is terminating with '\0' or with '\n' and '\r' characters are skipping.
        * @return obtained string
        */
        std::string getString()
        {
            if (isEmpty())
                return std::string();

            std::stringstream stream;

            constexpr char skipChars[] = { '\r' };
            constexpr char endOFStringChar[] = { '\n', '\0' };

            for (;;)
            {
                char curSymbol = getCharacter();
                
                bool endOFStringCharFound = false;
                for (size_t j = 0; j < sizeof(endOFStringChar) / sizeof(endOFStringChar[0]); ++j)
                {
                    if (curSymbol == endOFStringChar[j])
                    {
                        endOFStringCharFound = true;
                        break;
                    }
                }

                if (endOFStringCharFound)
                    break;

                bool skipCharsFound = false;
                for (size_t j = 0; j < sizeof(skipChars) / sizeof(skipChars[0]); ++j)
                {
                    if (curSymbol == skipChars[j])
                    {
                        skipCharsFound = true;
                        break;
                    }
                }

                if (!skipCharsFound)
                {
                    stream << curSymbol;
                }
            }
            return stream.str();
        }
#if 0
         static BitsData* getDataFromMutableData(const MutableData* m, bool shareMDataPtr);
#endif

    };

    typedef std::unique_ptr<BitsData> BitsDataAutoPtr;
}
