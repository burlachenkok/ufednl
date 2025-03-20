#include "dopt/copylocal/include/Data.h"
#include "dopt/copylocal/include/MutableData.h"

#include <ctype.h>
#include <sstream>

namespace dopt
{
    bool Data::isOnlyAsciiInside() const
    {
        for (size_t i = 0; i < length; i++)
        {
            const unsigned char ch = bits[i];

            // all chars ASCII are ok
            if (ch >= 32 && ch < 128)
                continue;

            // these are also
            if (ch == '\n' || ch == '\r' || ch == '\t')
                continue;

            // check for locale characters
            if (isalpha(ch))
                continue;

            return false;
        }

        return true;
    }

    size_t Data::getBytes(void* restrict_ext ptr, size_t len, bool moveDataWindow)
    {
        if (isEmpty())
            return 0;

        if (pos + len > length)
            len = length - pos;

        if (ptr)
        {
            memcpy(ptr, bits + pos, len);
        }

        if (moveDataWindow)
            pos += len;

        return len;
    }

    std::string Data::getString()
    {
        if (isEmpty())
            return std::string();

        std::stringstream stream;

        constexpr char skipChars[] = { '\r' };
        constexpr char endOFStringChar[] = { '\n', '\0' };

        for (; pos < length; pos++)
        {
            bool endOFStringCharFound = false;
            for (size_t j = 0; j < sizeof(endOFStringChar) / sizeof(endOFStringChar[0]); ++j)
            {
                if (bits[pos] == endOFStringChar[j])
                {
                    pos++;
                    endOFStringCharFound = true;
                    break;
                }
            }

            if (endOFStringCharFound)
                break;

            bool skipCharsFound = false;
            for (size_t j = 0; j < sizeof(skipChars) / sizeof(skipChars[0]); ++j)
            {
                if (bits[pos] == skipChars[j])
                {
                    skipCharsFound = true;
                    break;
                }
            }

            if (!skipCharsFound)
            {
                stream << bits[pos];
            }
        }
        return stream.str();
    }

    Data* Data::getDataFromMutableData(const MutableData* m, bool shareMDataPtr)
    {
        if (shareMDataPtr)
        {
            Data* result = new Data(m->getPtr(), m->getFilledSize(), MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            return result;
        }
        else
        {
            Data* result = new Data(m->getPtr(), m->getFilledSize(), MemInitializedType::eAllocAndCopy);
            return result;
        }
    }
}
