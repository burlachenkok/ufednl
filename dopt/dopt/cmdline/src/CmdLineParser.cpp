#include "CmdLineParser.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/fs/include/StringUtils.h"

#include <sstream>
#include <string_view>
#include <assert.h>

namespace
{
    /** Auxiliary help function for string comparison ignoring first one or two dashes
    * @param aASCIIZ one string
    * @param bASCIIZ another string
    * @return true if strings are equal
    */
    bool equalIgnorePrefix(const char* aASCIIZ, const char* bASCIIZ)
    {
        // Ignore first one or first two dashes
        if (*aASCIIZ == '-')
            aASCIIZ++;

        if (*aASCIIZ == '-')
            aASCIIZ++;

        // Ignore first one or first two dashes
        if (*bASCIIZ == '-')
            bASCIIZ++;

        if (*bASCIIZ == '-')
            bASCIIZ++;

        for (;; aASCIIZ++, bASCIIZ++)
        {
            if (*aASCIIZ != *bASCIIZ)
            {
                // Not equal
                // Case-1: different symbols
                // Case-2: a has been finished and b is not
                // Case-3: b has been finished and a is not
                return false;
            }
            else
            {
                // Symbols are equal
                // Case-1: Strings are equal and was finished in last symbol (Zero)
                // Case-2: String are equal for now, but "*a" is not zero.
                //         because "*b=*a" => *b is also not zero
                if (*aASCIIZ == 0)
                {
                    return true;
                }
            }
        }

        return true;
    }

    /**
     * Auxiliary help function for string comparison ignoring first one or two dashes.
     * @param aData one string not necessary finishing with zero
     * @param aLen length of the first string
     * @param bASCIIZ another string in ASCIIZ format necessary finishing with zer
     * @return true if strings are equal, otherwise false
     */
    bool equalIgnorePrefix(const char* aData, size_t aLen, const char* bASCIIZ)
    {
        // Ignore first one or first two dashes
        if (*aData == '-')
        {
            if (aLen == 0)
                return false;
            
            aData++;
            aLen--;
        }

        if (*aData == '-')
        {
            if (aLen == 0)
                return false;

            aData++;
            aLen--;
        }

        // Ignore first one or first two dashes
        if (*bASCIIZ == '-')
            bASCIIZ++;

        if (*bASCIIZ == '-')
            bASCIIZ++;

        for (;; aData++, bASCIIZ++, aLen--)
        {
            if (aLen == 0)
            {
                if (*bASCIIZ == 0)
                    return true;
                else
                    return false;
            }

            if (*aData != *bASCIIZ)
            {
                // Not equal
                // Case-1: different symbols
                // Case-2: a has been finished and b is not
                // Case-3: b has been finished and a is not
                return false;
            }
        }

        return true;
    }

    /** Auxiliary help function for string comparision ignoring first one or two dashes
    * @param a one string in std::string_view format
    * @param b another string in ASCIIZ
    * @return true if strings are equal
    */
    bool equalIgnorePrefix(std::string_view a, const char* b)
    {
        return equalIgnorePrefix(a.data(), a.size(), b);
    }
}

namespace dopt
{
    CmdLine::CmdLine(int theArgc, char** theArgv)
    : argc(theArgc)
    , argv(theArgv)
    {
#ifdef USE_TRIE_IN_CMDLINE_PARSER
        // Fill search trie
        for (int i = 1; i < theArgc; ++i)
        {
            char* curParameter = argv[i];
            
            // Skip parameters that are not started with '-' sign.
            if (curParameter[0] != '-')
                continue;
            
            if (*curParameter == '-')
                curParameter++;
            if (*curParameter == '-')
                curParameter++;

            if (i + 1 < theArgc)
                m_arguments.put(curParameter, i + 1);
            else
                m_arguments.put(curParameter, -1);
        }
#endif
    }

    int CmdLine::getArgCount() const {
        return argc;
    }

    std::string_view CmdLine::getArgumentByIndex(int index) const
    {
        if (index >= 0 && index < argc)
            return argv[index];
        else
            return std::string_view();
    }

    int CmdLine::getArgumentIndexByName(std::string_view argumentName) const
    {
        assert(argumentName.size() > 0 && argumentName[0] != '-');
        
#ifdef USE_TRIE_IN_CMDLINE_PARSER
        int* value = m_arguments.get(argumentName);

        if (value == nullptr)
        {
            return -1;
        }
        else
        {
            return *value;
        }
#else
        int myArgc = argc;

        for (int i = 1; i < myArgc; ++i)
        {
            // Skip parameters that are not started with '-' sign.
            if (argv[i][0] != '-')
                continue;

            if (equalIgnorePrefix(argumentName, argv[i]))
            {
                if (i + 1 < myArgc)
                    return i + 1;
                else
                    return -1;
            }
        }

        return -1;
#endif
    }

    bool CmdLine::getStringArgByName(std::string& argumentValue, std::string_view argumentName) const
    {
        int index = getArgumentIndexByName(argumentName);

        if (index < 0)
            return false;

        argumentValue = getArgumentByIndex(index);

        return true;
    }

    bool CmdLine::getStringViewArgByName(std::string_view& argumentValue, std::string_view argumentName) const
    {
        int index = getArgumentIndexByName(argumentName);

        if (index < 0)
            return false;

        argumentValue = getArgumentByIndex(index);

        return true;
    }

    bool CmdLine::getFloatArgByName(float& argumentValue, std::string_view argumentName) const
    {
        std::string_view argValueStr;

        if (!getStringViewArgByName(argValueStr, argumentName))
            return false;

        if (!dopt::string_utils::fromString(argumentValue, argValueStr))
            return false;

        return true;
    }

    bool CmdLine::getDoubleArgByName(double& argumentValue, std::string_view argumentName) const
    {
        std::string_view argValueStr;

        if (!getStringViewArgByName(argValueStr, argumentName))
            return false;

        if (!dopt::string_utils::fromString(argumentValue, argValueStr))
            return false;

        return true;
    }

    bool CmdLine::getIntArgByName(int& argumentValue, std::string_view argumentName) const
    {
        std::string_view argValueStr;

        if (!getStringViewArgByName(argValueStr, argumentName))
            return false;

        if (!dopt::string_utils::fromString(argumentValue, argValueStr))
            return false;

        return true;
    }

    bool CmdLine::getUnsignedArgByName(uint32_t& argumentValue, std::string_view argumentName) const
    {
        std::string_view argValueStr;

        if (!getStringViewArgByName(argValueStr, argumentName))
            return false;

        if (!dopt::string_utils::fromString(argumentValue, argValueStr))
            return false;

        return true;
    }
    
    bool CmdLine::isFlagSetuped(std::string_view flagName) const
    {
        assert(flagName.size() > 0 && flagName[0] != '-');
        
#ifdef USE_TRIE_IN_CMDLINE_PARSER
        int* nextIndex = m_arguments.get(flagName);

        // Flag has not been found
        if (nextIndex == nullptr)
            return false;
        else        
            return true;

        // No check that next parameter is not "off" "0" or "no"
#else
        int myArgc = argc;
        
        for (int i = 1; i < myArgc; ++i)
        {
            if (equalIgnorePrefix(flagName, argv[i]))
                return true;
        }
        
        return false;
#endif
    }
}
