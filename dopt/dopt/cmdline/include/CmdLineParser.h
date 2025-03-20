/** @file
* C++ cross-platform implementation of useful command line parsing mechanisms
*/

#pragma once

#ifndef USE_TRIE_IN_CMDLINE_PARSER
    // #define USE_TRIE_IN_CMDLINE_PARSER 1
#endif

#ifdef USE_TRIE_IN_CMDLINE_PARSER
    #include "dopt/math_routines/include/TernaryTrie.h"
#endif

#include <string_view>
#include <string>
#include <stdint.h>

namespace dopt
{
    /** Command line wrapper
    */
    class CmdLine
    {
    public:
        /** Ctor
        * @param theArgc number of arguments obtained from the entry point to CRT main()
        * @param theArgv argument which has been obtained from the entry point to CRT main()
        */
        CmdLine(int theArgc, char** theArgv);

        /** Get arguments count
        * @return get the number of arguments passed in the command line
        */
        int getArgCount() const;

        /* Get argument by index
        * @param index passed index
        * @return copy of string which represent argument at "index".
        * @remark If "index" is not correct function returns empty string
        */
        std::string_view getArgumentByIndex(int index) const;

        /** Find argumentName in command line which can be passed in command line as "argumentName value" or "-argumentName value" or "--argumentName value"
        * @param argumentName name of argument
        * @return index for value within a command line arguments list or -1 if argument has not been found
        * @remark argumentName is case sensitive, so be careful with capital case and lower case symbols in flags
        */
        int getArgumentIndexByName(std::string_view argumentName) const;

        /** Get string argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getStringArgByName(std::string& argumentValue, std::string_view argumentName) const;

        /** Get string argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark Use it if you know what you are doing
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getStringViewArgByName(std::string_view& argumentValue, std::string_view argumentName) const;

        /** Get float argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getFloatArgByName(float& argumentValue, std::string_view argumentName) const;

        /** Get double argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getDoubleArgByName(double& argumentValue, std::string_view argumentName) const;

        /** Get integer argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getIntArgByName(int& argumentValue, std::string_view argumentName) const;

        /** Get an unsiged integer argument by name
        * @param [out] argumentValue placeholder for response
        * @param [in] argumentName name of argument
        * @return true if argument has been obtained
        * @remark If argument is presented argumentValue obtained this value. If argument is not presented argumentValue is not changed.
        */
        bool getUnsignedArgByName(uint32_t& argumentValue, std::string_view argumentName) const;

        /** Check that flag is provided as "f" or "-f" or "--f" in the command line
        * @param flagName name of flag in which you're interested in
        * @return index within a command line
        * @remark Case sensitive test. Flag "-f" differs from flag "-F".
        * @remark Flag "[-|--]f 0" or "[-|--]f no" or "[-|--]f off" is the same as the flag is not presented.
        */
        bool isFlagSetuped(std::string_view flagName) const;

    private:

#ifdef USE_TRIE_IN_CMDLINE_PARSER
        TernaryTrie<char, int> m_arguments; ///< Trie which contains all arguments passed to command line
#endif
        int argc;                           ///< Number of arguments in command line
        char** argv;                        ///< Number of arguments passed to command line
    };
}
