#include "dopt/math_routines/include/TernaryTrie.h"
#include "dopt/random/include/RandomGenIntegerLinear.h"
#include "dopt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <string_view>
#include <algorithm>
#include <vector>
#include <stdint.h>
#include <string>
#include <sstream>

TEST(dopt, TernaryTrieGTest)
{
    {
        dopt::TernaryTrie<char, int> ctr;
        EXPECT_TRUE(ctr.isEmpty());        
        EXPECT_FALSE(ctr.has("Hi"));
        ctr["Hi"] = 12;
        EXPECT_FALSE(ctr.has("H"));
        EXPECT_TRUE(ctr.has("Hi"));
        ctr["Hi2"] = 123;
        ctr.remove("Hi");
        EXPECT_FALSE(ctr.isEmpty());
        EXPECT_FALSE(ctr.has("Hi"));
        EXPECT_TRUE(ctr.has("Hi2"));
        ctr.remove("Hi");
        EXPECT_TRUE(ctr.has("Hi2"));
        ctr.remove("Hi2");
        EXPECT_TRUE(ctr.isEmpty());
    }

    {
        dopt::TernaryTrie<char, int> ctr;
        ctr.put("wat", 1);
        ctr.put("we", 2);
        ctr.put("we1", 3);
        ctr.put("we12", 4);

        dopt::TernaryTrie<char, int> ctrCopy(ctr);
        EXPECT_TRUE(ctrCopy.has("wat"));
        EXPECT_TRUE(ctrCopy.has("we"));
        EXPECT_TRUE(ctrCopy.has("we1"));
        EXPECT_TRUE(ctrCopy.has("we12"));
        EXPECT_TRUE(ctrCopy.keys().size() == 4);

        ctrCopy.clear();
        EXPECT_TRUE(ctrCopy.keys().size() == 0);
        ctrCopy = ctr;
        EXPECT_TRUE(ctrCopy.keys().size() == 4);

        //ctr.put("", -2);
        //EXPECT_TRUE(ctr.has(""));
        //ctr.remove("");

        ctr.remove("we1");
        EXPECT_EQ(2, *ctr.get("we"));
        EXPECT_EQ(4, *ctr.get("we12"));
        EXPECT_EQ(1, *ctr.get("wat"));


        dopt::TernaryTrie<char, int> ctr2(ctr);
        EXPECT_EQ(2, *ctr2.get("we"));
        EXPECT_EQ(4, *ctr2.get("we12"));
        EXPECT_EQ(1, *ctr2.get("wat"));

        EXPECT_TRUE(ctr2.keys().size() == 3);
        EXPECT_TRUE(ctr.keys().size() == 3);

        ctr2.clear();
        EXPECT_TRUE(ctr2.isEmpty());
        ctr2 = ctr;

        EXPECT_EQ(2, *ctr2.get("we"));
        EXPECT_EQ(4, *ctr2.get("we12"));
        EXPECT_EQ(1, *ctr2.get("wat"));

        EXPECT_EQ(2, *ctr2.get(std::string_view("we")));
        EXPECT_EQ(4, *ctr2.get(std::string_view("we12")));
        EXPECT_EQ(1, *ctr2.get(std::string_view("wat")));
    }

    {
        const char* testStrings[] = { "b1", "c", "pop", "a", "po2", "zlast", "blabla" , "b" };
        dopt::TernaryTrie<char, const char*> ctr;

        for (size_t i = 0; i < sizeof(testStrings)/sizeof(testStrings[0]); ++i)
        {
            EXPECT_FALSE(ctr.has(testStrings[i]));
            ctr.put(std::string(testStrings[i]).c_str(), testStrings[i]);
            EXPECT_TRUE(ctr.has(testStrings[i]));
            EXPECT_STREQ(*ctr.get(testStrings[i]), testStrings[i]);
        }
        EXPECT_FALSE(ctr.isEmpty());

        auto keys = ctr.keys();
        EXPECT_EQ(sizeof(testStrings)/sizeof(testStrings[0]), keys.size());
        EXPECT_STREQ("a", keys.front().c_str());
        EXPECT_STREQ("b", keys[1].c_str());
        EXPECT_STREQ("zlast", keys.back().c_str());

        EXPECT_EQ(strlen("blabla"), ctr.longestPrefixForKey("blablapop"));
        EXPECT_EQ(1, ctr.longestPrefixForKey("blablQQ"));
        EXPECT_TRUE(ctr.keysWithPrefix("po1").empty());
        EXPECT_EQ(2, ctr.keysWithPrefix("po").size());

        auto prefixes = ctr.keysWithPrefix("po");
        std::string str1 = prefixes[0];
        std::string str2 = prefixes[1];
        
        EXPECT_TRUE(strcmp("po2", &str1[0]) == 0 && strcmp("pop", &str2[0]) == 0 || strcmp("po2", &str2[0]) == 0 && strcmp("pop", &str1[0]) == 0);

        for (size_t i = 0; i < sizeof(testStrings)/sizeof(testStrings[0]) - 1; ++i)
        {
            EXPECT_TRUE(ctr.has(testStrings[i]));
            ctr.remove(std::string(testStrings[i]).c_str());
            EXPECT_FALSE(ctr.has(testStrings[i]));
            EXPECT_TRUE(ctr.has(testStrings[i + 1]));
        }
        EXPECT_FALSE(ctr.isEmpty());
        ctr.remove(testStrings[sizeof(testStrings) / sizeof(testStrings[0]) - 1]);
        EXPECT_TRUE(ctr.isEmpty());
        ctr.put("es", "cleanup during dtor");
    }

    {
        dopt::TernaryTrie<char, int> ctr;
    }

    {
        dopt::TernaryTrie<char, int> ctr;
        ctr.put("abc", 1);
        ctr.put("azv", 2);
        ctr.put("q23", 3);
        ctr.put("", 4);
        ctr.put("z1", 5);    

        ctr.put("", 4);
        ctr.put("z1", 5);

        dopt::TernaryTrie<char, int> ctr2(ctr);

        ASSERT_TRUE(ctr2.keys().size() == 5);

        auto keys = ctr2.keys();
        EXPECT_TRUE(keys[0] == "");
        EXPECT_TRUE(keys[1] == "abc");
        EXPECT_TRUE(keys[2] == "azv");
        EXPECT_TRUE(keys[3] == "q23");
        EXPECT_TRUE(keys[4] == "z1");

        EXPECT_TRUE(ctr2.has(""));
        EXPECT_TRUE(ctr2.has("abc"));
        EXPECT_FALSE(ctr2.has("ab"));

        EXPECT_TRUE(ctr2.keysWithPrefix("").size() == 5);
        EXPECT_TRUE(ctr2.keysWithPrefix("a").size() == 2);
        EXPECT_TRUE(ctr2.keysWithPrefix("ab").size() == 1);
        EXPECT_TRUE(ctr2.keysWithPrefix("ab")[0] == "abc");
        EXPECT_TRUE(ctr2.keysWithPrefix("z")[0] == "z1");

    }
}


TEST(dopt, TernaryTrieGPerf)
{
    std::stringstream s;
    dopt::TernaryTrie<char, int> ctr;
    dopt::RandomGenIntegerLinear gen(1, 255);    
    
    gen.setSeed(123);

    dopt::HighPrecisionTimer tm1;
    dopt::HighPrecisionTimer tm1_extra;

    tm1_extra.pause();
    
    constexpr size_t iIterations = size_t(20) * size_t(1000);
    constexpr size_t kLen = 50;
    
    for (size_t i = 0; i < iIterations; ++i)
    {
        tm1_extra.resume();
        s.str(std::string());

        for (size_t k = 0; k < kLen; ++k)
        {
            s << char(gen.generateInteger());
        }
        
        std::string strInternal = s.str();
        const char* str = strInternal.c_str();
        tm1_extra.pause();

        ctr.put(str, i);
    }
    tm1.pause();

    gen.setSeed(123);
    dopt::HighPrecisionTimer tm2;
    dopt::HighPrecisionTimer tm2_extra;


    for (size_t i = 0; i < iIterations; ++i)
    {
        tm2_extra.resume();
        s.str(std::string());

        for (size_t k = 0; k < kLen; ++k)
        {
            s << char(gen.generateInteger());
        }
        
        std::string strInternal = s.str();
        const char* str = strInternal.c_str();
        tm2_extra.pause();

        EXPECT_TRUE(ctr.get(str) != nullptr);
    }
    tm2.pause();

    std::cout << "Read time from trie: " << tm2.getTimeMs() - tm2_extra.getTimeMs() << " msec\n";
    std::cout << "Write time to trie: " << tm1.getTimeMs() - tm1_extra.getTimeMs() << " msec\n";
}