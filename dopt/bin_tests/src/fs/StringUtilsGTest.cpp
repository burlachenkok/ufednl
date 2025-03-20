#include "dopt/fs/include/StringUtils.h"
#include "gtest/gtest.h"

TEST(dopt, ConvertersGTest)
{
    EXPECT_STREQ("-123", dopt::string_utils::toString(-123).c_str());
    EXPECT_STREQ("mmm", dopt::string_utils::toString("mmm").c_str());
    int arr[] = { 1,222,3 };
    EXPECT_STREQ("[1,222,3]", dopt::string_utils::toString(arr, sizeof(arr) / sizeof(arr[0])).c_str());

    int b = 1;
    EXPECT_FALSE(dopt::string_utils::fromString(b, "qw"));
    EXPECT_TRUE(dopt::string_utils::fromString(b, "111"));
    EXPECT_EQ(b, 111);

    EXPECT_TRUE(dopt::string_utils::fromString(b, "+111"));
    EXPECT_EQ(b, 111);

    EXPECT_TRUE(dopt::string_utils::fromString(b, "-1234"));
    EXPECT_EQ(b, -1234);

    bool bvalue1 = false;
    EXPECT_TRUE(dopt::string_utils::fromString(bvalue1, "1"));
    EXPECT_EQ(bvalue1, true);

    EXPECT_FALSE(dopt::string_utils::fromString(bvalue1, "abc"));

    bool bvalue2 = false;
    EXPECT_TRUE(dopt::string_utils::fromString(bvalue2, "0"));
    EXPECT_EQ(bvalue2, false);

    EXPECT_FALSE(dopt::string_utils::fromString(bvalue2, "--0"));
    EXPECT_FALSE(dopt::string_utils::fromString(bvalue2, "-+0"));
    EXPECT_FALSE(dopt::string_utils::fromString(bvalue2, "-1+"));

    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, "123"));
        EXPECT_DOUBLE_EQ(dvalue, 123);
    }
    
    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, "+123.456"));
        EXPECT_DOUBLE_EQ(dvalue, 123.456);
    }

    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, "-123.456e+0"));
        EXPECT_DOUBLE_EQ(dvalue, -123.456);
    }

    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, "-123.456e+1"));
        EXPECT_DOUBLE_EQ(dvalue, -1234.56);
    }

    {
        double dvalue = 0.0;
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "--123.456e-1"));    
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "-123.456e--1"));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "-123.456-1"));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "-123.456ee-1"));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "+123.456eE-1"));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "++123"));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "+123.4.5"));
    }
    
    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, "-.45"));
        EXPECT_DOUBLE_EQ(dvalue, -0.45);

        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, "-.45 "));
        EXPECT_FALSE(dopt::string_utils::fromString(dvalue, " -.45"));
    }

    {
        double dvalue = 0.0;
        EXPECT_TRUE(dopt::string_utils::fromString(dvalue, ".123e1"));
        EXPECT_DOUBLE_EQ(dvalue, 1.23);
    }
    
    std::string arg_1 = "a,b, c";
    auto vec = dopt::string_utils::splitToSubstrings(arg_1, ',');
    EXPECT_TRUE(vec.size() == 3);
    EXPECT_TRUE(vec[0] == std::string("a"));
    EXPECT_TRUE(vec[1] == std::string("b"));
    EXPECT_TRUE(vec[2] == std::string(" c"));

    std::string arg_2 = "a; ;;;b;c;d;";
    auto vec2 = dopt::string_utils::splitToSubstrings(arg_2, ';');
    EXPECT_TRUE(vec2.size() == 5);
    EXPECT_TRUE(vec2[0] == std::string("a"));
    EXPECT_TRUE(vec2[1] == std::string(" "));
    EXPECT_TRUE(vec2[2] == std::string("b"));
    EXPECT_TRUE(vec2[3] == std::string("c"));
    EXPECT_TRUE(vec2[4] == std::string("d"));

    std::string arg_3 = "1.2, 3.4, 5.6, 1.0";
    auto vec3 = dopt::string_utils::splitToSubstrings(arg_3, ',');
    EXPECT_TRUE(vec3.size() == 4);

    std::string arg_4 = "a,b,,,c1";
    auto vec4 = dopt::string_utils::splitToSubstrings<true>(arg_4, ',');
    EXPECT_TRUE(vec4.size() == 5);

    EXPECT_TRUE(vec4[0] == "a");
    EXPECT_TRUE(vec4[1] == "b");
    EXPECT_TRUE(vec4[2] == "");
    EXPECT_TRUE(vec4[3] == "");
    EXPECT_TRUE(vec4[4] == "c1");

    std::string arg_5 = "a1,b22,,,c1";
    auto vec5 = dopt::string_utils::splitToSubstrings<false>(arg_5, ',');
    EXPECT_TRUE(vec5.size() == 3);

    EXPECT_TRUE(vec5[0] == "a1");
    EXPECT_TRUE(vec5[1] == "b22");
    EXPECT_TRUE(vec5[2] == "c1");

    {
        unsigned int a = 0;
        EXPECT_TRUE(dopt::string_utils::fromString(a, "123"));
        EXPECT_EQ(a, 123);

        EXPECT_TRUE(dopt::string_utils::fromString(a, "+456"));
        EXPECT_EQ(a, 456);

        EXPECT_FALSE(dopt::string_utils::fromString(a, "++456"));
        EXPECT_FALSE(dopt::string_utils::fromString(a, "-456"));
    }


    {
        long long a = 0;
        EXPECT_TRUE(dopt::string_utils::fromString(a, "123"));
        EXPECT_EQ(a, 123);

        EXPECT_TRUE(dopt::string_utils::fromString(a, "-456"));
        EXPECT_EQ(a, -456);
    }
}
