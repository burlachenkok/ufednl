#include "dopt/cmdline/include/CmdLineParser.h"
#include "gtest/gtest.h"

TEST(dopt, CmdLineParserGTest)
{
    const char* testArgs[9] = {};
    testArgs[0] = "app";
    testArgs[1] = "launch";
    testArgs[2] = "-f";
    testArgs[3] = "1";
    testArgs[4] = "-m";
    testArgs[5] = "5";
    testArgs[6] = "-mf";
    testArgs[7] = "1.2";
    testArgs[8] = "--g";

    const dopt::CmdLine cmdline(9, (char**)testArgs);
    EXPECT_TRUE(cmdline.getArgCount() == 9);
    EXPECT_TRUE(cmdline.isFlagSetuped("f"));
    EXPECT_TRUE(!cmdline.isFlagSetuped("z"));

    int vv = 101;
    EXPECT_TRUE(cmdline.getIntArgByName(vv, "mmmm") == false);
    EXPECT_TRUE(vv == 101);

    int temp = 0;
    EXPECT_TRUE(cmdline.getIntArgByName(temp, "m"));
    EXPECT_TRUE(temp == 5);
    
    temp = 0;
    EXPECT_TRUE(cmdline.getIntArgByName(temp, "m"));
    EXPECT_TRUE(temp == 5);
    EXPECT_FALSE(cmdline.getIntArgByName(temp, "z"));
    EXPECT_FALSE(cmdline.getIntArgByName(temp, "g"));
    EXPECT_TRUE(cmdline.isFlagSetuped("g"));

    double doubleTemp = 0;
    EXPECT_TRUE(cmdline.getDoubleArgByName(doubleTemp, "mf"));
    EXPECT_DOUBLE_EQ(doubleTemp, 1.2);

    double floatTemp = 0;
    EXPECT_TRUE(cmdline.getDoubleArgByName(floatTemp, "mf"));
    EXPECT_DOUBLE_EQ(floatTemp, 1.2);

    std::string_view value = "";
    char mf_flag[] = "mf1111";
    std::string_view argName = {mf_flag, 2};
    
    EXPECT_TRUE(argName.data() == mf_flag) << "Check pointers";
    EXPECT_TRUE(cmdline.getStringViewArgByName(value, argName));
    EXPECT_TRUE(value.data() == testArgs[7]) << "Check pointers";
}
