#include "dopt/fs/include/FileNameHelpers.h"
#include "gtest/gtest.h"

TEST(dopt, FileNameHelpersGTest)
{
    EXPECT_STREQ(dopt::FileNameHelpers::changeFileExt("test.txt", ".exe").c_str(), "test.exe");
    EXPECT_STREQ(dopt::FileNameHelpers::changeFileExt("./test2", ".exe").c_str(), "./test2.exe");
    EXPECT_STREQ(dopt::FileNameHelpers::changeFileExt(".\\test3.tar.gz", ".gz2").c_str(), ".\\test3.tar.gz2");

    EXPECT_STREQ(dopt::FileNameHelpers::getFileExt("1.txt").c_str(), ".txt");
    EXPECT_STREQ(dopt::FileNameHelpers::buildFileName("./", "1.txt").c_str(), "./1.txt");
    EXPECT_STREQ(dopt::FileNameHelpers::buildFileName(".//", "").c_str(), ".//");
    EXPECT_STREQ(dopt::FileNameHelpers::buildFileName("", "2.zip").c_str(), "2.zip");

    EXPECT_STREQ(dopt::FileNameHelpers::extractBaseName("/user/test\\subf\\12.txt").c_str(), "12.txt");
    EXPECT_STREQ(dopt::FileNameHelpers::extractBaseName("/user/test\\subf\\/2").c_str(), "2");

    EXPECT_STREQ(dopt::FileNameHelpers::cutFileExt("/user/test\\subf\\12.txt").c_str(), "/user/test\\subf\\12");
    EXPECT_STREQ(dopt::FileNameHelpers::cutFileExt("23.a").c_str(), "23");

    EXPECT_STREQ(dopt::FileNameHelpers::extractFolderName("qwe/1").c_str(), "qwe/");
    EXPECT_STREQ(dopt::FileNameHelpers::extractFolderName("/1\\2\\/qwe\\123").c_str(), "/1\\2\\/qwe\\");

    EXPECT_STREQ(dopt::FileNameHelpers::normalizePath("qwe/1.txt").c_str(), "qwe/1.txt");
    EXPECT_STREQ(dopt::FileNameHelpers::normalizePath("qwe////2.txt").c_str(), "qwe/2.txt");
    EXPECT_STREQ(dopt::FileNameHelpers::normalizePath(".\\qwe//.//./3.txt").c_str(), "./qwe/3.txt");
}
