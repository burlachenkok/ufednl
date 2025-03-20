#include "dopt/fs/include/FileSystemHelpers.h"
#include "dopt/fs/include/FileNameHelpers.h"
#include "dopt/system/include/Version.h"

#include "gtest/gtest.h"

TEST(dopt, FileSystemHelpersGTest)
{
    const std::string cwdOrig = dopt::FileSystemHelpers::getCwd();

    EXPECT_TRUE(dopt::FileSystemHelpers::chDir(cwdOrig));
    const std::string upFolder = dopt::FileNameHelpers::buildFileName(cwdOrig, "/..");
    EXPECT_TRUE(dopt::FileSystemHelpers::chDir(upFolder));
    EXPECT_NE(dopt::FileSystemHelpers::getCwd(), cwdOrig);
	
    const std::string subFolderName = dopt::FileNameHelpers::extractBaseName(cwdOrig);

    EXPECT_TRUE(dopt::FileSystemHelpers::isDirExist(subFolderName));
    EXPECT_TRUE(dopt::FileSystemHelpers::isDirExist(cwdOrig));

    EXPECT_FALSE(dopt::FileSystemHelpers::isFileExist(subFolderName));
    EXPECT_TRUE(dopt::FileSystemHelpers::isFileOrFolderExist(subFolderName));
    EXPECT_FALSE(dopt::FileSystemHelpers::createDir(subFolderName));

    EXPECT_TRUE(dopt::FileSystemHelpers::createDir(subFolderName + "_salt3"));

    EXPECT_TRUE(dopt::FileSystemHelpers::removeDir(subFolderName + "_salt3"));
    EXPECT_TRUE(dopt::FileSystemHelpers::chDir(cwdOrig));
    EXPECT_EQ(dopt::FileSystemHelpers::getCwd(), cwdOrig);
}

TEST(dopt, TestFileMapping)
{    
    std::string existFileStr = dopt::FileNameHelpers::buildFileName(dopt::projectRootDirectory4Build, "../README_1_MIN_TOOLS.md");
    const char* existFile = existFileStr.c_str();   

    {
        dopt::FileSystemHelpers::FileMappingResult mapRes = 
            dopt::FileSystemHelpers::mapFileToMemory(existFile, true);
        
        EXPECT_TRUE(mapRes.fileSizeInBytes == mapRes.memorySizeInBytes);
        EXPECT_TRUE(dopt::FileSystemHelpers::getFileSize(existFile) == mapRes.fileSizeInBytes);

        EXPECT_TRUE(mapRes.isOk);
        EXPECT_TRUE(dopt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }

    {
        dopt::FileSystemHelpers::FileMappingResult mapRes =
            dopt::FileSystemHelpers::mapFileToMemory(existFile, false);

        EXPECT_TRUE(mapRes.fileSizeInBytes == mapRes.memorySizeInBytes);
        EXPECT_TRUE(dopt::FileSystemHelpers::getFileSize(existFile) == mapRes.fileSizeInBytes);

        EXPECT_TRUE(mapRes.isOk);
        EXPECT_TRUE(dopt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }

    {
        dopt::FileSystemHelpers::FileMappingResult mapRes = dopt::FileSystemHelpers::mapFileToMemory("not_existing_file.my", true);
        EXPECT_FALSE(mapRes.isOk);
        EXPECT_TRUE(dopt::FileSystemHelpers::unmapFileFromMemory(mapRes));
    }
}
