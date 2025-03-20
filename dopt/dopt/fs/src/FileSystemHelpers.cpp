#include "FileSystemHelpers.h"
#include "FileNameHelpers.h"

#include "dopt/system/include/PlatformSpecificMacroses.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

namespace dopt
{
    bool FileSystemHelpers::chDir(const std::string& path)
    {
#ifdef DOPT_WINDOWS
        return _chdir(path.c_str()) == 0;
#else
        return chdir(path.c_str()) == 0;
#endif
    }

    std::string	FileSystemHelpers::getCwd()
    {
        char buff[1024 * 12] = {};

#ifdef DOPT_WINDOWS
        if (buff != _getcwd(buff, sizeof(buff)))
        {
            assert(!"PROBLEM WITH CALLING getcwd()");
            return std::string();
        }

#else
        if (buff != getcwd(buff, sizeof(buff)))
        {
            assert(!"PROBLEM WITH CALLING getcwd()");
            return std::string();
        }
#endif
        return std::string(buff);
    }

    bool FileSystemHelpers::isFileExist(const std::string& path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) == -1)
            return false;

        // file is regular file
        bool existance = (info.st_mode & (S_IFREG)) != 0;

        return existance;
    }

    bool FileSystemHelpers::isFileOrFolderExist(const std::string& path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) == -1)
            return false;

        // file is regular file or symbolic link or directory
        bool existance = (info.st_mode & S_IFREG) != 0 || 
                         (info.st_mode & S_IFDIR) != 0;

        return existance;
    }

    bool FileSystemHelpers::isDirExist(const std::string& path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) == -1)
            return false;
        return (info.st_mode & S_IFDIR) != 0; // file is a directory
    }

    uint64_t FileSystemHelpers::getFileSize(const std::string& path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) == -1)
            return 0;
        uint32_t size = static_cast<uint64_t>(info.st_size);
        return size;
    }

    uint32_t FileSystemHelpers::nonEmptyLinesInFile(const std::string& path)
    {
        FILE* f = fopen(path.c_str(), "r");
        uint32_t lines = 0;
        int symbol = 0;
        int prevSymbol = 0;
        for (;;)
        {
            symbol = getc(f);
            if (symbol == '\r')
                continue;

            if (symbol == EOF)
            {
                break;
            }
            else if (symbol == '\n' && prevSymbol != '\n')
            {
                lines++;
            }
            prevSymbol = symbol;
        }
        fclose(f);
        return lines;
    }

    bool FileSystemHelpers::createDir(const std::string& path)
    {
        if (isFileOrFolderExist(path))
            return false;

#if DOPT_WINDOWS
        return _mkdir(path.c_str()) != -1;

#elif DOPT_LINUX || DOPT_MACOS
        // Create directory with the following permissions:
        // http://pubs.opengroup.org/onlinepubs/7908799/xsh/sysstat.h.html:
        //   S_IRWXU -- read, write, execute/search by owner
        //   S_IRWXG -- read, write, execute/search by group
        //   S_IRWXO -- read, write, execute/search by others
        return mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) != -1;
#endif
    }

    bool FileSystemHelpers::removeDir(const std::string& path)
    {
        if (isDirExist(path) == false)
            return false;

#if DOPT_WINDOWS
        return _rmdir(path.c_str()) != -1;

#elif DOPT_LINUX || DOPT_MACOS
        return rmdir ( path.c_str() ) != -1;
#endif
    }

    bool FileSystemHelpers::removeFile(const std::string& fileName)
    {
        std::string	fullFileName = FileNameHelpers::normalizePath(fileName);
        return remove(fullFileName.c_str()) == 0;
    }

    bool FileSystemHelpers::saveFile(const std::string& fileName, 
                                     void* rawBuffer, 
                                     size_t rawBufferSize)
    {
        int file = 0;
        std::string	fullFileName = FileNameHelpers::normalizePath(fileName);

        if (fullFileName == "stdout") 
        {
            file = 1; // stdout
        }
        else
        {
            file = open(fullFileName.c_str(), O_WRONLY | O_BINARY | O_CREAT | O_TRUNC,
                                              S_IWOTH | S_IROTH | S_IWGRP | S_IRGRP | S_IWUSR | S_IRUSR /*S_IRWXU|S_IRWXG|S_IRWXO*/);
        }

        if (file == -1)
            return false;

        size_t totalSizeToDump = rawBufferSize;
        size_t curPosToWrite = 0;
        size_t totalCharsWritten = 0;

        for (;;)
        {
            int charsWritten = write(file, 
                                     (unsigned char*)rawBuffer + curPosToWrite, 
                                     totalSizeToDump);

            if (charsWritten == -1)
                break;

            totalSizeToDump -= charsWritten;
            curPosToWrite += charsWritten;
            totalCharsWritten += charsWritten;

            if (totalSizeToDump == 0)
                break;
        }

        if (file != 1)
            close(file);

        return totalCharsWritten == rawBufferSize;
    }
    
#ifdef DOPT_WINDOWS
    FileSystemHelpers::FileMappingResult FileSystemHelpers::mapFileToMemory(const char* fname, bool isReadOnly, bool isCreareIfNotExist)
    {
        FileMappingResult res;

        res.memory = nullptr;
        res.memorySizeInBytes = 0;
        res.isReadOnly = isReadOnly;
        res.isOk = false;
        res.errorMsg = "";
        res.fileSizeInBytes = 0;

        DWORD dwFlagsAttr = FILE_ATTRIBUTE_NORMAL;

/*
        if (!optUseCaching)
        {
            dwFlagsAttr |= FILE_FLAG_NO_BUFFERING;
            dwFlagsAttr |= FILE_FLAG_WRITE_THROUGH;
        }

        if (optSequntialAccess)
        {
            dwFlagsAttr |= FILE_FLAG_SEQUENTIAL_SCAN;
        }
        else
        {
            dwFlagsAttr |= FILE_FLAG_RANDOM_ACCESS;
        }
*/
        
        HANDLE fhandle = CreateFile(fname, 
                                    isReadOnly ? GENERIC_READ : GENERIC_READ| GENERIC_WRITE, 
                                    FILE_SHARE_READ|FILE_SHARE_WRITE,
                                    nullptr,
                                    isCreareIfNotExist ? OPEN_ALWAYS : OPEN_EXISTING,
                                    dwFlagsAttr, NULL);

        if (fhandle == INVALID_HANDLE_VALUE)
        {
            res.errorMsg = "Can not create/open file";
            return res;
        }

        LARGE_INTEGER fileSize = {};

        if (GetFileSizeEx(fhandle, &fileSize) == 0)
        {
            CloseHandle(fhandle);
            res.errorMsg = "Can not get file size";
            return res;
        }
        res.fileSizeInBytes = fileSize.QuadPart;
        
        HANDLE hMapView = CreateFileMapping(fhandle, NULL,
                                            (isReadOnly ? PAGE_READONLY : PAGE_READWRITE),
                                            0, 0,  // Create file view for whole file
                                            NULL   // Name for inter process communication. We don't need it
                                            );

        CloseHandle(fhandle);
        
        res.memorySizeInBytes = res.fileSizeInBytes;
        
        if (hMapView == NULL)
        {
            res.errorMsg = "Can not create file mapping object";           
            return res;
        }

        res.memory = MapViewOfFile(hMapView,
                                   isReadOnly ? (FILE_MAP_READ) : (FILE_MAP_READ | FILE_MAP_WRITE),
                                   0, 0, // Offsets
                                   0);             // Create file view from the beginning up to the end of the file
        
        CloseHandle(hMapView);
        
        if (res.memory == nullptr)
        {
            res.errorMsg = "Can not create view file";
            return res;
        }
        
        res.isOk = true;
        return res;
    }

    bool FileSystemHelpers::unmapFileFromMemory(FileSystemHelpers::FileMappingResult& viewOfFile)
    {
        if (viewOfFile.memory == nullptr)
            return true;
        
        BOOL result = UnmapViewOfFile(viewOfFile.memory);
        
        if (result != 0)
        {
            viewOfFile.memory = nullptr;
            return true;
        }
        else
        {
            return false;
        }
    }
    
    bool FileSystemHelpers::flushAllChangesInMemoryMapping(FileSystemHelpers::FileMappingResult& viewOfFile)
    {
        if (viewOfFile.isReadOnly)
        {
            assert(!"Still not good. The files has been opened for read-only, and you're asking to flush changes.");
            return true;
        }
        BOOL res = FlushViewOfFile(viewOfFile.memory, viewOfFile.memorySizeInBytes);
        return res != 0;
    }

#else
    FileSystemHelpers::FileMappingResult FileSystemHelpers::mapFileToMemory(const char* fname, bool isReadOnly, bool isCreareIfNotExist)
    {
        FileMappingResult res;

        res.memory = nullptr;
        res.memorySizeInBytes = 0;
        res.isReadOnly = isReadOnly;
        res.isOk = false;
        res.errorMsg = "";
        res.fileSizeInBytes = 0;
        
        int file = open(fname, O_BINARY | (isCreareIfNotExist ? O_CREAT : 0) |
                               (isReadOnly ? O_RDONLY : O_RDWR)
                               // | (!optUseCaching ? O_DIRECT : 0)
                               , S_IWOTH | S_IROTH | S_IWGRP | S_IRGRP | S_IWUSR | S_IRUSR);

        if (file == -1)
        {
            res.errorMsg = "Can not create/open file";
            return res;
        }

        struct stat sb;
        if (fstat(file, &sb) == -1)
        {
            close(file);
            res.errorMsg = "Can not get file size";
            return res;
        }

        res.fileSizeInBytes = static_cast<uint64_t>(sb.st_size);
        res.memorySizeInBytes = res.fileSizeInBytes;

        void* ptr = mmap(nullptr,
                        res.memorySizeInBytes,
                        // The content can be read and write
                        isReadOnly ? PROT_READ : PROT_READ | PROT_WRITE,
                        MAP_SHARED,              // Result of file modifications are shared to other processes (e.g. try MEM_PRIVATE if it is not needed)
                        file,                    // File handle
                        0);                      // Offset

        close(file);

        if (ptr == MAP_FAILED)
        {
            res.errorMsg = "Can not view file";
            return res;
        }
        else
        {
            res.memory = ptr;
        }

        res.isOk = true;

        // File Mapping is finished.
        // The pages of the mapping are(automatically) loaded from the file as required

        // In fact Memory Mapped files exhibits better performace. For discussion see p.1026 in "The Linu x Programming Interface" book.
        return res;
    }

    bool FileSystemHelpers::unmapFileFromMemory(FileSystemHelpers::FileMappingResult& viewOfFile)
    {
        if (viewOfFile.memory == nullptr)
            return true;

        int result = munmap(viewOfFile.memory, viewOfFile.memorySizeInBytes);

        if (result != 0)
        {
            return false;
        }
        else
        {
            viewOfFile.memory = nullptr;
            return true;
        }
    }

    bool FileSystemHelpers::flushAllChangesInMemoryMapping(FileSystemHelpers::FileMappingResult& viewOfFile)
    {
        if (viewOfFile.isReadOnly)
        {
            assert(!"Still not good. The files has been opened for read-only, and you're asking to flush changes.");
            return true;
        }
        int  res = msync(viewOfFile.memory, viewOfFile.memorySizeInBytes, MS_SYNC);
        return res == 0;
    }

#endif
}
