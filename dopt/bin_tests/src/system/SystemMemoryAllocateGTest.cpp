#include "dopt/system/include/SystemMemoryAllocate.h"
#include "dopt/system/include/ProcessInfo.h"
#include "dopt/system/include/MemInfo.h"

#include "gtest/gtest.h"

TEST(dopt, SysMemAllocateGTest)
{
    size_t pageSize = dopt::virtualPageSize();

    void* mem = dopt::allocateVirtualMemory(pageSize, 1);
    EXPECT_TRUE(mem != nullptr);
    EXPECT_TRUE(dopt::lockVirtualMemory(mem, pageSize, 1));
    EXPECT_TRUE(dopt::unlockVirtualMemory(mem, pageSize, 1));
    EXPECT_TRUE(dopt::deallocateVirtualMemory(mem, pageSize, 1));
    void* memPinned = dopt::allocateVirtualMemory(pageSize, 1024);
    EXPECT_TRUE(memPinned != nullptr);

    EXPECT_TRUE(dopt::deallocateVirtualMemory(memPinned, pageSize, 1024));

    std::cout << "Information about memory" << '\n';
    std::cout << "  Physical Memory for process: " << dopt::physicalMemoryForProcess() / 1024 << " KBytes\n";
    std::cout << "  Virtual and Physical Memory for process: " << dopt::totalVirtualAndPhysicalMemoryForProcess() / 1024 << " KBytes\n";
    std::cout << "  Available DRAM memory in the system: " << dopt::availablalePhysicalMemoryInBytes() / 1024 / 1024 / 1024 << " GBytes\n";
    std::cout << "  Installed DRAM memory in the system: " << dopt::installedPhysicalMemoryInBytes() / 1024 / 1024 / 1024 << " GBytes\n";
    std::cout << '\n';
    std::cout << "  Page Size: " << dopt::virtualPageSize()/1024 << " KBytes\n";
    std::cout << "  Process ID: " << dopt::currentProcessId() << '\n';
    std::cout << "  Thread ID: " << dopt::currentThreadId() << '\n';
    std::cout << '\n';
    dopt::ProcessStatistics proc_stats = dopt::getProcessStatistics();
    std::cout << "  Memory for dynamic libraries and executable image itself: " << proc_stats.memoryForImages / 1024 << " KBytes\n";
    std::cout << "  Memory for mapped files: " << proc_stats.memoryForMappedFiles / 1024 << " KBytes\n";
    std::cout << "  Private memory allocated for process: " << proc_stats.memoryPrivateForProcess / 1024 << " KBytes\n";
    std::cout << '\n';
}
