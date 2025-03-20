#pragma once

#include "dopt/system/include/CompilerInfo.h"
#include "dopt/system/include/PlatformSpecificMacroses.h"
#include "dopt/system/include/Version.h"
#include "dopt/system/include/MemInfo.h"
#include "dopt/system/include/ProcessInfo.h"
#include "dopt/system/include/digest/Crc.h"

#include <iostream>
#include <string_view>

inline void printInformationAboutDataset(std::string_view datasetName, const DatasetType& dataset, bool dumpDigest, bool dumpInText)
{
    std::cout << "About data set: \"" << datasetName << "\"\n";
    std::cout << "=========================================================================\n";
    std::cout << "  Total number of samples in data set: " << dataset.totalSamples() << '\n';
    std::cout << "  Total number of attributes in data set: " << dataset.numberOfAttributesForSamples() << '\n';
    std::cout << "  Design matrix includes intercept term: " << (dataset.hasInterceptTerm() ? "[YES]" : "[NO]") << '\n';
    std::cout << '\n';
    
    if (dumpDigest)
    {
        std::cout << "  Dataset CRC-32 samples: " << dopt::crc32(dataset.train_samples_tr.matrixByCols.rawData(),
                                                                 dataset.train_samples_tr.matrixByCols.sizeInBytes(),
                                                                 dopt::crc32Seed()) << '\n';

        std::cout << "  Dataset CRC-32 labels: " << dopt::crc32(dataset.train_outputs.rawData(),
                                                                dataset.train_outputs.sizeInBytes(),
                                                                dopt::crc32Seed()) << '\n';
    }
    
    if (dumpInText)
    {
        dataset.printInTextFormat(std::cout);
    }
    
    std::cout << "=========================================================================\n";
}

inline void printMemoryInformation()
{
    const dopt::ProcessStatistics stats = dopt::getProcessStatistics();

    std::cout << "=========================================================================\n";
    std::cout << "Memory Information\n";
    std::cout << "=========================================================================\n";;
    std::cout << "  Installed DRAM memory: " << int(dopt::installedPhysicalMemoryInBytes() / 1024 / 1024 / 1024) << " GBytes\n";
    std::cout << "  Free DRAM memory: " << int(dopt::availablalePhysicalMemoryInBytes() / 1024 / 1024 / 1024)    << " GBytes\n";
    std::cout << "  Used memory for images of binary files: " << int(stats.memoryForImages / 1024/1024)          << " MBytes\n";
    std::cout << "             Used memory for maped files: " << int(stats.memoryForMappedFiles / 1024/1024)     << " MBytes\n";
    std::cout << "              Private memory for process: " << int(stats.memoryPrivateForProcess / 1024/1024)  << " MBytes\n";
    std::cout << "=========================================================================\n";
}
