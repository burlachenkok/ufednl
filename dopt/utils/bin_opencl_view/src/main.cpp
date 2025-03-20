#include "dopt/cmdline/include/CmdLineParser.h"

#if __has_include(<CL/cl.hpp>)
    #include <CL/cl.hpp>
#endif
#if __has_include(<CL/cl.h>)
    #include <CL/cl.h>
#endif
#if __has_include(<OpenCL/cl.hpp>)
    #include <OpenCL/cl.hpp>
#endif
#if __has_include(<OpenCL/cl.h>)
    #include <OpenCL/cl.h>
#endif

#include <vector>
#include <iostream>
#include <string_view>
#include <vector>

#include <assert.h>

std::vector<std::string_view> splitString(char* inputString, size_t inputStringLen, char separator = ' ')
{
    std::vector<std::string_view> res;

    for (size_t i = 0, j = 0;;)
    {
        if (j == inputStringLen || inputString[j] == '\0')
        {
            if (j != i)
                res.emplace_back(inputString + i, j - i);
            break;
        }

        if (inputString[j] == separator)
        {
            res.emplace_back(inputString + i, j - i);

            i = j + 1;

            while (inputString[i] == separator)
            {
                ++i;
            }
            j = i;
            continue;
        }
        ++j;
    }

    return res;
}

int main(int argc, char** argv)
{
#if DOPT_WINDOWS
    std::cout << "Operating System: Windows\n";
#elif DOPT_LINUX
    std::cout << "Operating System: Linux\n";
#elif DOPT_MACOS
    std::cout << "Operating System: macOS\n";
#else
    std::cout << "Operating System: Unknown\n";
#endif
    
    dopt::CmdLine cmdline(argc, argv);
    
    cl_uint num_platforms = 0;
    cl_int err = CL_SUCCESS;
    
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    assert(err == CL_SUCCESS);
    std::cout << "Available OpenCL Platforms: " << num_platforms << '\n';
    std::cout << '\n';
    if (num_platforms == 0)
        return 0;
    
    std::vector<cl_platform_id> platforms;
    platforms.resize(num_platforms);
    
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    assert(err == CL_SUCCESS);

    constexpr size_t kAttributes = 5; 
    const char* attributeNames[kAttributes]            = { "Platform Name", "Platform Vendor", "Platform Version", "Platform Profile", "Platform Extensions" };
    const cl_platform_info attributeTypes[kAttributes] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };

    for (size_t i = 0; i < num_platforms; ++i)
    {
        std::cout << "-----------------------------------------\n";
        std::cout << "Platform #" << i << '\n';
        std::cout << "-----------------------------------------\n";
        cl_uint numDevices = 0;
        cl_int status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::cout << " Available OpenCL Devices in Platform #" << i << " is: " << numDevices << '\n';

        for (size_t j = 0; j < kAttributes; j++) 
        {
            if (attributeTypes[j] == CL_PLATFORM_EXTENSIONS && cmdline.isFlagSetuped("show-extensions") == false)
                continue;
            
            size_t infoSize = 0; 
            cl_int getSizeSt = clGetPlatformInfo(platforms[i], attributeTypes[j], 0, nullptr, &infoSize);
            std::vector<char> info;
            info.resize(infoSize + 1);
            cl_int getValueSt = clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info.data(), nullptr);

            if (attributeTypes[j] == CL_PLATFORM_EXTENSIONS)
            {
                std::cout << " " << attributeNames[j] << ":" << '\n';
                std::vector<std::string_view> exts = splitString(info.data(), infoSize);
                for (std::string_view e : exts)
                {
                    std::cout << "   > " << e << '\n';
                }
            }
            else
            {
                std::cout << " " << attributeNames[j] << ":" << info.data() << '\n';
            }
        }
        std::cout << " " << "Number of devices: " << numDevices << '\n';
        std::cout << "-----------------------------------------\n";

        if (numDevices == 0)
            continue;

        // Allocate enough space for each device
        std::vector<cl_device_id> devices;
        devices.resize(numDevices);

        // Fill in the devices
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
 
        for (size_t d = 0; d < numDevices; ++d)
        {
            std::cout << "  Attributes of device #" << d << " at platform #" << i << '\n';
            std::cout << '\n';

            cl_ulong localMemSize = cl_ulong();
            cl_int errLocalMemSize = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);

            cl_ulong globalMemSize = cl_ulong();
            cl_int errGlobalMemSize = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);

            cl_uint ctas_or_compute_units = cl_uint();
            cl_int errCtas = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ctas_or_compute_units), &ctas_or_compute_units, nullptr);

            size_t kernerlArgsInBytes = size_t();
            cl_int kernerlArgsInBytesErr = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(kernerlArgsInBytes), &kernerlArgsInBytes, nullptr);

            cl_ulong globalCacheSize = cl_ulong();
            cl_int globalCacheSizeErr = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(globalCacheSize), &globalCacheSize, nullptr);
               
            cl_bool isLittleEndian = cl_bool();
            cl_int isLittleEndianErr = clGetDeviceInfo(devices[d], CL_DEVICE_ENDIAN_LITTLE, sizeof(isLittleEndian), &isLittleEndian, nullptr);
            
            cl_device_type devType = cl_device_type();
            cl_int devTypeErr = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(devType), &devType, nullptr);
            
            cl_uint clockFreq = cl_uint();
            cl_int clockFreqErr = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, nullptr);
            
            cl_uint cacheLineGlobalMem = cl_uint();
            cl_int cacheLineGlobalMemErr = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineGlobalMem), &cacheLineGlobalMem, nullptr);


            size_t extSize = 0;
            cl_int getSizeSt = clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, 0, nullptr, &extSize);
            std::vector<char> extPerDevice;
            extPerDevice.resize(extSize + 1);
            cl_int getExtErr = clGetDeviceInfo(devices[d], CL_DEVICE_EXTENSIONS, extSize, extPerDevice.data(), nullptr);


            if (devType == CL_DEVICE_TYPE_CPU)
                std::cout << "   Device #" << d << ": " << "Device is a CPU\n";
            else if (devType == CL_DEVICE_TYPE_GPU)
                std::cout << "   Device #" << d << ": " << "Device is a GPU\n";
            else if (devType == CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "   Device #" << d << ": " << "Device is an Accelerator\n";
            else if (devType == CL_DEVICE_TYPE_DEFAULT)
                std::cout << "   Device #" << d << ": " << "Device is a Default\n";
            else
                std::cout << "   Device #" << d << ": " << "Device is a Custom\n";
            
            std::cout << "   Device #" << d << ": " << "Maximum Clock Frequency (approximately): " << double(clockFreq / 1024) << " GHz\n";

            std::cout << "   Device #" << d << ": " << "Size of global memory: " << globalMemSize/1024/1024/1024 << " GBytes\n";
            std::cout << "   Device #" << d << ": " << "Size of global memory cache size: " << globalCacheSize / 1024 << " KBytes\n";
            std::cout << "   Device #" << d << ": " << "Cache Line Size for global memory: " << cacheLineGlobalMem << " Bytes\n";
                
            std::cout << "   Device #" << d << ": " << "Size of local [shared] memory: " << localMemSize / 1024 << " KBytes\n";
            std::cout << "   Device #" << d << ": " << "Size of passed arguments to kernel: " << kernerlArgsInBytes << " Bytes\n";
            std::cout << "   Device #" << d << ": " << "Number of compute units [ThreadBlocks executors]: " << ctas_or_compute_units << "\n";
            std::cout << "   Device #" << d << ": " << (isLittleEndian ? "Device is a Little Endian" : "Device is a Big Endian") << "\n";
            
            constexpr size_t kDevAttributes = 5;
            const char* devAttributeNames[kDevAttributes] = { "Name", "Vendor", "Driver Version", "Profile", "Device Version" };
            const cl_platform_info devAttributeTypes[kDevAttributes] = { CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DRIVER_VERSION, CL_DEVICE_PROFILE, CL_DEVICE_VERSION };
            
            for (size_t k = 0; k < kDevAttributes; ++k)
            {
                size_t infoSize = 0;
                cl_int getSizeSt = clGetDeviceInfo(devices[d], devAttributeTypes[k], 0, nullptr, &infoSize);
                std::vector<char> info;
                info.resize(infoSize + 1);
                cl_int getValueSt = clGetDeviceInfo(devices[d], devAttributeTypes[k], infoSize, info.data(), nullptr);
                std::cout << "   Device #" << d << ": " << devAttributeNames[k] << ": " << info.data() << '\n';               
            }

            if (cmdline.isFlagSetuped("show-extensions"))
            {
                std::cout << "   Device #" << d << ": " << "Device Extensions:" << '\n';
                std::vector<std::string_view> exts = splitString(extPerDevice.data(), extSize);
                for (std::string_view e : exts)
                {
                    std::cout << "     > " << e << '\n';
                }
            }
        }
        
        std::cout << '\n';
    }
}
