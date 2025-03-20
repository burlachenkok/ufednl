cmake_minimum_required(VERSION 3.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#=================================================================================================
# BINARY
#=================================================================================================
string(REPLACE " " "_" ProjectId ${ProjectId})

project(${ProjectId} LANGUAGES CXX C)
file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "src/*.cu")
file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp")

if(original_src)
    createSourceGrouping(${original_src})
endif()

if (original_headers)
    createHeadersGrouping(${original_headers})
endif()

if (${DOPT_CUDA_SUPPORT})
  cuda_add_executable(${PROJECT_NAME} ${original_src} ${original_headers})
else()
  add_executable(${PROJECT_NAME} ${original_src} ${original_headers})
endif()

target_link_libraries(${PROJECT_NAME} system)
target_link_libraries(${PROJECT_NAME} cmdline)
target_link_libraries(${PROJECT_NAME} copylocal)
target_link_libraries(${PROJECT_NAME} fs)
target_link_libraries(${PROJECT_NAME} linalg_vectors)
target_link_libraries(${PROJECT_NAME} linalg_matrices)
target_link_libraries(${PROJECT_NAME} random)
target_link_libraries(${PROJECT_NAME} timers)
target_link_libraries(${PROJECT_NAME} gtest)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
configureCompileFlags()

#=================================================================================================
# SHARED LIBRARY
#=================================================================================================
string(REPLACE "bin" "shared" ProjectId ${ProjectId})
if (DOPT_BUILD_SHARED_LIBRARIES)
    project(${ProjectId} LANGUAGES CXX C)

    file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "src/*.cu")
    file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp" "exported/*.h")

    if(original_src)
        createSourceGrouping(${original_src})
    endif()

    if (original_headers)
        createHeadersGrouping(${original_headers})
    endif()

    if (${DOPT_CUDA_SUPPORT})
      cuda_add_library(${PROJECT_NAME} SHARED ${original_src} ${original_headers})
    else()
      add_library(${PROJECT_NAME} SHARED ${original_src} ${original_headers})
    endif()

    target_link_libraries(${PROJECT_NAME} system)
    target_link_libraries(${PROJECT_NAME} cmdline)
    target_link_libraries(${PROJECT_NAME} copylocal)
    target_link_libraries(${PROJECT_NAME} fs)
    target_link_libraries(${PROJECT_NAME} linalg_vectors)
    target_link_libraries(${PROJECT_NAME} linalg_matrices)
    target_link_libraries(${PROJECT_NAME} random)
    target_link_libraries(${PROJECT_NAME} timers)
    target_link_libraries(${PROJECT_NAME} gtest)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    configureCompileFlagsForSharedLibraryBuild()
    configureCompileFlags()
endif()

#=================================================================================================
# SWIG WRAPPER
#=================================================================================================
string(REPLACE "shared" "python" ProjectId ${ProjectId})
if(DOPT_SWIG_INTERFACE_GENERATOR)
    project(${ProjectId} LANGUAGES CXX C)

    file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "src/*.cu")
    file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp" "exported/*.h" "exported/*.i")

    if(original_src)
        createSourceGrouping(${original_src})
    endif()

    if (original_headers)
        createHeadersGrouping(${original_headers})
    endif()

    swig_add_library(${PROJECT_NAME} LANGUAGE python SOURCES ${original_src} ${original_headers})
    swig_link_libraries(${PROJECT_NAME} system)
    swig_link_libraries(${PROJECT_NAME} cmdline)
    swig_link_libraries(${PROJECT_NAME} copylocal)
    swig_link_libraries(${PROJECT_NAME} fs)
    swig_link_libraries(${PROJECT_NAME} linalg_vectors)
    swig_link_libraries(${PROJECT_NAME} linalg_matrices)
    swig_link_libraries(${PROJECT_NAME} random)
    swig_link_libraries(${PROJECT_NAME} timers)
    swig_link_libraries(${PROJECT_NAME} gtest)
    swig_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})

    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/exported)
    configureCompileFlagsForSharedLibraryBuild()
    configureCompileFlags()

endif()
