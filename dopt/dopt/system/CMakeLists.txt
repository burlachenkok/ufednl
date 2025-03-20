cmake_minimum_required(VERSION 3.12)

include(DoptVersionGit)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
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

#============= BUILD TARGETS =================================================================
if (${DOPT_CUDA_SUPPORT})
  cuda_add_library(${PROJECT_NAME} STATIC ${original_src} ${original_headers})
  target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
else()
  add_library(${PROJECT_NAME} STATIC ${original_src} ${original_headers})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#============= BUILD TARGETS =================================================================

target_link_libraries(${PROJECT_NAME} linalg_vectors)

configureCompileFlags()
