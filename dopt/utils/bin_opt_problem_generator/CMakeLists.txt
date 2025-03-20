cmake_minimum_required(VERSION 3.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId} LANGUAGES CXX)

file(GLOB_RECURSE original_src "src/*.cpp" "src/*.c" "src/*.cxx" "src/*.cu")
#file(GLOB_RECURSE original_headers "include/*.h" "include/*.hpp")

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


target_link_libraries(${PROJECT_NAME} cmdline)
target_link_libraries(${PROJECT_NAME} copylocal)
target_link_libraries(${PROJECT_NAME} fs)
target_link_libraries(${PROJECT_NAME} linalg_vectors)
target_link_libraries(${PROJECT_NAME} linalg_matrices)
target_link_libraries(${PROJECT_NAME} random)
target_link_libraries(${PROJECT_NAME} timers)
target_link_libraries(${PROJECT_NAME} system)

configureCompileFlags()
