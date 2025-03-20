set(GIT_LAUNCH_WD ${CMAKE_CURRENT_SOURCE_DIR})
#====================================================================================
# SHA-1 of last git change commit
#====================================================================================
set(cmdline git rev-parse HEAD)
execute_process(COMMAND ${cmdline} RESULT_VARIABLE my_result OUTPUT_VARIABLE my_out WORKING_DIRECTORY ${GIT_LAUNCH_WD})
if(NOT ${my_result} EQUAL 0)
    set(my_out "undefined")
endif(NOT ${my_result} EQUAL 0)
string(REPLACE "\n" "" my_out ${my_out})
set(DOPT_LAST_CHANGE ${my_out})
#====================================================================================
# SHA-1 of previous git change commit
#====================================================================================
set(cmdline git rev-parse HEAD~1)
#execute_process(COMMAND ${cmdline} RESULT_VARIABLE my_result OUTPUT_VARIABLE my_out WORKING_DIRECTORY ${GIT_LAUNCH_WD})
#if(NOT ${my_result} EQUAL 0)
#    set(my_out "undefined")
#endif(NOT ${my_result} EQUAL 0)
#string(REPLACE "\n" "" my_out ${my_out})
#set(DOPT_PREVIOUS_CHANGE ${my_out})
#====================================================================================
# Current branch for git repo
#====================================================================================
set(cmdline git rev-parse --abbrev-ref HEAD)
execute_process(COMMAND ${cmdline} RESULT_VARIABLE my_result OUTPUT_VARIABLE my_out WORKING_DIRECTORY ${GIT_LAUNCH_WD})
if(NOT ${my_result} EQUAL 0)
    set(my_out "undefined")
endif(NOT ${my_result} EQUAL 0)
string(REPLACE "\n" "" my_out ${my_out})
set(DOPT_BRANCH_NAME ${my_out})
#====================================================================================
# Get last change commit data
#====================================================================================
set(cmdline git log -n1 --date=short --pretty=format:%cD)
execute_process(COMMAND ${cmdline} RESULT_VARIABLE my_result OUTPUT_VARIABLE my_out WORKING_DIRECTORY ${GIT_LAUNCH_WD})
if(NOT ${my_result} EQUAL 0)
    set(my_out "undefined")
endif(NOT ${my_result} EQUAL 0)
string(REPLACE "\n" "" my_out ${my_out})
set(DOPT_LAST_CHANGE_DATE ${my_out})
#====================================================================================
# Get remote repository name
#====================================================================================
set(cmdline git remote get-url --push origin)
execute_process(COMMAND ${cmdline} RESULT_VARIABLE my_result OUTPUT_VARIABLE my_out WORKING_DIRECTORY ${GIT_LAUNCH_WD})
if(NOT ${my_result} EQUAL 0)
    set(my_out "undefined")
endif(NOT ${my_result} EQUAL 0)
string(REPLACE "\n" "" my_out ${my_out})
set(DOPT_DEFAULT_REPOSITORY_URL ${my_out})

#====================================================================================
# Configure file
#====================================================================================
# https://cmake.org/cmake/help/v3.0/command/configure_file.html
#  This command replaces any variables in the input file referenced as ${VAR} or @VAR@ 
#  with their values as determined by CMake. If a variable is not defined, it will be replaced with nothing.

configure_file("${DOPT_PROJECT_ROOT}/dopt/system/include/Version.h.in"
               "${DOPT_PROJECT_ROOT}/dopt/system/include/Version.h")
message(STATUS "${DOPT_PROJECT_ROOT}/dopt/system/include/Version.h have been configured...")

#configure_file("${DOPT_PROJECT_ROOT}/dopt/system/include/BuildConfiguration.h.in"
#               "${DOPT_PROJECT_ROOT}/dopt/system/include/BuildConfiguration.h")
#message(STATUS "${DOPT_PROJECT_ROOT}/dopt/system/include/BuildConfiguration.h have been configured...")
#====================================================================================
