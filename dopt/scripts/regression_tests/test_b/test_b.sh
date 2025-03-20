#!/usr/bin/env bash

set -e

echo "Description:"
echo "Test launchability of build binaries"
echo ""

echo "============================================================================"
fednl_local="./../../../scripts/build_release/bin_fednl_local/bin_fednl_local"
if [ ! -e ${fednl_local} ];
then
    fednl_local="./../../../scripts/build_release/bin_fednl_local/Release/bin_fednl_local"
fi
echo "Binary: ${fednl_local}"
echo ""
${fednl_local} --version
echo ""

dopt_utests="./../../../scripts/build_release/bin_tests/bin_tests"
if [ ! -e ${dopt_utests} ];
then
    dopt_utests="./../../../scripts/build_release/bin_tests/Release/bin_tests.exe"
fi
echo "Binary: ${dopt_utests}"
echo ""
echo "Number of Unit Tests: " `"${dopt_utests}" --gtest_list_tests | grep GTest | wc -l`
echo "Number of Performace Tests: " `"${dopt_utests}" --gtest_list_tests | grep GPerf | wc -l`
echo ""

echo "============================================================================"
sysview_bin="./../../../scripts/build_release/utils/bin_host_view/bin_host_view"
if [ ! -e ${sysview_bin} ];
then
    sysview_bin="./../../../scripts/build_release/utils/bin_host_view/Release/bin_host_view"
fi
echo "Binary: ${sysview_bin}"
echo ""
${sysview_bin}
echo ""

echo "Regression tests `basename $0` for dataset ${dataset} has been finished [OK]"
