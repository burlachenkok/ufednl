#!/usr/bin/env bash

rounds=100
compressor=identical

bin="./../dopt/scripts/build_release/bin_fednl_distr_master/Release/bin_fednl_distr_master.exe"

echo ""
echo "TEST FOR $version. START."

${bin} --version

echo "..RANDK.."
${bin} --algorithm gd --k_compressor_as_d_mult 8 --compressor ${compressor} --train --clients 10 --lambda 0.001 --rounds ${rounds} --tracking --global_lr 0.01 --theoretical_alpha --use_theoretical_alpha_option_1 --fednl-option-b --out result.bin --master tcpv4:localhost:3212 --compute-L-smooth --theoretical_global_lr

echo "Completed successfully"
