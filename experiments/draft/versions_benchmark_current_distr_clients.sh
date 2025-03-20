#!/usr/bin/env bash

rounds=100
compressor=identical

bin="./../dopt/scripts/build_release/bin_fednl_distr_client/Release/bin_fednl_distr_client.exe"

echo ""
echo "TEST FOR $version. START."

${bin} --version

echo "..RANDK.."
for i in {0..9}
do
   ${bin} --algorithm gd --k_compressor_as_d_mult 8 --compressor ${compressor} --train --train_dataset C:/projects/new/fednl_impl/dopt/datasets/a1a --add_intercept --reshuffle_train --clients 10 --lambda 0.001 --rounds ${rounds} --silent --tracking --global_lr 0.01 --theoretical_alpha --use_theoretical_alpha_option_1 --fednl-option-b --out result.bin --master tcpv4:localhost:3212 --iam client:${i} --compute-L-smooth --theoretical_global_lr&
   echo "Client $i has been launched"
done

wait

echo "Completed successfully"
