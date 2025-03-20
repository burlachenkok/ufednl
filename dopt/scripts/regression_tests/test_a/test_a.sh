#!/usr/bin/env bash

exit 0

set -e

#============================================================================
fednl_local="./../../build_release/bin_fednl_local/bin_fednl_local"

if [ ! -e ${fednl_local} ];
then
    fednl_local="./../../build_release/bin_fednl_local/Release/bin_fednl_local"
fi
#============================================================================
dataset="./../../../datasets/phishing"

export rounds_arr=(15 5)
export clients_arr=(5 10)

echo "Description:"
echo "Test that FedNL with TopK, RandK (with the same x0) reduce to Identical compressor in case of using TopK and RandK with specific K multiplier which makes TopK and RandK no compress at all. Last iterate should be the same bit to bit."
echo ""
echo "Binary: ${fednl_local}"
echo "Regression tests `basename $0` for datataset ${dataset} has been started"

#============================================================================
for clients in "${clients_arr[@]}"
do
for rounds in "${rounds_arr[@]}"
do
#============================================================================
${fednl_local} --lambda 0.001 --transfer_indicies_for_randk --compressor randk --k_compressor_as_d_mult 35.5 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${clients} --fednl-option-b --rounds ${rounds} --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_0.txt
${fednl_local} --lambda 0.001 --compressor randk --k_compressor_as_d_mult 35.5 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${clients} --fednl-option-b --rounds ${rounds} --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_1.txt
${fednl_local} --lambda 0.001 --use_theoretical_alpha_option_1 --compressor topk --k_compressor_as_d_mult 35.5 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${clients} --fednl-option-b --rounds ${rounds} --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_2.txt
${fednl_local} --lambda 0.001 --use_theoretical_alpha_option_2 --compressor topk --k_compressor_as_d_mult 35.5 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${clients} --fednl-option-b --rounds ${rounds} --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_3.txt
${fednl_local} --lambda 0.001 --k_compressor_as_d_mult 1.0 --compressor identical --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${clients} --fednl-option-b --rounds ${rounds} --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_4.txt

diff -u launch_0.txt launch_1.txt
diff -u launch_1.txt launch_2.txt
diff -u launch_2.txt launch_3.txt
#diff -u launch_3.txt launch_4.txt

echo "Regression test for clients=${clients}, rounds=${rounds} has been finished. [OK]"
#============================================================================
done
done
#============================================================================

#${fednl_local} --lambda 0.001 --compressor randk --k_compressor_as_d_mult 35.5 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 10 --fednl-option-b --rounds 12 --tracking --check_split --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_with_ind.bin --silent | grep "CRC-32" > launch_separate.txt
#diff -u launch_separate.txt baseline.txt

echo ""
echo "Regression tests `basename $0` for dataset ${dataset} has been finished [OK]"
