#!/usr/bin/env bash

rounds=1000
workers=12
server_grad_workers=4
server_hessian_workers=4

bin="./../../dopt/scripts/build_release/bin_fednl_local/bin_fednl_local"

if [[ -f $bin ]]; then
   echo "Used binary: ${bin}"
else
   bin="./../../dopt/scripts/build_release/bin_fednl_local/Release/bin_fednl_local"
   if [[ -f $bin ]]; then
     echo "Used binary: ${bin}"
   else
     echo "Executable binary can not be found"
     exit -1
   fi
fi

${bin} --version

for datashort in "w8a" "a9a" "phishing"
do 

echo "Processing dataset: ${datashort}"
echo ""

dataset="./../../dopt/datasets/${datashort}"

echo "..FedNL TOPK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor topk --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_topk_option_2.bin --silent 1>behcnmark_topk_${datashort}.stdout

echo "..FedNL RANDK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor randk --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk.bin  --silent 1>behcnmark_randk_${datashort}.stdout

echo "..FedNL TOPLEK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor toplek --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_toplek_option_2.bin --silent 1>behcnmark_toplek_${datashort}.stdout

echo "..FedNL RANDSEQK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor seqk --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randseqk.bin  --silent 1>behcnmark_seqk_${datashort}.stdout
echo ""

echo "..FedNL IDENTICAL ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor identical --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_ident.bin  --silent 1>behcnmark_ident_${datashort}.stdout
echo ""

echo "..FedNL NATURAL ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor natural --theoretical_alpha --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_natural.bin --silent 1>behcnmark_natural_${datashort}.stdout
echo ""

done

echo "Completed successfully"
