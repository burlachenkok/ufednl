#!/usr/bin/env bash

rounds=1000
workers=12
server_grad_workers=4
server_hessian_workers=4

bin="./../../dopt/scripts/build_release/bin_fednl_local/bin_fednl_local"

${bin} --version

for datashort in "w8a" "a9a" "phishing"
do 

echo "Processing dataset: ${datashort}"
echo ""

dataset="./../../dopt/datasets/${datashort}"

echo "..FedNL-LS TOPK ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor topk --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_topk_option_2_${datashort}.bin --silent 1>bench_topk_ls_${datashort}.stdout

echo "..FedNL-LS RANDK ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor randk --theoretical_alpha --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk_${datashort}.bin --silent 1>bench_randk_ls_${datashort}.stdout

echo "..FedNL-LS TOPLEK ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor toplek --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_toplek_option_2_${datashort}.bin --silent 1>bench_toplek_ls_${datashort}.stdout

echo "..FedNL-LS RANDSEQK ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor seqk --theoretical_alpha --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randseqk_${datashort}.bin --silent 1>bench_seqk_ls_${datashort}.stdout

echo "..FedNL-LS IDENTICAL ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor identical --theoretical_alpha --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_idenical_${datashort}.bin --silent 1>bench_idenical_ls_${datashort}.stdout

echo "..FedNL-LS NATURAL ${datashort}.."
${bin} --line_search --c_line_search 0.49 --gamma_line_search 0.5 --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients 142 --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor natural --theoretical_alpha --out result_ls_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_natural_${datashort}.bin --silent 1>bench_natural_ls_${datashort}.stdout

done

echo "Completed successfully"
