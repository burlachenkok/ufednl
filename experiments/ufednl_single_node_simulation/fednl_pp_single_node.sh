#!/usr/bin/env bash

rounds=1000
workers=12
server_grad_workers=4
server_hessian_workers=4
clients_per_round=12

bin="./../../dopt/scripts/build_release/bin_fednl_local_pp/bin_fednl_local_pp"
${bin} --version

for datashort in "w8a" "a9a" "phishing"
do 

echo "Processing dataset: ${datashort}"
echo ""

echo "..FedNL-PP TOPK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a-49700 --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor topk --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_topk_option_2.bin --silent 1>benchmark_pp_topk_${datashort}.stdout

echo "..FedNL-PP RANDK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a --add_intercept --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --clients-per-round ${client_per_round--lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor randk --theoretical_alpha --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randk.bin --silent 1>benchmark_pp_randk_${datashort}.stdout

echo "..FedNL-PP TOPLEK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a --add_intercept --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --clients-per-round ${client_per_round--lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor toplek --theoretical_alpha  --use_theoretical_alpha_option_2 --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_toplek_option_2.bin --silent 1>benchmark_pp_toplek_${datashort}.stdout

echo "..FedNL-PP RANDSEQK ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a --add_intercept --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --clients-per-round ${client_per_round--lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor seqk --theoretical_alpha --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_randseqk.bin --silent 1>benchmark_pp_seqk_${datashort}.stdout

echo "..FedNL-PP IDENTICAL ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a --add_intercept --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --clients-per-round ${client_per_round--lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor seqk --theoretical_alpha --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_identical.bin --silent 1>benchmark_pp_identical_${datashort}.stdout

echo "..FedNL-PP NATURAL ${datashort}.."
${bin} --server_grad_workers ${server_grad_workers} --server_hessian_workers ${server_hessian_workers} --workers ${workers} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ./../dopt/datasets/w8a --add_intercept --reshuffle_train --clients 142 --clients-per-round ${clients_per_round} --clients-per-round ${client_per_round--lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor natural --theoretical_alpha --out result_pp_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_natural.bin --silent 1>benchmark_pp_natural_${datashort}.stdout

done

echo "Completed successfully"
