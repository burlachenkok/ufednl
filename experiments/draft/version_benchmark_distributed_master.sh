#!/usr/bin/env bash

rounds=30

dataset="./../dopt/datasets/w8a"

bin="./../dopt/scripts/build_release/bin_fednl_distr_master/bin_fednl_distr_master"

role="server"

master_host=kw60797

master_port_base=34234

nclients=1

echo ${bin} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${nclients} --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor topk --theoretical_alpha --use_theoretical_alpha_option_2 --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_topk_option_2.bin --master tcpv4:${master_host}:${master_port_base} --iam ${role}

${bin} --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${nclients} --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor topk --theoretical_alpha --use_theoretical_alpha_option_2 --out result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_topk_option_2.bin --master tcpv4:${master_host}:${master_port_base} --iam ${role}
