#!/usr/bin/env bash

export log="./worker_${SLURM_PROCID}_job_${SLURM_JOBID}.txt"


echo "========================================================" >> $log
echo "SLURM INFORMATION" >> $log
echo "  Job Name: ${SLURM_JOB_NAME}" >> $log
echo "  Job ID: ${SLURM_JOB_ID}" >> $log
echo "  SLURM cluster name: ${SLURM_CLUSTER_NAME}" >> $log
echo "  Total number of tasks: ${SLURM_NTASKS}" >> $log
echo "  Number of tasks per node: ${SLURM_NTASKS_PER_NODE}" >> $log
echo "  CPU cores per task: ${SLURM_CPUS_PER_TASK}" >> $log
echo "  List of nodes: ${SLURM_JOB_NODELIST}" >> $log
echo "  AVAILABLE CPU CORES" >> $log
taskset -cp `bash -c 'echo $PPID'` >> $log
echo "  PROCESS ID[RANDK]" >> $log
echo ${SLURM_PROCID} >> $log
echo "" >> $log

echo "========================================================" >> $log
echo "NETWORK INFORMATION" >> $log
echo "  HOSTNAME" >> $log
hostname >> $log
echo "" >> $log
echo "  LIST OF NETWORK INTERFACES WITH MTU" >> $log
netstat -i >> $log
echo "" >> $log

echo "========================================================" >> $log
echo "INSTALLED SOFTWARE" >> $log
echo "   Computer architecture: $(arch)" >> $log
echo "   GCC version: $(gcc --version)" >> $log
echo "   Python version: $(python --version)" >> $log
echo "" >> $log
echo "========================================================" >> $log
echo "INSTALLED CPU" >> $log
lscpu >> $log
lscpu --extended >> $log
echo "Physical And Swap Memory" >> $log
free -h >> $log
echo "" >> $log

echo "=======================================================================" >> $log
echo "EXTA SYSTEM COMPUTE INFORMATION" >> $log
./../../../../dopt/scripts/build_release/utils/bin_host_view/bin_host_view  >> $log
echo "" >> $log
echo "=======================================================================" >> $log
echo "LISTENING TCP PORTS" >> $log
netstat -nap | grep -E "tcp(.)*LISTEN" >> $log
echo "Ephemeral ports range: " >> $log
cat /proc/sys/net/ipv4/ip_local_port_range >> $log

#===================================================================================================
ntasks=${SLURM_NTASKS}
nclients=$((${ntasks}-1))

role="server"
client_number=0
bin="./../../../../dopt/scripts/build_release/bin_fednl_distr_client/bin_fednl_distr_client"

if ((SLURM_PROCID == 0)); then
  role="server"
  bin="./../../../../dopt/scripts/build_release/bin_fednl_distr_master/bin_fednl_distr_master"
else
  client_number=$((SLURM_PROCID - 1))
  role="client:${client_number}"
  bin="./../../../../dopt/scripts/build_release/bin_fednl_distr_client/bin_fednl_distr_client"
fi

master_host=`echo ${SLURM_JOB_NODELIST} | tr "," "\n" | head -n1`
master_port_base=34234

#===================================================================================================
echo "ROLE: ${role}" >> $log
echo "BINARY: ${bin}" >> $log
echo "SERVER HOSTNAME: ${master_host}:${master_port_base}" >> $log
#===================================================================================================
rounds=400
dataset="./../../../../dopt/datasets/phishing"
extra_flags="--silent"

#compressor="toplek"
#compressor="randk"
#compressor="seqk"
#compressor="identical"
compressor="natural"

${bin} --server_grad_workers 0 --server_hessian_workers 0 --k_compressor_as_d_mult 8 --algorithm fednl1 --train --train_dataset ${dataset} --add_intercept --reshuffle_train --clients ${nclients} --lambda 0.001 --fednl-option-b --rounds ${rounds} --tracking --check_split --compressor ${compressor} --theoretical_alpha --use_theoretical_alpha_option_2 --out "result_run_{run_id}_{algorithm}_{dataset}_m{kmultiplier}_${compressor}_option_2.bin" --master tcpv4:${master_host}:${master_port_base} --iam ${role} ${extra_flags} 1 >> $log
#===================================================================================================
