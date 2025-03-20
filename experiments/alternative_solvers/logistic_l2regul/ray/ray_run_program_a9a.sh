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
echo "LISTENING TCP PORTS" >> $log
netstat -nap | grep -E "tcp(.)*LISTEN" >> $log
echo "Ephemeral ports range: " >> $log
cat /proc/sys/net/ipv4/ip_local_port_range >> $log

#===================================================================================================
ntasks=${SLURM_NTASKS}
nclients=$((${ntasks}-1))

role="server"
client_number=0
bin="python3 solve_distr_a9a.py"

if ((SLURM_PROCID == 0)); then
  export role="server"
  bin="python3 solve_distr_a9a.py" 
  export is_master=1
else
  client_number=$((SLURM_PROCID - 1))
  export role="client:${client_number}"
  bin="python3 solve_distr_a9a.py"
fi

master_host=`echo ${SLURM_JOB_NODELIST} | tr "," "\n" | head -n1`
master_port_base=6379

#===================================================================================================
echo "ROLE: ${role}" >> $log
echo "BINARY: ${bin}" >> $log
echo "SERVER HOSTNAME: ${master_host}:${master_port_base}" >> $log
#===================================================================================================

#===================================================================================================
if ((SLURM_PROCID == 0)); then
    echo "-----------------------------------------" >> ${log}
    ray start --head --port=${master_port_base} --num-cpus 1 --num-gpus 0 >> ${log}
    echo "-----------------------------------------" >> ${log}

    touch sig_server_is_started_${SLURM_JOB_ID}
    sync
    
    echo "Wait for clients" >> $log

    for (( ; ; ))
    do
	nodes=`ray status | grep node_ | wc -l`
	if (($nodes == $ntasks))
        then
	    echo "All ${nodes} clients are connected to master" >> $log
  	    break
        else
	    echo "Current number of connected clients: $nodes" >> $log
	fi
    done
    echo $bin

    export RAY_ADDRESS="${master_host}:${master_port_base}"
    ray job submit --working-dir . -- ${bin} >> $log 2>&1

    #export RAY_ADDRESS='http://127.0.0.1:8265' 
    #ray job submit --working-dir . -- python3 solve_distr_a9a.py

    touch sig_server_is_stopped_${SLURM_JOB_ID}
    sync

    ray stop >> $log
    echo "Server has finished it's work" >> $log
else
    echo "Wait for server to start" >> $log
    # Wait for server to start
    for (( ; ; ))
    do
	if [[ -f sig_server_is_started_${SLURM_JOB_ID} ]]
	then
	  echo "Server has been started" >> $log
	  break
        fi
    done
    echo "-----------------------------------------" >> ${log}
    ray start --address=${master_host}:${master_port_base} --num-cpus 1 --num-gpus 0 --block >> ${log}
    echo "-----------------------------------------" >> ${log}
    echo "Client agent is activated" >> $log

   
    for (( ; ; ))
    do
	if [[ -f sig_server_is_stopped_${SLURM_JOB_ID} ]]
	then
	  echo "Server has been stopped" >> $log
	  break
	fi
    done
    ray stop >> $log
    echo "Client has finished it's work" >> $log
fi
#===================================================================================================
