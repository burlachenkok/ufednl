#!/usr/bin/env bash

export log="./worker_${SLURM_PROCID}_job_${SLURM_JOBID}.txt"

echo "========================================================" >> $log
date >> $log
echo "" >> $log
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
bin="python3 solve_distr.py"

if ((SLURM_PROCID == 0)); then
  export role="server"
  bin="python3 solve_distr.py" 
  export is_master=1
else
  client_number=$((SLURM_PROCID - 1))
  export role="client:${client_number}"
  bin="python3 solve_distr.py"
fi

export master_host=`echo ${SLURM_JOB_NODELIST} | tr "," "\n" | head -n1`

#===================================================================================================
echo "ROLE: ${role}" >> $log
echo "BINARY: ${bin}" >> $log
echo "SERVER HOSTNAME: ${master_host}:${master_port_base}" >> $log
#===================================================================================================

date >> $log
echo "" >> $log

#===================================================================================================
if ((SLURM_PROCID == 0)); then
    echo "-----------------------------------------" >> ${log}
    bash start-master.sh >> ${log}
    echo "-----------------------------------------" >> ${log}

    touch sig_server_is_started_${SLURM_JOB_ID}
    sync
    
    echo "Wait for clients" >> $log

    for (( ; ; ))
    do
	nodes=`ls -1 sig_client_is_started_${SLURM_JOB_ID}_* | wc -l`
	if (($nodes == $nclients))
        then
	    echo "All ${nodes} clients are connected to master" >> $log
  	    break
        else
	    echo "Current number of connected clients: $nodes" >> $log
            sleep 1
	fi
    done
    echo $bin

    python3 init_distr.py >> $log 2>&1

    stop-master.sh >> $log
    touch sig_server_is_stopped_${SLURM_JOB_ID}
    sync
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
    start-worker.sh ${master_host}:7077 >> ${log}
    echo "-----------------------------------------" >> ${log}

    touch  sig_client_is_started_${SLURM_JOB_ID}_client_${SLURM_PROCID}
    sync
    echo "Client agent is activated" >> $log
    sleep 1
   
    for (( ; ; ))
    do
	if [[ -f sig_server_is_stopped_${SLURM_JOB_ID} ]]
	then
	  echo "Server has been stopped" >> $log
	  break
	fi
    done
    stop-worker.sh >> $log
    echo "Client has finished it's work" >> $log
fi
#===================================================================================================
