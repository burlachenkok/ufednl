#!/bin/bash
#SBATCH --job-name=fednl-spark-test
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=51
#SBATCH --ntasks-per-node=1


#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G

#SBATCH --mail-type=ALL
#SBATCH --open-mode=append

export dataset="w8a"
srun bash spark_run_program.sh
