#!/bin/bash
#SBATCH --job-name=fednl-test
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=51
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --constraint="(cascadelake|skylake)"

srun bash natural_fp.sh


