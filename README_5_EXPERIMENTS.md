# u-FedNL: Reproduce Experiments

----

The goal of this document is to describe how after preparing the environment, build the project, carry unit tests in your platform, and launch experiments.

----

# BaseLine
## Launch BaseLine Python code for FedNL from the Original Work

We used the code from [https://github.com/Rustem-Islamov/FedNL-Public](https://github.com/Rustem-Islamov/FedNL-Public).

However, we augmented it with two bootstrap scripts.

To launch experiments with training logistic regression via FedNL (Python/NumPy) with **RandK[k=8d]** in the `W8A` dataset for `1000` rounds you should execute the:

```bash
cd ./experiments/experiments/baseline_python_fednl/
python3 test_run_rand_k.py
```

To launch experiments with training logistic regression via FedNL (Python/NumPy) with **TopK[k=8d]** in the `W8A` dataset for `1000` rounds you should execute the:

```bash
cd ./experiments/experiments/baseline_python_fednl/
python3 test_run_top_k.py
```

# u-FedNL

## u-FedNL: Launch Single-Node Simulation with FedNL

First change the working directory to this path:
```bash
cd ./experiments/ufednl_single_node_simulation
```

Then to launch **FedNL local node simulation** with n=142 clients during r=1000 rounds for solving Logistic Regression on LIBSVM datasets:
```bash
bash fednl_single_node.sh
```

To launch **FedNL-LS local node simulation** with n=142 clients during r=1000 rounds for solving Logistic Regression on LIBSVM datasets:
```bash
bash fednl_ls_single_node.sh
```

To launch **FedNL-PP local node simulation** with n=142 total clients  and s=12 clients per round during r=1000 rounds for solving Logistic Regression on LIBSVM datasets:
```bash
bash fednl_pp_single_node.sh
```

## u-FedNL: Launch Multi-Node Simulation with FedNL

To launch FedNL, FedNL-PP, and FedNL-LS in the [Slurm](https://slurm.schedmd.com/documentation.html) management system in the multi-node setting we have prepared a series of startup scripts.
To launch this you should have a Configured [Slurm](https://slurm.schedmd.com/documentation.html) Cluster management system and working cluster.

Once you have please build the project in the already described way:
```bash
./project_scripts.py -c -gr -br -tr -j 48
```

You need to switch to the working directory:
```bash
cd ./experiments/ufednl_multi_node_simulation/
```

Select of specific settings in which you are interested and launch the job via `batch <name>.sbatch`.

The output of execution is the following files:
* The results of the simulation are saved into `*.bin`
* Log files are available in `*.txt`, and `*.out`.

Complete list [Slurm](https://slurm.schedmd.com/documentation.html) configuration files:

* ./a9a/fp/natural_fp.slurm
* ./a9a/fp/randk_fp.slurm
* ./a9a/fp/seqk_fp.slurm
* ./a9a/fp/toplek_fp.slurm
* ./a9a/ls/natural_ls.slurm
* ./a9a/ls/randk_ls.slurm
* ./a9a/ls/seqk_ls.slurm
* ./a9a/ls/toplek_ls.slurm
* ./a9a/pp/natural_pp.slurm
* ./a9a/pp/randk_pp.slurm
* ./a9a/pp/seqk_pp.slurm
* ./a9a/pp/toplek_pp.slurm
* ./phishing/fp/natural_fp.slurm
* ./phishing/fp/randk_fp.slurm
* ./phishing/fp/seqk_fp.slurm
* ./phishing/fp/toplek_fp.slurm
* ./phishing/ls/natural_ls.slurm
* ./phishing/ls/randk_ls.slurm
* ./phishing/ls/seqk_ls.slurm
* ./phishing/ls/toplek_ls.slurm
* ./phishing/pp/natural_pp.slurm
* ./phishing/pp/randk_pp.slurm
* ./phishing/pp/seqk_pp.slurm
* ./phishing/pp/toplek_pp.slurm
* ./w8a/fp/natural_fp.slurm
* ./w8a/fp/randk_fp.slurm
* ./w8a/fp/seqk_fp.slurm
* ./w8a/fp/toplek_fp.slurm
* ./w8a/ls/natural_ls.slurm
* ./w8a/ls/randk_ls.slurm
* ./w8a/ls/seqk_ls.slurm
* ./w8a/ls/toplek_ls.slurm
* ./w8a/pp/natural_pp.slurm
* ./w8a/pp/randk_pp.slurm
* ./w8a/pp/seqk_pp.slurm
* ./w8a/pp/toplek_pp.slurm

##  u-FedNL: Visualize Results of Experiments from Utilizing FedNL Implementation

The results of experiments are saved in the file in a (custom) binary format.

To assist with parsing the file and visualizing plots we have prepared the script `show.py`.

To use this script please call it as follows:
```bash
python3 experiments/show.py <file_1.bin> <file_2.bin> ...
```

This script utilizes [Matplotlib](https://matplotlib.org/) functionality to visualize plots, and it has two optional flags:

* `--save-fig` will force save figures in pdf format in the host where the script was invoked into the current working directory.
* `--no-gui` will force the visualize script to not activate GUI. It may be helpful if the host does not have a GUI interface.


# Competitors

## CVXPY: Launch Single-Node CVXPY Solvers Experiments

Firstly you may wish to check that all solvers for CVXPY were installed correctly. Please active your Conda environment and execute the:
```bash
cd ./experiments/alternative_solvers/logistic_l2regul/cvx/cvx_info.py
python3 cvx_info.py
```

If all is installed fine you can invoke solving Logistic Regression in your machine via [CLARABEL](https://clarabel.org/stable/), [ECOS](https://github.com/embotech/ecos), [ECOS_BB](https://github.com/embotech/ecos), [SCS](https://www.cvxgrp.org/scs/), and [MOSEK](https://www.mosek.com/) solvers in the following way:

```bash
python3 cvx_solve_a9a.py
python3 cvx_solve_phishing.py
python3 cvx_solve_w8a.py
```

The selection of solvers is defined in `used_solvers` variables.
The measured solve time is measured both with [CVXPY](https://www.cvxpy.org/) internal mechanisms.
In the paper, we reported seconds from CVXPY internals.

Unfortunately [GUROBI](https://www.gurobi.com/) solver, a popular commercial solver for Mixed Integer Programming (MIP) tasks and others
does not support the logistic regression objective.

## Ray: Launch Experiments with Ray

To utilize [Ray](https://www.ray.io/) we also have prepared bootstrap scripts for Slurm Management System.
After that change the working directory to:
```bash
cd ./experiments/alternative_solvers/logistic_l2regul/ray
```

Scripts:
* `solve_distr_a9a.py`
* `solve_distr_phishing.py`
* `solve_distr_w8a.py`

Ray unfortunately not always work fine with relative filesystem paths.
Therefore we recommend changing the code in the script in the sense that explicitly specifies full paths for the datasets.
This file path should be valid for all workers in the RAY Cluster.

After this launch one of the available configurations:
* `sbatch ray_run_a9a.slurm`
* `sbatch ray_run_phishing.slurm`
* `sbatch ray_run_w8a.slurm`

## Apache Spark: Launch Experiments with Apache Spark

To utilize Apache Spark we also have prepared bootstrap scripts for Slurm Management System.
Please change the working directory to:
```bash
cd ./experiments/alternative_solvers/logistic_l2regul/spark
```

Firstly to check all is fine with your installation you can run Apache Spark locally via:
```bash
python3 try_local.py
```

Next, once you find that all is fine you can further launch one of the available configurations:

* `sbatch spark_run_a9a.slurm`
* `sbatch spark_run_phishing.slurm`
* `sbatch spark_run_w8a.slurm`
