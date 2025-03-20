# u-FedNL: Minimal Environment

----

The goal of this document is to describe how to prepare the main build and runtime environment for building our project and provide needed runtimes for alternative solutions in Windows, Linux, and macOS.

----

# Prepare to Build

## If you are Working under Windows OS

1. Install Visual Studio 2022 (or newer) in your Windows OS. To install Microsoft Visual Studio please visit the Microsoft website [https://visualstudio.microsoft.com/vs/](https://visualstudio.microsoft.com/vs/) and follow Microsoft instructions.

2. Install/Update CMake to version 3.12 or higher from https://cmake.org/download/

## If you are Working under Linux OS

1. Install GCC-11. See GCC release: https://gcc.gnu.org/releases.html.

```bash
sudo apt-get install gcc-11 g++-11
```

If your Linux distribution does not have Advanced Package Tool (Apt) Package Manager, please use a similar tool distributed with your Operating System.

2. To get recent versions of CMake under at least Ubuntu Linux distributive in 2024 is not so easy. To get it you will need to do some manual work:

* If your CPU has ARM/AArch64 architecture execute the following:

```bash
sudo apt remove cmake
# For AArch64 CPUs
fname=cmake-3.27.0-rc5-linux-aarch64.sh
wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc5/${fname}
sudo cp $fname /opt/
cd /opt
sudo bash $fname
# For AArch64 CPUs
sudo ln -s /opt/cmake-3.27.0-rc5-linux-aarch64/bin/cmake /usr/local/bin/
```

* If your CPU has x86-64 architecture execute the following:


```bash
sudo apt remove cmake
# For Intel/AMD CPUs
fname=cmake-3.27.0-rc5-linux-x86_64.sh
wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc5/${fname}
sudo cp $fname /opt/
cd /opt
sudo bash $fname
# For Intel/AMD CPUs
sudo ln -s /opt/cmake-3.27.0-rc5-linux-x86_64/bin/cmake /usr/local/bin/
```

## If you are Working under macOS

1. Install GCC-11 for example from [brew](https://brew.sh/). See GCC release: https://gcc.gnu.org/releases.html.
```bash
brew install gcc@11
```

2. Install the recent version of CMake in macOS. For our project, CMake 3.12 is enough:
```bash
brew install cmake
```

## Installations for Benchmarking

### Prepare Python and Conda Environment

We do not use Python, however, CVXPY and RAYs/SkLearn are used via the Python interface. To not collude with the system good practice is to install Python and its
packages via an extra layer of package managers. One popular solution is Conda.

If you don't have the Conda package and environment manager you can install them via the following steps for Linux OS:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="${PATH}:~/miniconda3/bin"
~/miniconda3/bin/conda init bash && source ~/.bashrc && conda config --set auto_activate_base false
```

For refresh conda commands, you can look into the official [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

### Install CVXPY and RAYS

For Windows OS, Linux OS, and macOS please use the following commands to prepare the environment.

```bash
conda create -n dopt python=3.9.1 -y
conda activate dopt
# Need packages for experiments using CVXPY
python3 -m pip install Cython clarabel cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS] matplotlib pyspark numpy matplotlib pandas scikit-learn
# Need packages for experiments with using RAYS
python3 -m pip install "ray[all]"
```

### Install Commerical MOSEK via Academic License

The [CVXPY](https://www.cvxpy.org/) supports [MOSEK](https://www.mosek.com/), which is a commercial optimization software.

Fortunately, we have the ability to obtain an academic license.

To install Mosek you need to perform the following steps:

1. Execute: `python3 -m pip install Mosek`
2. Fill out the needed forms to obtain in the Mosek website to obtain the license file: `mosek.lic`
3. Create folder in home directory folder `mosek` via `mkdir ${HOME}/mosek` in Linux and macOS and as `mkdir %USERPROFILE%/mosek` in Windows OS.
4. Physically put `mosek.lic` into the created folder `mosek`.

### Install Apache Spark

To install Apache Spark on Posix, follow these general steps (assuming you have the appropriate version of the installed Java Virtual Machine).

The exact commands may vary slightly depending on your OS, but mainly you should follow the following steps:

```bash
curl -O https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar xvf spark-3.5.0-bin-hadoop3.tgz
sudo mkdir /opt/spark
sudo chmod -R 777 /opt/spark
sudo mv spark-3.5.0-bin-hadoop3/* /opt/spark/
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
source ~/.bashrc
```

To run Apache Spark in our machine we had OpenJDK Runtime Environment 11.0.19, part of the Java 11 series.

Next, you should activate the master node via the command:
```bash
/opt/spark/sbin/start-master.sh
```

* 7077 - is a default TCP port for MASTER in Apache Spark.
* 8080 - is a default TCP port HTTP interface for monitoring.

After this, you need to activate clients. To achieve this on client hosts you should execute the command:
```bash
start-worker.sh <master-host-name>:7077
```

To stop the master in Apache Spark there is a similar command:
```bash
/opt/spark/sbin/stop-master.sh
```

**Remark:**  We provide launch information just in case you will have any problems with launches. None of these commands for experiment reproducing should be launched manually.
