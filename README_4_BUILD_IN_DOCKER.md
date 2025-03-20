# u-FedNL: Build in Docker

----

The goal of this document is to describe how instead of Step 3 build a project in the docker container.
It can be useful to build and try code in another (foreign) CPU architecture.

----

# About Docker

## Installation

To install docker use provided information from https://docs.docker.com/engine/install/ and https://docs.docker.com/:

## Overview

Docker is a platform for build, run, and share applications.

* **Docker daemon** - The Docker daemon (dockerd) listens for Docker API requests.
* **Docker client** - The Docker client (docker) is the primary way that many Docker users interact
  with Docker. Typically via console interface.
* **Docker Registries** - A Docker registry stores Docker images.
* **Docker Hub** – common repository with docker-images. To use them you need to install docker in your local
  machine.
* **Docker Image** - The read-only template with instructions for creating a Docker container. The image does not have any runtime behavior. An image includes
  everything needed to run an application. You can list docker images via `docker image ls`.
* **Docker Container** - a runnable instance of an image. You can: create, start, stop, move, delete a container, connect to the network, attach storage, and create a new image from the container. You can list all containers via `docker container ls --all`.

Docker command line interface to work with images and containers:
https://docs.docker.com/engine/reference/commandline/cli/

Dockerfile commands to create new images:
https://docs.docker.com/engine/reference/builder/

## Internals

A container almost always is nothing but a running process (first-order approximation).
A container by docker is kept isolated from the host and other containers. Each
container interacts with its private filesystem.

To implement it Docker leverages several Linux features:
1. Docker uses a technology called Linux namespaces to provide the isolated workspace
   called the container.
2. When you run a container, Docker creates a set of namespaces for
   that container.
2. Docker Engine on Linux also relies on another technology called control groups (cgroups).
   A cgroup limits an application to a specific set of resources.

## Verify Installation
```
docker run hello-world
```

# Use Docker in Conjunction with Processor Emulation

## Obtain Information about Available Ubuntu Distributions in Docker

First step to to get available CPU Architectures for Ubuntu OS distributed via Docker:
```bash
docker run --rm mplatform/mquery ubuntu:latest
```

In a moment of writing this text, it includes:
* linux/amd64 - Linux on x86-64 (64-bit logical address, Little Endian) CPU ISA. +
* linux/arm/v7 - Linux on ARM v7 (32-bit logical address, Little Endian) CPU ISA. +
* linux/arm64/v8 - Linux on ARM v8 (64-bit logical address, Little Endian) CPU ISA. +
* linux/ppc64le - Linux on PowerPC (64-bit logical address, Little Endian) CPU ISA. +
* linux/riscv64 - Linux on RISC-V (64-bit logical address, Little Endian) CPU ISA. +
* linux/s390x - Linux running on IBM's z/Architecture (64-bit address, Big Endian) CPU ISA. +

## Install Need Emulation

For tests and build tests we can emulate CPU via emulating target ISA due
to big efforts constructing CPU Emulator [QEMU (Quick Emulator)](https://www.qemu.org/).

First of all install this emulator (qemu) and binaries to detect foreign architectures (binfmt-support, qemu-user-static).

* For Linux:
```bash
sudo apt-get install qemu binfmt-support qemu-user-static
sudo docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

* For macOS docker will automatically perform need emulations.

## Run Containers

Create default network:
```bash
docker network create mynet
```

Start the docker container starting with bash and providing a network for the container:
```
docker run -it --network mynet --platform linux/arm64 ubuntu:latest bash
```

Copy the entire project to the container to build it inside it:
```
cd <PATH TO ROOT [FEDNL_IMP]L>
docker cp fednl_impl mycontainer-id:/fednl_impl
```

## Inside Container

```bash
apt-get update && apt-get install git g++ cmake mc python3 iputils-ping net-tools
cd /fednl_impl/dopt/scripts
# Mark copied directory from Host os as safe
git config --global --add safe.directory /fednl_impl
# Build project, run test, run viewer
./project_scripts.py -c -gr -br -tr -j 8 -ub
./project_scripts.py -ir
```

## Restart/Reattach to Container

You can get a list of all container names and containers id via
```bash
docker ps -a
```

You can start, stop, and restart the container with the following command
```bash
docker start|stop|restart <my_ctr_id>
```

To view what is ongoing with output or via terminal you can attach to the container:
```bash
docker attach <my_ctr_id>
```

### 5. Dump the image of the Container and Serialize as an Image

First, you create a new image from a container's changes. Put container `id` instead of d8b4cd12e02b:

```bash
sudo docker commit d8b4cd12e02b                
```

Next, check the ID of recently created containers:

```bash
docker image ls
```

Next, you can create a tag such as "fednl_dev_ubuntu" that refers for example to "11f8c9e35a87" image:

```bash
docker tag 11f8c9e35a87 fednl_dev_ubuntu
```

Finally, you can save the image into the file locally:

```bash
sudo docker save fednl_dev_ubuntu:latest | gzip > fednl_dev_ubuntu.tar.gz
```

After this, the image can be loaded as:
```bash
docker load --input fednl_dev_ubuntu.tar.gz
```

## Stop and Remove all Containers and Remove all Images

```bash
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
```
