#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cvxpy as cvx
import time, sys
import sklearn
import matplotlib
import os, sys, platform, socket
import ray
import Cython

if __name__ == "__main__":
  print("Envrionment Information")
  print("=========================================")
  print("NumPy version:", np.__version__)
  print("CVXPY version:", cvx.__version__)
  print("Pandas verion:", pd.__version__)
  print("SKlearn verion:", sklearn.__version__)
  print("MatplotLib version:", matplotlib.__version__)
  print("Ray version:", np.__version__)
  print("Cython version:", Cython.__version__)
  print("")
  print("Current working directory:", {os.getcwd()})
  print("Executed script:", {__file__})
  print("Python binary:", sys.executable)
  print("Python version:", sys.version)
  print("Platform name:", sys.platform)
  print("")
  (system, node, release, version, machine, processor) = platform.uname()
  print(f"System/OS name: {system}/{release}/{version}")
  print(f"Machine name: {machine}")
  print(f"Host name: {socket.gethostname()}")
  print(f"IP address of the host: {socket.gethostbyname(socket.gethostname())}")
  print("")
  print("CVXPY version:", cvx.__version__)
  print("=========================================")
  solvers = cvx.installed_solvers()
  print("CVXPY AVAILABLE SOLVERS:")
  for s in solvers:
    print(" ", s)
