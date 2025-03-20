#!/usr/bin/env python3

from utils import default_dataset_parameters
from utils import read_data, generate_synthetic
from oracles import LogReg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import time

from methods import FedNL, FedNL_CR, FedNL_LS
from methods import Standard_Newton, Newton_Star

#=======================================================================

data_name = 'w8a'
dataset_path = './Datasets/{}.txt'.format(data_name)

# regularization parameter
lmb = 0.001

# number of nodes, size of local data, and dimension of the problem
# according to the paper
N = default_dataset_parameters[data_name]['N']# size of the whole data set
n = default_dataset_parameters[data_name]['n']# number of nodes
m = default_dataset_parameters[data_name]['m']# size of local data set
d = default_dataset_parameters[data_name]['d']# dimension of the problem

print(N,n,m,d)
A, b = read_data(dataset_path=dataset_path, N=N, n=n, m=m, d=d, lmb=lmb, labels=['+1', '-1'])

# set the problem 
logreg = LogReg(A=A, b=b, reg_coef=lmb, n=n, m=m)

# find the solution using Newton's method starting from zeros for 20 iterations
time1 = time.time()
Newton = Standard_Newton(logreg)
Newton.find_optimum(x0=np.zeros(d), n_steps=2,verbose=True)
time2 = time.time()
print('-- Standard_Newton took time {:.3f} ms'.format((time2-time1)*1000.0))

# initial point
np.random.seed(1)

x = np.zeros(d)

time1 = time.time()
# define the method
fednl = FedNL(logreg)
time2 = time.time()
print('-- FedNL init took time {:.3f} ms'.format((time2-time1)*1000.0))

time1 = time.time()
# run the method
fv, bi, it = fednl.method(x=x,hes_comp_name='TopK', hes_comp_param=8*d, 
                          init='nonzero', init_cost=True,
                          option=2, upd_rule=2,
                          lr=None, max_iter=1000, tol=0.0,
                          bits_compute='OneSided', verbose=True)
time2 = time.time()
print('-- FedNL method took time {:.3f} ms'.format((time2-time1)*1000.0))

#func_value - numpy array containing function value in each iteration of the method
#bits - numpy array containing transmitted bits by one node to the server
#iterates - numpy array containing distances from current point to the solution
