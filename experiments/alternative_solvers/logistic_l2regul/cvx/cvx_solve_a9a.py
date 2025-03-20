#!/usr/bin/env python3
import time
prepare_s = time.time()

import numpy as np
import pandas as pd
import cvxpy as cvx
import time, sys, os
from sklearn.datasets import load_svmlight_file
import pickle

#=======================================================================================================
fname = "./../../../../dopt/datasets/a9a"
kDebug = False
l2reg = 0.001
nClients = 142
add_intercept = True
save_solution = True

#=======================================================================================================

def evaluateLogisticGradient(designMatrix, targetLabelSigns, lambda_regul, x):
  x= x.reshape((-1, 1))
  A = designMatrix
  b = targetLabelSigns
  m = A.shape[0]
  Ab = np.multiply(designMatrix, targetLabelSigns.reshape((m,1)))

  g = x * lambda_regul

  for i in range(m):
      tmp = (-1.0/m) * Ab[i,:]* ( 1.0 / (1.0 + np.exp(np.dot(Ab[i,:],x))) )
      tmp = tmp.reshape((-1,1))
      g += tmp

  return g

def execute(comment, function, *args, **kw):
  start = time.time()
  res = function(*args, **kw)
  end = time.time()
  print(function, "execution takes", end-start, "seconds", "[", comment, "]")
  return res

def trainModelViaCvx(designMatrix, targetLabel, m, d, lambda_regul, usedSolver, verboseModeForSolver, **extra):
  constraints = []
  warm_start = True

  x = cvx.Variable(d)
  if warm_start:
      x.value = np.zeros(d)

  A = designMatrix
  b = targetLabel

  # Multiply arguments element wise with broadcasting
  # Essentially scales each row of "A" by +1/-1
  Ab = np.multiply(designMatrix, targetLabel.reshape((m,1)))

  cost = cvx.sum_squares(x) * (lambda_regul/2.0)
  cost += (1.0/m) * cvx.sum(cvx.logistic(-(cvx.matmul(Ab,x))))
  #for i in range(m):
  #    cost += (1.0/m) * cvx.logistic( -(cvx.matmul(Ab[i,:],x)) )
  prob = cvx.Problem(cvx.Minimize(cost), constraints)

  try:
    prob.solve(solver = usedSolver, warm_start = warm_start, verbose = verboseModeForSolver, **extra)
    print("[OK] Solve time for %s: " % str(usedSolver), prob.solver_stats.solve_time, "seconds (from CVX internals!) <<<< ")
  except cvx.error.SolverError as err:
    print("ERROR DURING SOLVE BY '%s' -- %s" % (usedSolver, str(err)))
    return False, None

  print("status:", prob.status, "('optimal' - the problem was solved. Other statuses: 'unbounded', 'infeasible')")

  if usedSolver!=None:
      print("Used solver: ", usedSolver)
  else:
      print("Used solver: default")
  print("Design matrix shape [samples, dimension]: ", designMatrix.shape)
  xSolution = x.value
  print()
  print("dimension: ", xSolution.size)
  print("optimal value f*: ", prob.value)
  print("norm of approximate solution x*: ", np.linalg.norm(xSolution))

  g = evaluateLogisticGradient(designMatrix, targetLabel, l2reg, xSolution)
  print("Norm of gradient in approximate solution x*: ", np.linalg.norm(g))

  if save_solution:
      fname = usedSolver + "_result.bin"
      with open(fname, "wb") as ofile:
          pickle.dump(xSolution, ofile)

  return True, xSolution

#=======================================================================================================================================

if __name__ == "__main__":
  solvers = cvx.installed_solvers()
  print(f"CVXPY {cvx.__version__} Available Solvers: ", solvers)
  data = execute("LOAD DATA", load_svmlight_file, fname, zero_based = not True)

  designMat, labels = None, None

  if type(data[0]) == np.ndarray:
      designMat = data[0]
  else:
      designMat = data[0].toarray()

  if type(data[1]) == np.ndarray:
      labels = data[1]
  else:
      labels = data[1].toarray()

  strip = (designMat.shape[0]//nClients) * nClients
  designMat = designMat[:strip,]
  labels = labels[:strip,]

  # Add intercept term
  if add_intercept:
    designMat = np.append(designMat, np.ones((designMat.shape[0],1)), axis=1)

  print("  Labels type:", labels.dtype)
  print("  Design matrix type:", designMat.dtype)
  print("  Input file:", os.path.abspath(fname))
  print("  Design matrix shape::", designMat.shape)
  print("  Labels shape::", labels.shape)

  # FOR DEBUG [START]
  if kDebug:
     kSlice = 100
     designMat = designMat[:kSlice,]
     labels = labels[:kSlice,]
  # FOR DEBUG [END]

  pos_samples = sum(labels >= 0.99)
  neg_samples = sum(labels <= -0.99)
  total_samples = len(labels)
  if neg_samples + pos_samples != total_samples:
      labels = (v + v - 1)
      pos_samples = sum(labels >= 0.99)
      neg_samples = sum(labels <= -0.99)

  print("  Positive samples: ", pos_samples)
  print("  Negative samples: ", neg_samples)
  print("  Total samples: ", total_samples)

  if neg_samples + pos_samples != total_samples:
    print("Error With Labels in dataset: ", fname)
    sys.exit(-1)

  prepare_e = time.time()

  print(f"Actual time to prepare input, libraries, etc.: {prepare_e-prepare_s} seconds")

  print("Regulirized Logistics Regression with L2 square regularization term:", l2reg)

  used_solvers = ['CLARABEL', 'ECOS', 'ECOS_BB', 'SCS', 'MOSEK']
  extra_options = [ {}, {}, {}, {}, {}]
  # used_solvers = [used_solvers[3]]

  for i, s in enumerate(used_solvers):
      print("==============================================================")
      execute(f"SOLVE WITH {s}", trainModelViaCvx, designMat, labels, designMat.shape[0], designMat.shape[1], l2reg, s, False, **(extra_options[i]))
      print("==============================================================")

  # print("Press key")
  # input()
