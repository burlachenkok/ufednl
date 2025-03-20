#!/usr/bin/env python3

# Available Docs:
#
# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
# Stopping criterias are pretty vague

from datetime import datetime

import numpy as np
import pandas as pd

# Import sklearn relative things
from sklearn.linear_model import LogisticRegression

import time, sys, os
from sklearn.datasets import load_svmlight_file
import json

#=======================================================================================================
# CHANGE IT TO ABSOLUTE PATH WITH DATASET
fname = os.path.join(os.getenv('HOME'), "test/fednl_impl/dopt/datasets/phishing")
#=======================================================================================================

kDebug = False
l2reg = 0.001
rounds=300
nClients = 50
add_intercept = True
#=======================================================================================================

def execute(comment, function, *args, **kw):
  start = time.time()
  res = function(*args, **kw)
  end = time.time()
  print(function, "execution takes", end-start, "seconds", "[", comment, "]")
  return res

def evaluateLogisticObjective(designMatrix, targetLabelSigns, lambda_regul, x):
  A = designMatrix
  b = targetLabelSigns
  m = A.shape[0]
  Ab = np.multiply(designMatrix, targetLabelSigns.reshape((m,1)))

  cost = (x**2).sum() * lambda_regul / 2.0

  for i in range(m):
      cost += (1.0/m) * np.log(1.0 + np.exp(-(np.matmul(Ab[i,:],x)) ) )

  return cost

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



def trainModelViaSkLearn(designMatrix, targetLabelZeroOne, m, d, lambda_regul, usedSolver, verboseModeForSolver):
  model = LogisticRegression(penalty='l2', C=1.0/m * 1.0/lambda_regul, fit_intercept = False, 
                             n_jobs = None, warm_start = True,
                             tol = 0.0,
                             solver = usedSolver, verbose = verboseModeForSolver,
                             max_iter=rounds)
  model.coeff_ = np.zeros(d)

  model.fit(designMatrix, targetLabelZeroOne.flatten())

  xSolution = model.coef_
  xSolution.resize((d,1))

  print("Design matrix shape [samples, dimension]: ", designMatrix.shape)
  print()
  print("Dimension: ", xSolution.size)
  print("Norm of approximate solution x*: ", np.linalg.norm(xSolution))

  return True, xSolution

#=======================================================================================================================================

if __name__ == "__main__":
  print("Timestamp 1: Current Time =", datetime.now().strftime("%H:%M:%S"))

  data = execute("LOAD DATA", load_svmlight_file, fname, zero_based = True)
  print("Timestamp 3: Current Time =", datetime.now().strftime("%H:%M:%S"))

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
      # Encoding of labels in {0,1}
      labels = (labels - 0.5)*2
      pos_samples = sum(labels >= 0.99)
      neg_samples = sum(labels <= -0.99)

  print("  Positive samples: ", pos_samples)
  print("  Negative samples: ", neg_samples)
  print("  Total samples: ", total_samples)

  if neg_samples + pos_samples != total_samples:
    print("Error With Labels in dataset: ", fname)
    sys.exit(-1)

  print("Regulirized Logistics Regression with L2 square regularization term:", l2reg)

  for solver in ["saga",  "sag", "newton-cg"]:
      retValue, xSolution = execute(f"SOLVE WITH SKLEARN/{solver}", trainModelViaSkLearn, designMat, labels, designMat.shape[0], designMat.shape[1], l2reg, solver, False)
      fValue = evaluateLogisticObjective(designMat, labels + labels - 1, l2reg, xSolution)
      print("Optimal value f*: ", fValue)

      g = evaluateLogisticGradient(designMat, labels + labels - 1, l2reg, xSolution)
      print("Norm of gradient in approximate solution x*: ", np.linalg.norm(g))

  ray.shutdown()
  #input()
