import time,os

s = time.time()
import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import array_min
from sklearn.datasets import load_svmlight_file
import pickle, os

e = time.time()
print("LOAD NEED LIBRARIES", e-s, "seconds")

def evaluateLogisticObjective(designMatrix, targetLabelSignsZeroOne, lambda_regul, x):
  x = x.reshape((-1, 1))
  d = len(x)
  m = designMatrix.shape[0]
  Ab = np.zeros((m, d))

  for i in range(m):
      if targetLabelSignsZeroOne.loc[i] > 0.5:
          Ab[i,:] = designMatrix.loc[i].toArray()
      else:
          Ab[i, :] = -designMatrix.loc[i].toArray()

  cost = (x**2).sum() * lambda_regul / 2.0

  for i in range(m):
      cost += (1.0/m) * np.log(1.0 + np.exp(-(np.dot(Ab[i,:],x)) ) )

  return cost

def evaluateLogisticGradient(designMatrix, targetLabelSignsZeroOne, lambda_regul, x):
  x = x.reshape((-1, 1))
  d = len(x)
  m = designMatrix.shape[0]
  Ab = np.zeros((m, d))

  for i in range(m):
      if targetLabelSignsZeroOne.loc[i] > 0.5:
          Ab[i,:] = designMatrix.loc[i].toArray()
      else:
          Ab[i, :] = -designMatrix.loc[i].toArray()

  g = x * lambda_regul

  for i in range(m):
      tmp = (-1.0/m) * Ab[i,:]* ( 1.0 / (1.0 + np.exp(np.dot(Ab[i,:],x))) )
      tmp = tmp.reshape((-1,1))
      g += tmp

  return g


#import findspark
#findspark.init()
#import scipy as scipy
#import math
#from pyspark import SparkConf, SparkContext

fname = "./../../../../dopt/datasets/" + os.getenv("dataset")
kDebug = False
l2reg = 0.001

start = time.time()

s = time.time()

# Connect to default port: 7077
# Web UI is available at: 8080

master_port  = 7077
master_host  = os.getenv('master_host')
spark = SparkSession.builder.appName("test").master(f"spark://{master_host}:{master_port}").getOrCreate()

e = time.time()
print("session init ", e-s, "seconds")

s = time.time()
df = spark.read.format("libsvm").load(fname)
e = time.time()
print("load data ", e-s, "seconds")

if df.agg({'label':'min'}).toPandas().loc[0].values[0] <= -0.9999:
    df_zero_one_labels = df.withColumn("label", (df.label + 1.0)/2.0)
else:
    df_zero_one_labels = df

#======================================================================================================================
df_zero_one_labels_pd = df_zero_one_labels.toPandas()
m = df_zero_one_labels_pd.shape[0]
#======================================================================================================================
s = time.time()
reg = LogisticRegression(featuresCol='features',
                         labelCol='label',
                         predictionCol='prediction', 
                         maxIter=300, 
                         tol=1e-12,
                         standardization=False,
                         regParam=l2reg,
                         fitIntercept = False)

model = reg.fit(df_zero_one_labels)
xSolution = model.coefficients.toArray()
e = time.time()

print("solve time", e-s, "seconds")

#======================================================================================================================

fxk = evaluateLogisticObjective(designMatrix = df_zero_one_labels_pd["features"],
                                targetLabelSignsZeroOne = df_zero_one_labels_pd["label"],
                                lambda_regul = l2reg,
                                x = xSolution)

print("Optimal value f*: ", fxk)
print("Design matrix shape [samples, dimension]: ", m, len(xSolution))
print()
print("Dimension: ", len(xSolution))
print("Norm of approximate solution x*: ", np.linalg.norm(xSolution))
g = evaluateLogisticGradient(designMatrix = df_zero_one_labels_pd["features"],
                             targetLabelSignsZeroOne = df_zero_one_labels_pd["label"],
                             lambda_regul = l2reg,
                             x = xSolution)

print("Norm of gradient in approximate solution x*: ", np.linalg.norm(g))
#======================================================================================================================
