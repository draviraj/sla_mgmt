# Databricks notebook source
from pyspark.sql import *
from sklearn.model_selection import train_test_split
import logging
import os
import sys
import time
import warnings
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    # return self.model.predict_proba(model_input)[:,1]
    return self.model.predict(model_input)[:,1]

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Read the performance data
spartDf = spark.sql("select * from perf_data")

# Convert the Spark dataframe to pandas
pandaDf = spartDf.toPandas()

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(pandaDf)

# Drop the Timestamp and other dependent variables aggr_cpu, aggr_mem, aggr_disk, aggr_link_bw
train_x = train.drop(["Timestamp", "aggr_cpu", "aggr_mem", "aggr_disk", "aggr_link_bw"], axis=1)
test_x = test.drop(["Timestamp", "aggr_cpu", "aggr_mem", "aggr_disk", "aggr_link_bw"], axis=1)

# Only select the dependent variables
train_y = train[["aggr_cpu", "aggr_mem", "aggr_disk", "aggr_link_bw"]]
test_y = test[["aggr_cpu", "aggr_mem", "aggr_disk", "aggr_link_bw"]]
    
mlflow.set_experiment(experiment_name="/perf_data_training")

with mlflow.start_run(run_name='my_first_run'):

    # Set the model parameters. 
    n_estimators = 100
    max_depth = 6
    max_features = 3

    lr = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)        

    lr.fit(train_x, train_y)

    # modelpath = "/dbfs/FileStore/models"
    # dbutils.fs.rm(modelpath)    

    # mlflow.sklearn.save_model(lr, modelpath)

    predicted_qualities = lr.predict(test_x)

    print("predicted_qualities", predicted_qualities[0])

    print("lr.n_outputs_", lr.n_outputs_)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # mlflow.log_param("alpha", alpha)
    # mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    print("mlflow.get_tracking_uri()", urlparse(mlflow.get_tracking_uri()))
    signature = infer_signature(train_x, lr.predict(train_x))
    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="RandomForestRegressorModel", signature=signature)
    else:
        mlflow.sklearn.log_model(lr, "model", signature=signature)
