# Databricks notebook source
import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# load model
model_path = "/dbfs/FileStore/models"
model = mlflow.pyfunc.load_model(model_path)
 
# define search space
search_space = {
  'max_depth': hp.quniform('max_depth', 2, 10, 1),
  'n_estimators': hp.quniform('n_estimators', 200, 1000, 100),
  'max_features': hp.quniform('max_features', 3, 8, 1),
}
 
    
def train_model(n_estimators):
   
  print("n_estimators:", n_estimators)
  # Create and train model.
#   rf = RandomForestRegressor()
#   rf.fit(X_train, y_train)
  
#   predictions = rf.predict(X_test)
  
#   # Evaluate the model
#   mse = mean_squared_error(y_test, predictions)
  
  return {"loss": 1, "status": STATUS_OK}
  
  
spark_trials = SparkTrials()
 
with mlflow.start_run() as run:
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=1,
    trials=spark_trials)

print("successfully loaded")

print(best_params)
