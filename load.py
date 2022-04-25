# Databricks notebook source
import logging
from pyspark.sql import *

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# dbfs:/FileStore/tables/dataset_v1.csv  Path in Spark API format
# /dbfs/FileStore/tables/dataset_v1.csv  Path in file API format

try:
    sparkDF = spark.read.csv("dbfs:/FileStore/tables/dataset_v1.csv", header="true", inferSchema="true")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
    
tables_collection = spark.catalog.listTables("default")

table_names_in_db = [table.name for table in tables_collection]

table_exists = "perf_data" in table_names_in_db

if not table_exists:
    sparkDF.write.saveAsTable("perf_data")
else:
    sparkDF.write.insertInto("perf_data")

