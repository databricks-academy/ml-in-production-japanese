# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC モデルの名前をロードします。 Webhookによって **`event_message`** を自動的に入力されます。

# COMMAND ----------

import json
 
event_message = dbutils.widgets.get("event_message")
event_message_dict = json.loads(event_message)
model_name = event_message_dict.get("model_name")

print(event_message_dict)
print(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC モデルの名前を使ってモデルの最新バージョンを取得します。

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

version = client.get_registered_model(model_name).latest_versions[0].version
version

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC モデルの名前とバージョンを使って本番用の **`pyfunc`** モデルをロードします。

# COMMAND ----------

import mlflow

pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ログされたモデルの入力と出力スキーマを取得します。

# COMMAND ----------

input_schema = pyfunc_model.metadata.get_input_schema().as_spark_schema()
output_schema = pyfunc_model.metadata.get_output_schema().as_spark_schema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ここではあるべき入力と出力のスキーマを定義します。

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, LongType, DoubleType

expected_input_schema = (StructType([
    StructField("host_total_listings_count", DoubleType(), True),
    StructField("neighbourhood_cleansed", LongType(), True),
    StructField("zipcode", LongType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("property_type", LongType(), True),
    StructField("accommodates", DoubleType(), True),
    StructField("bathrooms", DoubleType(), True),
    StructField("bedrooms", DoubleType(), True),
    StructField("beds", DoubleType(), True),
    StructField("bed_type", LongType(), True),
    StructField("minimum_nights", DoubleType(), True),
    StructField("number_of_reviews", DoubleType(), True),
    StructField("review_scores_rating", DoubleType(), True),
    StructField("review_scores_accuracy", DoubleType(), True),
    StructField("review_scores_cleanliness", DoubleType(), True),
    StructField("review_scores_checkin", DoubleType(), True),
    StructField("review_scores_communication", DoubleType(), True),
    StructField("review_scores_location", DoubleType(), True),
    StructField("review_scores_value", DoubleType(), True)
]))

expected_output_schema = StructType([StructField("price", DoubleType(), True)])

# COMMAND ----------

assert expected_input_schema.fields.sort(key=lambda x: x.name) == input_schema.fields.sort(key=lambda x: x.name)
assert expected_output_schema.fields.sort(key=lambda x: x.name) == output_schema.fields.sort(key=lambda x: x.name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC データセットをロードして予測を実施します。モデルを正しく動作していることを確認します。

# COMMAND ----------

import pandas as pd

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
predictions = pyfunc_model.predict(df)

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 予測結果のデータ型が正しいことを確認します。

# COMMAND ----------

import numpy as np

assert type(predictions) == np.ndarray
assert type(predictions[0]) == np.float64

# COMMAND ----------

print("All tests passed!")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
