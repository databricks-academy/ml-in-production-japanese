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
# MAGIC 
# MAGIC # ラボ: バッチ処理によるモデルのデプロイ（Deploying a Model in Batch）
# MAGIC 
# MAGIC バッチによるモデルのデプロイは、ほとんどの機械学習アプリケーションで推奨されるソリューションです。このラボでは、Spark UDFとMLflowの **`pyfunc`** を使用してシングルノードモデルのデプロイメントをスケールアップすることができます。
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このラボでは以下のことを実施します。<br>
# MAGIC  - MLflowモデルの開発と登録
# MAGIC  - Spark UDFとしてのモデルのデプロイ
# MAGIC  - Deltaでの読み込みと推論最適化

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

df = (spark
      .read
      .parquet(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
      .select("price", "bathrooms", "bedrooms", "number_of_reviews")
     )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### MLflowモデルの開発と登録（Develop and Register an MLflow Model）
# MAGIC 
# MAGIC この演習では、Scikit-learn と MLflow を使用して XGBoost モデルを構築し、記録し、登録します。
# MAGIC 
# MAGIC このモデルは、 **`bathrooms`** , **`bedrooms`** , **`number_of_reviews`** を特徴量として使用して、 **`price`** 変数を予測します。

# COMMAND ----------

import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error
import uuid

# Start run
with mlflow.start_run(run_name="xgboost_model") as run:
    train_pdf = df.toPandas()
    X_train = train_pdf.drop(["price"], axis=1)
    y_train = train_pdf["price"]

    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X_train, y_train)

    # Evaluate model
    predictions = regressor.predict(X_train)
    rmse = mean_squared_error(predictions, y_train, squared=False)

    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.xgboost.log_model(regressor, "xgboost-model")
    
# Register model
uid = uuid.uuid4().hex[:6]
model_name = f"{DA.unique_name}_xgboost-model_{uid}"
model_uri = f"runs:/{run.info.run_id}/xgboost-model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Spark UDFとしてのモデルのデプロイ（Deploy Model as a Spark UDF）
# MAGIC 
# MAGIC 次に、Spark UDFを使って予測処理をします。

# COMMAND ----------

# ANSWER
# Create the prediction UDF
predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/xgboost-model")

# Compute the predictions
prediction_df = df.withColumn("prediction", predict(*df.drop("price").columns))
                 
# View the results
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のレッスン <br>
# MAGIC 
# MAGIC 次のレッスンに進みましょう。 [Real Time]($../02-Real-Time)　

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
