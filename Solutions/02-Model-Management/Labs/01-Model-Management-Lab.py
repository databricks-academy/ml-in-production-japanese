# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2b9aa913-4058-48fd-9d2a-cf99c3171893"/>
# MAGIC 
# MAGIC 
# MAGIC # Lab: 後処理ロジックの追加 (Lab: Adding Post-Processing Logic)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このラボでは、次のことを行います。<br>
# MAGIC  - データをインポートし、ランダムフォレストモデルをトレーニングします。
# MAGIC  - 後処理ステップを追加します。

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="9afdeb4a-5436-4775-b091-c20451ab9229"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## データのインポート (Import Data)

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="b0f36204-cc7e-4bdd-a856-e8e78ba4673c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ランダムフォレストのトレーニング (Train Random Forest)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Fit and evaluate a random forest model
rf_model = RandomForestRegressor(n_estimators=15, max_depth=5)

rf_model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md <i18n value="17863f12-50a2-42d5-bb2f-47d7e647ab2e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 後処理ステップでPyfuncの作成 (Create Pyfunc with Post-Processing Steps)
# MAGIC 前のデモ・ノートブックでは、 **`preprocess_input(self, model_input)`** ヘルパー関数を使用したカスタム **`RFWithPreprocess`** モデルクラスを構築し、受け取った生の入力データを自動的に前処理してから、トレーニング済みモデルの **`.predict()`** 関数に渡します。
# MAGIC 
# MAGIC ここで予測結果の数値そのものではなく、100ドルを超えると **`高価`** 、超えないと **`高価でない`** の分類をします。まったく新しい分類モデルを再トレーニングする代わりに、上記トレーニング済みモデルに後処理ステップを追加するだけで、数値の価格ではなく予測ラベルが返されます。
# MAGIC 
# MAGIC **新しい`postprocess_result(self, result)`** 関数を使用した次のモデルクラスを完成してください。 **`X_test`** をモデルに渡すと行ごとに **`高価な`** または **`高価ではない`** のラベルを返すようにしてください。

# COMMAND ----------

# ANSWER
import mlflow

# Define the model class
class RFWithPostprocess(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def postprocess_result(self, results):
        """return post-processed results
        Expensive: predicted price > 100
        Not Expensive: predicted price <= 100"""
        
        return ["Expensive" if result > 100 else "Not Expensive" for result in results]
    
    def predict(self, context, model_input):
        results = self.rf.predict(model_input)
        return self.postprocess_result(results)

# COMMAND ----------

# MAGIC %md <i18n value="25109107-4520-4146-9435-6841fd514c16"/>
# MAGIC 
# MAGIC 
# MAGIC モデルを作成して保存し、 **`X_test`** に適用します。

# COMMAND ----------

# Construct model
rf_postprocess_model = RFWithPostprocess(trained_rf=rf_model)

# log model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model("rf_postprocess_model", python_model=rf_postprocess_model)

# Load the model in `python_function` format
model_path = f"runs:/{run.info.run_id}/rf_postprocess_model"
loaded_postprocess_model = mlflow.pyfunc.load_model(model_path)

# Apply the model
loaded_postprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="19dc2c17-fe8c-4229-9d5d-8808c64a30b2"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png">次のステップ</h2>
# MAGIC 
# MAGIC 次のレッスン[Model Registry]($../02-Model-Registry)に進みます。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
