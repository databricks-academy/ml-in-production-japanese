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
# MAGIC # Lab:MLflow によるグリッド検索 (Lab: Grid Search with MLflow)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このラボでは、次のことを実施します。<br>
# MAGIC  - scikit-learnを使用してグリッド検索を実行します。
# MAGIC  - MLflowを使って、最適なモデルをログに保存します。
# MAGIC  - 保存したモデルを読み込みます。

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## データインポート (Data Import)
# MAGIC 
# MAGIC 同じAirbnbデータにロードし、学習データとテストデータを作成します。

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## scikit-learnを使用してグリッド検索の実行 (Perform Grid Search using scikit-learn)
# MAGIC 
# MAGIC どのハイパーパラメータ値の組合せが最も効果的であるかを知るために、コードを書いて<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> **`sklearn`** </a>を使用してグリッド検索を実行します。
# MAGIC 
# MAGIC **`n_estimators`** を **`[50, 100]`** および **`max_depth`** を **`[3, 5]`** に設定します。

# COMMAND ----------

# ANSWER
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# dictionary containing hyperparameter names and list of values we want to try
parameters = {"n_estimators": [50, 100], 
              "max_depth": [3, 5]}

rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)

best_rf = grid_rf_model.best_estimator_
for p in parameters:
    print(f"Best '{p}': {best_rf.get_params()[p]}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## MLflowでベストモデルをログに保存 (Log Best Model with MLflow)
# MAGIC 
# MAGIC 新しいMLflow Experimentの下に、run_nameを **`RF-Grid-Search`** 、モデル名を **`grid-random-forest-model`** にし、最適なモデルとそのパラメータ、およびMSEメトリックをログに記録します。

# COMMAND ----------

# ANSWER
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="RF-Grid-Search") as run:
    # Create predictions of X_test using best model
    predictions = best_rf.predict(X_test)

    # Log model
    model_name = "grid-random-forest-model"
    mlflow.sklearn.log_model(best_rf, model_name)

    # Log params
    model_params = best_rf.get_params()
    mlflow.log_params(model_params)

    # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    run_id = run.info.run_id
    print(f"Inside MLflow Run with id {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 保存したモデルの読み込み (Load the Saved Model)
# MAGIC 
# MAGIC 保存したトレーニング済みかつチューニング済みモデルをロードします。このモデルのハイパーパラメータが、先ほど発見したベストモデルのものと一致していることを確認します。

# COMMAND ----------

# ANSWER
model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}/")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 余裕があれば、`MlflowClient`を使用してrunとプログラム的にやり取りしてみましょう。

# COMMAND ----------

# ANSWER

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ
# MAGIC 
# MAGIC 次のレッスンを開始します。[Advanced Experiment Tracking]($../03-Advanced-Experiment-Tracking)。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
