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
# MAGIC # ラボ: MLflow Model Servingでのリアルタイム・モデルのデプロイ（Deploying a Real-time Model with MLflow Model Serving）
# MAGIC MLflow Model Serving は、事前に計算された予測を提供する、あるいはリアルタイムで予測値を作成する高速な方法を提供します。このラボでは、MLflow Model Serving を使ってモデルをデプロイします。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このラボでは、以下を実施します。<br>
# MAGIC  - 登録したモデルに対して、MLflow Model Serving を有効化します。
# MAGIC  - 登録したモデルに対して、REST APIリクエストでリアルタイムに予測値を計算します。
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> *モデル提供エンドポイントを作成するには、<a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> 権限が必要です。講師はこのノートブックのデモを行うか、管理コンソールから生徒のためにクラスタ作成権限を有効にします*。

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC まず始めに、データをロードし、モデルを構築し、そのモデルを登録しましょう。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 今回はAirbnbのリスティング価格を予測するためにランダムフォレストモデルを構築しています。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import uuid

# Load data
df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = df[["bathrooms", "bedrooms", "number_of_reviews"]]
y = df["price"]

# Start run
with mlflow.start_run(run_name="Random Forest Model") as run:
    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X, y)
    
    # Evaluate model
    y_pred = regressor.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    
    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)
    
    # Log model
    mlflow.sklearn.log_model(regressor, "model")
    
# Register model
uid = uuid.uuid4().hex[:6]
model_name = f"{DA.unique_name}_rfr-model_{uid}"
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
model_version = model_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 次に、モデルをステージングへ移行します。

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 登録したモデルに対して、MLflow Model Serving を有効化（Enable MLflow Model Serving for the Registered Model）
# MAGIC 
# MAGIC 最初のタスクは、登録されたモデルに対して MLflow Model Serving を有効にすることです。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;　**ヒント:** 　UI から　Model servingを有効にする方法について、<a href="https://docs.databricks.com/applications/mlflow/model-serving.html#enable-and-disable-model-serving" target="_blank">こちら</a> を確認してください。
# MAGIC 
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/mlflow/demo_model_register.png" width="600" height="20"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## リアルタイム予測の計算（Compute Real-time Predictions）
# MAGIC 
# MAGIC モデルが登録されたので、入力値とともにモデルに問い合わせをします。
# MAGIC 
# MAGIC これを行うには、まず適切なトークンとapi_urlが必要です。

# COMMAND ----------

# We need both a token for the API, which we can get from the notebook.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's context
api_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
print(api_url)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC エンドポイントを有効にします。

# COMMAND ----------

import requests

url = f"{api_url}/api/2.0/mlflow/endpoints/enable"

r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 先に進む前にリソースの準備が整っていることを確認するために、2 つの待機メソッドを再定義できます。

# COMMAND ----------

def wait_for_endpoint():
    import time
    while True:
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/get-status?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ENDPOINT_STATE_READY": print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds

# COMMAND ----------

def wait_for_version():
    import time
    while True:    
        url = f"{api_url}/api/2.0/preview/mlflow/endpoints/list-versions?registered_model_name={model_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        state = response.json().get("endpoint_versions")[0].get("state")
        if state == "VERSION_STATE_READY": print("-"*80); return
        else: print(f"Version not ready ({state}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 次に、1つのレコードを入力として受け取り、エンドポイントから予測値を返す関数を作成します。

# COMMAND ----------

# ANSWER
import requests

def score_model(dataset: pd.DataFrame, model_name: str, token: str, api_url: str):
    url = f"{api_url}/model/{model_name}/1/invocations"
    data_json = dataset.to_dict(orient="split")
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC さて、Pandas DataFrameの1行をスコアリングする関数を使ってみましょう。
# MAGIC 
# MAGIC Now, use that function to score a single row of a Pandas DataFrame.

# COMMAND ----------

wait_for_endpoint()
wait_for_version()

# COMMAND ----------

# ANSWER

single_row_df = pd.DataFrame([[2, 2, 150]], columns=["bathrooms", "bedrooms", "number_of_reviews"])
score_model(single_row_df, model_name, token, api_url)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
