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
# MAGIC # MLflow Webhooksとテスト (MLflow Webhooks & Testing)
# MAGIC 
# MAGIC Webhookとは、イベントが発生した際に、HTTP リクエストを介してアクションを呼び出す一般的なメカニズムです。モデル・レジストリのWebhookにより、テストやデプロイ・パイプラインの実行、通知を送信するプッシュメカニズムの提供が可能になり、CI/CDプロセスを円滑に実行できます。モデルレジストリのWebhookは、新たなモデルバージョンの作成、新規コメントの追加、モデルバージョンのステージ移行などによって起動します。このレッスンでは、Webhookを使用してモデル・レジストリ内のモデルに対して自動テストをトリガーする方法について説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC - MLパイプラインにおけるWebhookの役割を探ります。
# MAGIC - モデル・レジストリでモデルをテストするジョブを作成します。
# MAGIC - MLflow Webhookを使用してそのジョブを自動化します。
# MAGIC - Slackに通知を送信するHTTP Webhookを作成します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ## 自動テスト (Automated Testing)
# MAGIC 
# MAGIC 継続的インテグレーション、継続的デプロイ (CI/CD) プロセスのバックボーンは、コードの自動ビルド、テスト、デプロイです。**Webhookまたはトリガー**は、何らかのイベントに基づいてコードを実行します。 一般的には、新しいコードがコードリポジトリにプッシュされるときに使うものです。 機械学習ジョブの場合、モデル・レジストリに新しいモデルが登録されるときなどに使います。
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/applications/mlflow/model-registry-webhooks.html" target="_blank">**MLflowモデル・レジストリのWebhookは以下2つのタイプがあります**</a>:
# MAGIC  - ジョブトリガー付きのWebhook: Databricksワークスペースでジョブをトリガーします。
# MAGIC  - HTTPエンドポイントを持つWebhook: 任意のHTTPエンドポイントにトリガーを送信します。
# MAGIC  
# MAGIC このレッスンでは以下を使用します。
# MAGIC 1. **Job Webhook**でDatabricks Jobの実行をトリガーします。 
# MAGIC 2. **HTTP Webhook**でSlackに通知を送信します。 
# MAGIC 
# MAGIC モデルレジストリに特定の名前を持つモデルの新しいバージョンが登録されると、Databricks Jobは次のタスクを実行します。<br>
# MAGIC - モデルの新しいバージョンをインポートします。
# MAGIC - 入力と出力のスキーマをテストします。
# MAGIC - モデルにサンプルコードを渡します。
# MAGIC 
# MAGIC これは、MLモデルに必要な主要なテストをカバーしています。 ただし、このパラダイムを使用してスループットをテストすることもできます。また、モデルを自動的に「Production」ステージに移行させることもできます。

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ## モデルとJobを作成する (Create a Model and Job)
# MAGIC 
# MAGIC 以下の手順は、このディレクトリにある別のノートブック( **`03B-Webhooks-JOB-demo`** )を使用してDatabricks Jobを作成します。
# MAGIC 
# MAGIC **注意**: 
# MAGIC * あなたがこのワークスペースの管理者であり、Community Edition版 (Job機能が無効) を使用していないことを確認してください。
# MAGIC * 管理者でない場合は、講師にトークンの共有を依頼してください。
# MAGIC * あるいは、  **`token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)`** を設定することもできます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### ユーザー・アクセス　トークンを作成する (Create a user access token)
# MAGIC 
# MAGIC 次の手順を使用して、ユーザー・アクセス　トークンを作成します。<br>
# MAGIC 
# MAGIC 1. [設定]アイコンをクリックします
# MAGIC 2. [ユーザー設定] をクリックします
# MAGIC 3. [アクセス　トークン] タブに移動します。
# MAGIC 4. [新規トークンを生成] ボタンをクリックします
# MAGIC 5. オプションで、コメントと存続期間 (日)を入力します
# MAGIC 6. [生成] ボタンをクリックします
# MAGIC 7. 生成されたトークンを**コピーして、次のセルに貼り付けます。**
# MAGIC 
# MAGIC **注意**:
# MAGIC * あなたがこのワークスペースの管理者であり、Community Edition版 (Job機能が無効) を使用していないことを確認してください。
# MAGIC * 管理者でない場合は、講師にトークンの共有を依頼してください。
# MAGIC * あるいは、 **`token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)`** を設定することもできます。
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/dev-tools/api/latest/authentication.html" target="_blank">アクセストークンの詳細はこちら</a>

# COMMAND ----------

# ANSWER
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

instance = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ### モデルのトレーニングと登録 (Train and Register a Model)
# MAGIC 
# MAGIC モデルを構築してログに記録します。

# COMMAND ----------

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="Webhook RF Experiment") as run:
    # Data prep
    df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    example = X_train.head(3)

    # Train and log model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    mlflow.sklearn.log_model(rf, "random-forest-model", signature=signature, input_example=example)
    mse = mean_squared_error(y_test, rf.predict(X_test))
    mlflow.log_metric("mse", mse)
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC モデルを登録します。

# COMMAND ----------

import uuid

uid = uuid.uuid4().hex[:6]
name = f"{DA.unique_name}_webhook-demo_{uid}"
model_uri = f"runs:/{run_id}/random-forest-model"

model_details = mlflow.register_model(model_uri=model_uri, name=name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Jobの作成 (Creating the Job)
# MAGIC 
# MAGIC 以下の手順は、このディレクトリにある別のノートブック( **`03B-Webhooks-JOB-demo`** )を使用してDatabricks Jobを作成します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC このノートブックと同じフォルダにノートブック **`03B-Webhooks-JOB-Demo`** を実行するJobを作成します。<br><br>
# MAGIC 
# MAGIC - 左側のDatabricks UIのサイドバーにある[ワークフロー]にカーソルを合わせます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/ClickWorkflows.png" alt="step12" width="150"/>
# MAGIC 
# MAGIC - [Jobを作成] をクリックします。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/CreateJob.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - タスク名を入力します。
# MAGIC - ノートブック（ **`03b-Webhooks-Job-Demo`** ）を選択します。 
# MAGIC - 現在のクラスターを選択します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/JobInfo.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - JOB IDをコピーする
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/JobID.png" alt="step12" width="450"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC あるいは、以下のコードはプログラムによってJobを作成します。

# COMMAND ----------

import requests

def find_job_id(instance, headers, job_name, offset_limit=1000):
    params = {"offset": 0}
    uri = f"{instance}/api/2.1/jobs/list"
    done = False
    job_id = None
    while not done:
        done = True
        res = requests.get(uri, params=params, headers=headers)
        assert res.status_code == 200, f"Job list not returned; {res.content}"
        
        jobs = res.json().get("jobs", [])
        if len(jobs) > 0:
            for job in jobs:
                if job.get("settings", {}).get("name", None) == job_name:
                    job_id = job.get("job_id", None)
                    break

            # if job_id not found; update the offset and try again
            if job_id is None:
                params["offset"] += len(jobs)
                if params["offset"] < offset_limit:
                    done = False
    
    return job_id

def get_job_parameters(job_name, cluster_id, notebook_path):
    params = {
            "name": job_name,
            "tasks": [{"task_key": "webhook_task", 
                       "existing_cluster_id": cluster_id,
                       "notebook_task": {
                           "notebook_path": notebook_path
                       }
                      }]
        }
    return params

def get_create_parameters(job_name, cluster_id, notebook_path):
    api = "api/2.1/jobs/create"
    return api, get_job_parameters(job_name, cluster_id, notebook_path)

def get_reset_parameters(job_name, cluster_id, notebook_path, job_id):
    api = "api/2.1/jobs/reset"
    params = {"job_id": job_id, "new_settings": get_job_parameters(job_name, cluster_id, notebook_path)}
    return api, params

def get_webhook_job(instance, headers, job_name, cluster_id, notebook_path):
    job_id = find_job_id(instance, headers, job_name)
    if job_id is None:
        api, params = get_create_parameters(job_name, cluster_id, notebook_path)
    else:
        api, params = get_reset_parameters(job_name, cluster_id, notebook_path, job_id)
    
    uri = f"{instance}/{api}"
    res = requests.post(uri, headers=headers, json=params)
    assert res.status_code == 200, f"Expected an HTTP 200 response, received {res.status_code}; {res.content}"
    job_id = res.json().get("job_id", job_id)
    return job_id

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().replace("03a-Webhooks-and-Testing", "03b-Webhooks-Job-Demo")

# We can use our utility method for creating a unique 
# database name to help us construct a unique job name.
job_name = f"{DA.unique_name}_webhook_job"

# if the Job was created via UI, set it here.
job_id = get_webhook_job(instance, 
                         headers, 
                         job_name,
                         spark.conf.get("spark.databricks.clusterUsageTags.clusterId"),
                         notebook_path
                        )

print(f"Job ID:   {job_id}")
print(f"Job name: {job_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Jobの確認(Examine the Job)
# MAGIC 
# MAGIC [今スケジュールしたノートブック]($./03b-Webhooks-Job-Demo) を見て、何ができたか見てみましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ##Job Webhookの作成 (Create a Job Webhook)
# MAGIC 
# MAGIC Webhookをトリガーできるいくつかのイベントがあります。このノートブックでは、モデルがステージ間で移行するときにJobをトリガーする例にします。

# COMMAND ----------

import json
from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds

endpoint = "/api/2.0/mlflow/registry-webhooks/create"
host_creds = get_databricks_host_creds("databricks")

job_json = {"model_name": name,
            "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
            "description": "Job webhook trigger",
            "status": "Active",
            "job_spec": {"job_id": job_id,
                         "workspace_url": instance,
                         "access_token": token}
           }

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="POST",
    json=job_json
)
assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Webhookを登録したので、**エクスペリメントUIのステージを`None`から `Staging`にモデルを移行することでWebhookをテストします。** [Job] タブにJobが実行されたことを確認できるはずです。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC アクティブなWebhookのリストを取得するには、LISTエンドポイントで GETリクエストを使用します。モデルにWebhookが作成されていない場合、このコマンドにはエラーを返すことに注意してください。

# COMMAND ----------

endpoint = f"/api/2.0/mlflow/registry-webhooks/list/?model_name={name.replace(' ', '%20')}"

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="GET"
)
assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 最後に、Webhook IDをcurlまたは pythonリクエストにコピーし、Webhookを削除します。Webhookが削除されたことを確認するには、LISTリクエストを使用します。

# COMMAND ----------

# ANSWER
delete_hook = response.json().get("webhooks")[0].get("id")
print(delete_hook)

# COMMAND ----------

new_json = {"id": delete_hook}
endpoint = f"/api/2.0/mlflow/registry-webhooks/delete"

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="DELETE",
    json=new_json
)
assert response.status_code == 200, f"Expected HTTP 200, received {response.status_code}"

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ## HTTP Webhookの作成 (Create a HTTP webhook)
# MAGIC 
# MAGIC このセクションでは、Slackワークスペースへのアクセス権とWebhookの作成権限が必要です。このデザインパターンは、TeamsやHTTPリクエストを受け入れる他のエンドポイントでも使えます。
# MAGIC  
# MAGIC <a href="https://api.slack.com/messaging/webhooks" target="_blank">このページに従って</a> Slack Incoming Webhookを設定します。Webhookを以下のコードに貼り付け、コードのコメントを外します。 **`https://hooks.slack.com...`** というようなものになります。モデルレジストリに指定名前のモデルの新しいバージョンが登録されると、Slackチャンネルに通知が送信されます。
# MAGIC 
# MAGIC <a href="https://github.com/mlflow/mlflow/blob/master/mlflow/utils/rest_utils.py" target="_blank">**`mlflow`** RESTユーティリティ関数の詳細については、こちらを参照してください。</a>

# COMMAND ----------

# from mlflow.utils.rest_utils import http_request
# from mlflow.utils.databricks_utils import get_databricks_host_creds
# import urllib

# slack_incoming_webhook = "<insert your token here>" 

# endpoint = "{instance}/api/2.0/mlflow/registry-webhooks/create"
# host_creds = get_databricks_host_creds("databricks")

# ## specify http url of the slack notification
# http_json = {"model_name": name,
#   "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
#   "description": "Job webhook trigger",
#   "status": "Active",
#   "http_url_spec": {
#     "url": slack_incoming_webhook,
#     "enable_ssl_verification": "false"}}

# response = http_request(
#   host_creds=host_creds, 
#   endpoint=endpoint,
#   method="POST",
#   json=http_json
# )

# print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Webhook を登録したので、**エクスペリメントUI上のステージ `None` から `Staging` にモデルを移行することでWebhookをテストできます。** 関連付けられたSlackチャネルに受信メッセージが表示されるはずです。
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/ml-deployment/webhook_slack.png" alt="webhook_通知" width="400"/>
# MAGIC <br>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ## リソース (Resources)
# MAGIC 
# MAGIC - <a href="https://databricks.com/blog/2020/11/19/mlflow-model-registry-on-databricks-simplifies-mlops-with-ci-cd-features.html" target="_blank">CI/CD とWebhookの詳細については、このブログを参照してください。</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
