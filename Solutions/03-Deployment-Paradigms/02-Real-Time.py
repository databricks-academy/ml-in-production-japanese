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
# MAGIC # リアルタイム・デプロイ（Real Time Deployment）
# MAGIC 
# MAGIC リアルタイムでデプロイする必要のあるユースーケースは、デプロイ全体に占める割合は小さいですが、これらの多くは価値の高いタスクです。 このレッスンでは、POCからカスタムソリューションやマネージドソリューションまで、リアルタイム・デプロイについて調査します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します:<br>
# MAGIC  - リアルタイム・デプロイのオプションを調査する
# MAGIC  - MLflowを使用したRESTfulサービスのプロトタイプ作成
# MAGIC  - MLflow Model Serving を使って登録されたモデルをデプロイする
# MAGIC  - MLflow Model Serving のエンドポイントへのリクエスト、個々のレコードとバッチリクエストを使用して推論を行う
# MAGIC  
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> *モデル提供エンドポイントを作成するには、<a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> 権限が必要です。講師はこのノートブックのデモを行うか、管理コンソールから生徒のためにクラスタ作成権限を有効にします。* 

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### リアルタイム・デプロイの理由と方法（The Why and How of Real Time Deployment）
# MAGIC 
# MAGIC リアルタイム推論とは...<br><br>
# MAGIC 
# MAGIC * 少数のレコードに対する予測を高速に生成すること（例：ミリ秒単位での結果）
# MAGIC * リアルタイムの導入を検討する際に、まず問われるのは、それが必要なのかどうかということです。 
# MAGIC   - 機械学習推論のユースケースの中では少数派 &mdash; 特徴量（feature）を提供時にしか利用できない場合に必要となります。
# MAGIC   - より複雑なモデルのデプロイ方法の1つでです。
# MAGIC   - とはいえ、リアルタイム・デプロイが必要とされることが多い領域は、ビジネス的に大きな価値があることが多いのも事実です。 
# MAGIC   
# MAGIC リアルタイム・デプロイが必要な領域は...<br><br>
# MAGIC 
# MAGIC  - 金融サービス（特に詐欺の検出（Fraud Detection））
# MAGIC  - モバイル
# MAGIC  - アド・テクノロジー

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルをデプロイする方法はいくつもありますが...<br><br>
# MAGIC 
# MAGIC * 多くはRESTを使用
# MAGIC * 基本的なプロトタイプの場合、MLflowは開発時のデプロイ用のサーバーとし利用することができます。
# MAGIC   - MLflowの実装はPythonのライブラリであるFlaskによって支えられています。
# MAGIC   - *本番環境での利用を想定したものではありません*　。
# MAGIC 
# MAGIC さらに、Databricksはマネージドなサービスとして　**MLflow Model Serving** を提供しています。このソリューションでは、モデルのバージョンとそのステージの可用性に基づいて自動的に更新されるRESTエンドポイントとして、Model Registryから機械学習モデルをホストすることができます。
# MAGIC 
# MAGIC 
# MAGIC 本番環境におけるRESTfulなデプロイに関しては、主に2つのオプションがあります...<br><br>
# MAGIC 
# MAGIC * マネージド・ソリューション 
# MAGIC   - Azure ML
# MAGIC   - SageMaker (AWS)
# MAGIC   - Vertex AI (GCP)
# MAGIC * カスタム・ソリューション  
# MAGIC   - 様々なツールを使ったデプロイメントを含む
# MAGIC   - DockerやKubernetesの利用
# MAGIC * デプロイの重要な要素の1つであるコンテナ化について
# MAGIC   - ソフトウエアは、独自のアプリケーション、ツール、ライブラリとして分離さて、でパッケージ化されます。
# MAGIC   - コンテナは、仮想マシンに代わるより軽量な実装方法の選択肢です。
# MAGIC 
# MAGIC 最後に、組み込みソリューションは、推論のためにIoTデバイスにモデルを格納するなど、機械学習モデルをデプロイするもう1つの方法です。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC ## MLflowを使ったプロトタイピング（Prototyping with MLflow）
# MAGIC 
# MAGIC MLflow は <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank"> 開発目的に限って、Flask-backed deployment serverを提供しています。</a>
# MAGIC 
# MAGIC それでは、簡単なモデルを作ってみましょう。　このモデルは常に５を予測します。

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pandas as pd

class TestModel(mlflow.pyfunc.PythonModel):
  
    def predict(self, context, input_df):
        return 5

model_run_name="pyfunc-model"

with mlflow.start_run() as run:
    model = TestModel()
    mlflow.pyfunc.log_model(artifact_path=model_run_name, python_model=model)
    model_uri = f"runs:/{run.info.run_id}/{model_run_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC テスト用に開発用サーバーにリクエストを送るには、いくつかの方法があります。
# MAGIC * **`click`** ライブラリを使用します。
# MAGIC * MLflow Model Serving API を使用します。
# MAGIC * CLI を通して **`mlflow models serve`** を使用します。
# MAGIC 
# MAGIC このレッスンでは、  **`click`** ライブラリとMLflow Model Serving APIの両方を使用する方法を説明します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> ここで、基本的な開発用サーバーがどのように動作するかを示すだけです。　このデザインパターン(Sparkクラスタのドライバと同じサーバ上にホストする)は、本番環境向けには推奨されません。
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> モデルは、他の言語でもこの方法で提供することができます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Method 1: `click` ライブラリを使用する

# COMMAND ----------

import time
from multiprocessing import Process

server_port_number = 6501
host_name = "127.0.0.1"

def run_server():
    try:
        import mlflow.models.cli
        from click.testing import CliRunner

        CliRunner().invoke(mlflow.models.cli.commands, 
                         ["serve", 
                          "--model-uri", model_uri, 
                          "-p", server_port_number, 
                          "-w", 4,
                          "--host", host_name, # "127.0.0.1", 
                          "--no-conda"])
    except Exception as e:
        print(e)

p = Process(target=run_server) # Create a background process
p.start()                      # Start the process
time.sleep(5)                  # Give it 5 seconds to startup
print(p)                       # Print it's status, make sure it's runnning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC REST入力のためのinputを作成します。

# COMMAND ----------

import pandas as pd

input_df = pd.DataFrame([0])
input_json = input_df.to_json(orient="split")

input_json

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC エンドポイントに対してPOSTリクエストを実行します。

# COMMAND ----------

import requests
from requests.exceptions import ConnectionError
from time import sleep

headers = {"Content-type": "application/json"}
url = f"http://{host_name}:{server_port_number}/invocations"

try:
    response = requests.post(url=url, headers=headers, data=input_json)
except ConnectionError:
    print("Connection fails on a Run All.  Sleeping and will try again momentarily...")
    sleep(5)
    response = requests.post(url=url, headers=headers, data=input_json)

print(f"Status: {response.status_code}")
print(f"Value:  {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC bash上で同じことをします。

# COMMAND ----------

# MAGIC %sh (echo -n '{"columns":[0],"index":[0],"data":[[0]]}') | curl -H "Content-Type: application/json" -d @- http://127.0.0.1:6501/invocations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC バックグランドプロセスをクリーンアップします。

# COMMAND ----------

p.terminate()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Method 2: MLflow Model Serving
# MAGIC それでは、MLflow Model Servingを使ってみましょう。 
# MAGIC 
# MAGIC Step 1: まず、MLflow Model Registryにモデルを登録し、モデルをロードします。このステップでは、モデルのステージを指定しないので、ステージのバージョンは **`None`** になります。
# MAGIC 
# MAGIC MLflow のドキュメントは、 <a href="https://www.mlflow.org/docs/latest/model-registry.html#api-workflow" target="_blank">こちら</a>をご覧ください.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC モデルをトレーニングします。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import uuid

uid = uuid.uuid4().hex[:6]
model_name = f"{DA.unique_name}_demo-model_{uid}"

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf, 
                             "model", 
                             input_example=input_example, 
                             signature=signature, 
                             registered_model_name=model_name
                            )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Step 2: Stageing に昇格するために、登録されたモデルのテストを実行します。

# COMMAND ----------

time.sleep(10) # to wait for registration to complete

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC ここで、モデルをサービングを有効にするために、MLflow Model Registry　にアクセスします。
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/mlflow/demo_model_register.png" width="600" height="20"/>

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
# MAGIC エンドポイントを有効にします。

# COMMAND ----------

import requests

url = f"{api_url}/api/2.0/mlflow/endpoints/enable"

r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC エンドポイントとモデルの準備が整うまでに数分かかります。
# MAGIC 
# MAGIC 
# MAGIC **wait_for_endpoint()** と **wait_for_model()** 関数を定義します。

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
# MAGIC 
# MAGIC **`score_model()`** 関数を定義します。

# COMMAND ----------

def score_model(dataset: pd.DataFrame, timeout_sec=300):
    import time
    start = int(time.time())
    print(f"Scoring {model_name}")
    
    url = f"{api_url}/model/{model_name}/1/invocations"
    data_json = dataset.to_dict(orient="split")
    
    while True:
        response = requests.request(method="POST", headers=headers, url=url, json=data_json)
        elapsed = int(time.time()) - start
        
        if response.status_code == 200: return response.json()
        elif elapsed > timeout_sec: raise Exception(f"Endpoint was not ready after {timeout_sec} seconds")
        elif response.status_code == 503: 
            print("Temporarily unavailable, retr in 5")
            time.sleep(5)
        else: raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC model serving clusterが **`ready`** 状態になったら、REST エンドポイントにリクエストを送ることができるようになります。

# COMMAND ----------

wait_for_endpoint()
wait_for_version()

# Give the system just a couple
# extra seconds to transition
time.sleep(5)

# COMMAND ----------

score_model(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC また、オプションとして、<a href="https://www.mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage" target="_blank">MLflow Model Registry</a> を使用して、モデルを **`Staging`** または **`Production`** ステージに遷移させることも可能です。
# MAGIC 
# MAGIC サンプルコードは以下の通りです。
# MAGIC 
# MAGIC ```
# MAGIC client.transition_model_version_stage(
# MAGIC     name=model_name,
# MAGIC     version=model_version,
# MAGIC     stage="Staging"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **予期せぬコストが発生しないよう、Model Serving Clusterをシャットダウンすることを忘れないでください**。自動的に終了することはありません **`Status`** の横にある **`Stop`** をクリックして、Model Serving Clusterを停止してください。
# MAGIC <Br>
# MAGIC 
# MAGIC <div><img src="http://files.training.databricks.com/images/mlflow/demo_model_hex.png" style="height: 250px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **構築したインフラは、コース終了後に必ず削除して、予期せぬ出費が発生しないようにしてください。**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## AWS SageMaker
# MAGIC 
# MAGIC - <a href="https://docs.aws.amazon.com/sagemaker/index.html" target="_blank">mlflow.sagemaker</a> は　一つの関数を使って学習済みモデルをSageMakerにデプロイすることができます。<a href="https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow.sagemaker.deploy" target="_blank">  **`mlflow.sagemaker.deploy`** </a> です。
# MAGIC - デプロイの際、MLflow はモデルのロードとサービングに必要なリソースを持つ専用の Docker コンテナを使用します。このコンテナの名前は **`mlflow-pyfunc`** です。
# MAGIC - デフォルトでは、MLflowはAWS Elastic Container Registry (ECR)内でこのコンテナを検索します。このコンテナをビルドしてECRにアップロードするには
# MAGIC MLflowの<a href="https://www.mlflow.org/docs/latest/cli.html#mlflow-sagemaker-build-and-push-container" target="_blank">CLI</a>にある **`mlflow sagemaker build-and-push-container`** 関数を使用して、このコンテナを構築し、ECRにアップロードできます。 また、以下のように環境変数を設定することで、このコンテナに別のURLを指定することもできます。
# MAGIC 
# MAGIC ```
# MAGIC   # ECR　の　URLは以下のようになりますe:
# MAGIC   {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
# MAGIC   
# MAGIC   image_ecr_url = "<ecr-url>"
# MAGIC   # URLを元に環境変数を設定する
# MAGIC   os.environ["SAGEMAKER_DEPLOY_IMG_URL"] = image_ecr_url
# MAGIC   
# MAGIC   # モデルをデプロイする
# MAGIC   mlflow.sagemaker.deploy(
# MAGIC      app_name, # application/model name
# MAGIC      model_uri, # model URI in Model Registry
# MAGIC      image_url=image_ecr_url, region_name=region, mode="create")
# MAGIC   )
# MAGIC ```
# MAGIC - Databricksノートブックでデプロイと推論を実行するには、これらの操作を実行する権限を持つAWS IAMロールがDatabricksクラスタに設定されていることが必要です。
# MAGIC - エンドポイントが稼働すると、 **`boto3`** の **`sagemaker-runtime`** API が REST API に対してクエリを発行できるようになります。
# MAGIC ```python
# MAGIC client = boto3.session.Session().client("sagemaker-runtime", "{region}")
# MAGIC   
# MAGIC   response = client.invoke_endpoint(
# MAGIC       EndpointName=app_name,
# MAGIC       Body=inputs,
# MAGIC       ContentType='application/json; format=pandas-split'
# MAGIC   )
# MAGIC   preds = response['Body'].read().decode("ascii")
# MAGIC   preds = json.loads(preds)
# MAGIC   print(f"Received response: {preds}")
# MAGIC   ```
# MAGIC 
# MAGIC **Tip**: Sagemaker の各エンドポイントは、1つのリージョンにスコープされています。リージョンをまたいだデプロイが必要な場合、Sagemaker のエンドポイントは各リージョンに存在する必要があります。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Azure
# MAGIC 
# MAGIC - AzureML と MLflow は、どちらかを使うことで <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models" target="_blank">REST endpoints</a> としてモデルをデプロイすることができます。  
# MAGIC   - **Azure Container Instances**: ACI を通してデプロイする場合、自動的にモデルを登録し、コンテナを作成・登録し（まだ存在しない場合）、イメージを構築し、エンドポイントを設定します。その後、AzureML studio UI からエンドポイントを監視することができます。 **ACIよりも Azure Kubernetes Serviceの方が一般的に本番環境では推奨されることに注意してください。** <img src="http://files.training.databricks.com/images/mlflow/rest_serving.png" style="height: 700px; margin: 10px"/>  
# MAGIC   - **Azure Kubernetes Service**: AKS経由でデプロイする場合、K8sクラスタをコンピュートターゲットとして設定するには、 `deployment_configuration()` <a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aks.akswebservice?view=azure-ml-py#deploy-configuration-autoscale-enabled-none--autoscale-min-replicas-none--autoscale-max-replicas-none--autoscale-refresh-seconds-none--autoscale-target-utilization-none--collect-model-data-none--auth-enabled-none--cpu-cores-none--memory-gb-none--enable-app-insights-none--scoring-timeout-ms-none--replica-max-concurrent-requests-none--max-request-wait-time-none--num-replicas-none--primary-key-none--secondary-key-none--tags-none--properties-none--description-none--gpu-cores-none--period-seconds-none--initial-delay-seconds-none--timeout-seconds-none--success-threshold-none--failure-threshold-none--namespace-none--token-auth-enabled-none--compute-target-name-none--cpu-cores-limit-none--memory-gb-limit-none-" target="_blank">function</a> createを使用する必要があります。Azure Kubernetes Servicesは、ロードバランシングやフォールオーバーなどの機能を備えているため、より堅牢なプロダクションサービングオプションと言えます。  
# MAGIC   - Azure Machine Learningエンドポイント（現在プレビュー中）
# MAGIC   
# MAGIC - Azure上にモデルをデプロイする場合、Databricksワークスペースから<a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow" target="_blank">MLflow Tracking URI</a> をAzureMLワークスペースに接続する必要があることに注意してください。一度接続が確立されると、両者間で実験の追跡が可能になります。
# MAGIC 
# MAGIC 
# MAGIC **Tip: `azureml-mlflow` は ML runtimeとして含まれて *いない* ので、クラスターにインストールする必要があります。**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## GCP
# MAGIC 
# MAGIC GCPユーザーは、GCP Databricksワークスペース上でモデルを学習し、学習したモデルをMLFlow Model Registryに記録し、<a href="https://cloud.google.com/vertex-ai" target="_blank">Vertex AI</a>に本番用のモデルをデプロイして、Model-serving endpointを作成することが可能です。GCPサービスアカウントの設定と、Google Cloud用のMLflowプラグインのインストール（`%pip install google_cloud_mlflow`）が必要です。
# MAGIC 
# MAGIC ####**GCPサービスアカウントの設定（To set up GCP service account）**:
# MAGIC 
# MAGIC - GCPプロジェクトの作成（<a href="https://cloud.google.com/apis/docs/getting-started" target="_blank">こちら</a>を参照してください）。　Databricksワークスペースが所属するプロジェクトを使用することができます。
# MAGIC - GCPプロジェクトのVertex AIとCloud Build APIの有効化します。
# MAGIC - 以下の最小限のIAM権限を持つサービスアカウントを作成します（GCSからMlflowモデルをロードし、コンテナを構築し、コンテナをVertex AIエンドポイントにデプロイする手順は、<a href="https://cloud.google.com/iam/docs/creating-managing-service-accounts" target="_blank">こちら</a>を参照してください。
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC cloudbuild.builds.create
# MAGIC cloudbuild.builds.get
# MAGIC storage.objects.create
# MAGIC storage.buckets.create
# MAGIC storage.buckets.get
# MAGIC aiplatform.endpoints.create
# MAGIC aiplatform.endpoints.deploy
# MAGIC aiplatform.endpoints.get
# MAGIC aiplatform.endpoints.list
# MAGIC aiplatform.endpoints.predict
# MAGIC aiplatform.models.get
# MAGIC aiplatform.models.upload
# MAGIC ```
# MAGIC 
# MAGIC - クラスタを作成し、サービスアカウントをアタッチします。Compute --> Create Cluster --> (通常の設定終了後) Advanced options --> Google Service Account --> type in your Service Account email --> start cluster
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/gcp_image_2.png" style="height: 700px; margin: 10px"/>
# MAGIC 
# MAGIC 
# MAGIC ####**MLflowとGCP python　APIを利用して、ログ化されたモデルのエンドポイントを作成する（Create an endpoint of a logged model with the MLflow and GCP python API）**
# MAGIC - 次のライブラリをノートブック野中でインストールします。
# MAGIC ```
# MAGIC %pip install google_cloud_mlflow
# MAGIC %pip install google-cloud-aiplatform
# MAGIC ```
# MAGIC 
# MAGIC - デプロイします。
# MAGIC 
# MAGIC ```
# MAGIC import mlflow
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC 
# MAGIC vtx_client = mlflow.deployments.get_deploy_client("google_cloud") # Instantiate VertexAI client
# MAGIC deploy_name = <enter-your-deploy-name>
# MAGIC model_uri = <enter-your-mlflow-model-uri>
# MAGIC deployment = vtx_client.create_deployment(
# MAGIC     name=deploy_name,
# MAGIC     model_uri=model_uri,
# MAGIC     # config={}   # set deployment configurations, see an example: https://pypi.org/project/google-cloud-mlflow/
# MAGIC     )
# MAGIC ```
# MAGIC 
# MAGIC 上記のコードは、MLflowからGoogle Storageにモデルをエクスポートし、Google Storageからモデルをインポートし、Vertex AIで画像を生成するという重いデプロイメントを行います。**デプロイメントが完了するまでに20分ほどかかるかもしれません**。
# MAGIC 
# MAGIC **注意:**
# MAGIC - `Destination_image_uri` が設定されていない場合、`gcr.io/<your-project-id>/mlflow/<deploy_name>` が使用されます。
# MAGIC - サービスアカウントは、Cloud Buildのそのストレージの場所にアクセスできる必要があります。
# MAGIC 
# MAGIC 
# MAGIC #### エンドポイントから予測値を取得する（Get predictions from the endpoint）
# MAGIC 
# MAGIC - まず、エンドポイントを取得します
# MAGIC ```
# MAGIC deployments = vtx_client.list_deployments()
# MAGIC endpt = [d["resource_name"] for d in deployments if d["name"] == deploy_name][0]
# MAGIC ```
# MAGIC 
# MAGIC - 次に、`google.cloud` の `aiplatform` モジュールを使用して、生成されたエンドポイントにクエリを発行します。 
# MAGIC ```
# MAGIC from google.cloud import aiplatform
# MAGIC aiplatform.init()
# MAGIC vtx_endpoint = aiplatform.Endpoint(endpt_resource)
# MAGIC arr = X_test.tolist() ## X_test is an array
# MAGIC pred = vtx_endpoint.predict(instances=arr)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー（Review）
# MAGIC **質問:** リアルタイム・デプロイに最適なツールは何ですか？  
# MAGIC **答え:** これは、何を望むかに大きく依存します。 考慮すべき主なツールは、コードをコンテナ化する方法と、RESTエンドポイントまたは埋め込みモデルのいずれかです。 これらの方法で、リアルタイムのデプロイメントオプションの大部分をカバーしています。
# MAGIC 
# MAGIC **質問:** 最適なRESTfulサービスは何ですか？  
# MAGIC **答え:** 主要なクラウドプロバイダーはすべて、それぞれのデプロイに関する選択肢を持っています。 Azure環境では、Azure MLは、Dockerイメージを使用してデプロイメントを管理します。これは、あなたのインフラの様々なサブシステムから問い合わせることができるRESTエンドポイントを提供します。
# MAGIC 
# MAGIC **質問:** RESTデプロイのレイテンシーに影響を与える要因は何ですか？  
# MAGIC **答え:** 応答時間は、いくつかの要因の関数と考えられます。 バッチ予測は、REST接続のオーバーヘッドを下げることによってスループットを向上させるので、必要なときに使用されるべきです。 地理的な位置も、サーバーの負荷と同様に問題です。 これは、どの地理的場所にデプロイするかと、より多くのリソースによるロードバランシングで対処できます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Lab<br>
# MAGIC 
# MAGIC このレッスンのラボを実施します。[Real Time Lab]($./Labs/02-Real-Time-Lab) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他のトピックとリソース（Additional Topics & Resources）
# MAGIC 
# MAGIC **Q:** MLflowの **`pyfunc`** に関する詳しい情報はどこで得られますか？  
# MAGIC **A:** <a href="https://www.mlflow.org/docs/latest/models.html#pyfunc-deployment" target="_blank">the MLflow documentation</a>をご覧ください。
# MAGIC 
# MAGIC **Q:** Databricksの MLflow Model Serving の詳しい情報はどこで得られますか？  
# MAGIC **A:** <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#language-python" target="_blank">この MLflow Model Serving のドキュメント</a>をご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
