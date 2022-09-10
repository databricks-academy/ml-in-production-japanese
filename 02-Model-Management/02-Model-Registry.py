# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d5223bc2-6b7c-4596-b945-ed2f2cb22d2e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # モデル・レジストリ (Model Registry)
# MAGIC 
# MAGIC MLflow Model Registryは、チームがMLモデルを共有したり、実験からオンラインテストおよび本番稼働まで一緒に作業したりすることができる共同ハブです。モデルについての承認およびガバナンスワークフローと統合し、MLモデルのデプロイとパフォーマンスの監視もできます。 このレッスンでは、MLflow モデル・レジストリを使用したモデルの管理方法について説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC  - MLflowを利用したモデルの登録
# MAGIC  - 本番システムへのモデルデプロイ
# MAGIC  - 本番稼働中のモデルを新バージョンに更新（テストのためのステージングを含む）
# MAGIC  - モデルのアーカイブと削除

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="c4f47d56-1cc8-4b97-b89f-63257dbb3e31"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### モデル・レジストリ (Model Registry)
# MAGIC 
# MAGIC MLflow Model Registryは、MLflow Modelの完全なライフサイクルを共同で管理するための一元化されたモデルストア、APIのセット、およびUIを揃えているコンポーネントです。モデル・リネージ(どのMLflow Experiment、どのRunで構築されたモデルか)、モデルのバージョン管理、ステージの移行(「Staging」から「Production」への移行など)、アノテーション(コメント、タグなど)、デプロイ管理(どの本番用ジョブが特定のモデルバージョンをリクエストしたかなど)を提供します。
# MAGIC 
# MAGIC モデルレジストリには以下の機能があります。<br>
# MAGIC 
# MAGIC * **中央リポジトリ:** モデルをMLflowモデル・レジストリに登録します。登録済みモデルには、一意の名前、バージョン、ステージ、およびその他のメタデータがあります。
# MAGIC * **モデルのバージョン管理:** 更新時に、登録済みモデルのバージョンを自動的に追跡します。
# MAGIC * **モデルのステージ:** モデルのライフサイクルを表す「Staging」や「Production」など、各モデルバージョンに割り当てられたものです。
# MAGIC * **モデルステージの移行:** モデルを別のステージへ変更することです。その際に、新しいモデルの登録またはモデルのステージの変更を行動履歴として記録し、ユーザー名、変更点、およびコメントなどの追加メタデータを自動的にログに記録します。
# MAGIC * **CI/CDワークフロー統合:** より良い制御とガバナンスを行うため、ステージの移行・変更・レビュー・承認をCI/CDパイプラインの一部として記録します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> モデル・レジストリの詳細については、<a href="https://mlflow.org/docs/latest/registry.html" target="_blank">MLflowのドキュメント</a>を参照してください。

# COMMAND ----------

# MAGIC %md <i18n value="ae6da8a8-dcca-4d34-b7fc-f06395f339f0"/>
# MAGIC 
# MAGIC ### モデルの登録 (Registering a Model)
# MAGIC 
# MAGIC 次のワークフローは、UIまたは純粋なPythonで動作します。 このノートブックは純粋なPythonを使用します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/>スクリーンの左側にある"モデル"をクリックして同じことをUIで操作してみてください。

# COMMAND ----------

# MAGIC %md <i18n value="c06f6e05-ab27-41da-8466-c092b3fcc1f6"/>
# MAGIC 
# MAGIC モデルをトレーニングし、MLflowのログに記録します。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

n_estimators = 100
max_depth = 5

rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf, "model", input_example=input_example, signature=signature)
    mlflow.log_metric("mse", mean_squared_error(y_test, rf.predict(X_test)))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="ee061209-c472-4fea-9a05-52fc45a80bb4"/>
# MAGIC 
# MAGIC 他のワークスペースユーザーが作ったモデルと混在しないように、一意のモデル名を作成します。

# COMMAND ----------

import uuid

uid = uuid.uuid4().hex[:6]
model_name = f"airbnb-rf-model_{uid}"
model_name

# COMMAND ----------

# MAGIC %md <i18n value="0650f73b-bf51-4494-8262-c04f470c73bd"/>
# MAGIC 
# MAGIC モデルを登録します。

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="99071a6c-e260-40d2-b795-dc9782d3fc2b"/>
# MAGIC 
# MAGIC **画面左側の*モデル*タブを開き、登録済みモデルを探索します。** 次の点に注目してください。<br>
# MAGIC 
# MAGIC * どのユーザーがモデルをトレーニングしたか、どのコードを使用したかを記録しています。
# MAGIC * このモデルで実行されたアクションの履歴を記録しています。
# MAGIC * このモデルはバージョン１として記録されています。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/registered_model_new.png" style="height: 600px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="cd30915c-1fa2-422e-9137-c9c509bed8f4"/>
# MAGIC 
# MAGIC ステータスを確認します。 最初は **`PENDING_REGISTRATION`** ステータスになります。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="2b32a505-bbbd-4c5b-9967-bab1054784c8"/>
# MAGIC 
# MAGIC モデルに関する説明を追加します。

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

# MAGIC %md <i18n value="89c43607-2402-4a80-97ee-b6b1378d5563"/>
# MAGIC 
# MAGIC バージョンの説明を追加します。

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using sklearn."
)

# COMMAND ----------

# MAGIC %md <i18n value="932e4581-9172-4720-833c-210174f8814e"/>
# MAGIC 
# MAGIC 
# MAGIC ### モデルのデプロイ (Deploying a Model)
# MAGIC 
# MAGIC MLflow model registoryでは、 **`None`** , **`Staging`** , **`Production`** と **`Archived`** という4つのモデルステージが定義されています。各ステージには固有の意味があります。例えば、 **`Staging`** はモデルテスト用、 **`Production`** はテストまたはレビュープロセスを完了し、アプリケーションにデプロイされたモデル用です。
# MAGIC 
# MAGIC 適切な権限を持つユーザーは、ステージ間でモデルを移行できます。 組織の管理者は、ユーザごとおよびモデルごとにこれらの権限を制御できます。
# MAGIC 
# MAGIC モデルを特定のステージに移行させる権限がある場合は、 **`MlflowClient.update_model_version()`** 関数を使用して直接移行させることができます。権限がない場合は、REST APIを使用してステージ移行をリクエストできます。例えば、 ***```%sh curl -i -X POST -H "X-Databricks-Org-Id: <YOUR_ORG_ID>" -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" https://<YOUR_DATABRICKS_WORKSPACE_URL>/api/2.0/preview/mlflow/transition-requests/create -d '{"comment": "このモデルを「production」ステージに移行してください！", "model_version": {"version": 1, "registered_model": {"name": "power-forecasting-model"}}, "stage": "Production"}'```*** 

# COMMAND ----------

# MAGIC %md <i18n value="64af3f73-388f-42a4-86cc-a3e2e5f713db"/>
# MAGIC 
# MAGIC モデルを **`Production`** ステージに移行します。

# COMMAND ----------

import time

time.sleep(10) # In case the registration is still pending

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md <i18n value="d486cd3a-485e-4470-9017-6b03058901da"/>
# MAGIC 
# MAGIC モデルの現在のステータスを取得します。

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md <i18n value="2bb842d8-88dd-487a-adfb-3d69aa9aa1a5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **`pyfunc`** を使って最新モデルをロードします。 この方法でモデルをロードすると、トレーニングに使用されたパッケージに関係なくモデルを使用できます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 指定したバージョンのモデルもロードできます。

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md <i18n value="a7ee0d75-f55c-477a-938b-719ea17c2b8c"/>
# MAGIC 
# MAGIC モデルを適用します。

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="2cdee524-1465-4bfb-9e62-5ed1307b353e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### モデルの新しいバージョンのデプロイ (Deploying a New Model Version)
# MAGIC 
# MAGIC MLflowモデルレジストリを使用すると、1つの登録済みモデルに対応する複数のモデルバージョンを作成できます。ステージ移行を実行することにより、モデルの新しいバージョンをステージングまたは本番にシームレスにデプロイできます。

# COMMAND ----------

# MAGIC %md <i18n value="d0bda674-ba42-4fa5-9d7c-194e567e1478"/>
# MAGIC 
# MAGIC モデルの新しいバージョンを作成し、ログに記録するときにそのモデルを登録することもできます。

# COMMAND ----------

n_estimators = 300
max_depth = 10

rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)

input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))

with mlflow.start_run(run_name="RF Model") as run:
    # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
    # function to register the model with the MLflow Model Registry. This automatically
    # creates a new model version
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="sklearn-model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature
    )
    mlflow.log_metric("mse", mean_squared_error(y_test, rf.predict(X_test)))

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="27044c82-1822-4090-96a1-82e266b1ce98"/>
# MAGIC 
# MAGIC 
# MAGIC UIでモデルの新しいバージョンを確認します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/model_version_new.png" style="height: 600px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="7d124f10-d8ef-4af6-8e5d-6d0b3d5eecae"/>
# MAGIC 
# MAGIC 検索機能を使用してモデルの最新バージョンを取得します。

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
print(f"New model version: {new_model_version}")

# COMMAND ----------

# MAGIC %md <i18n value="d10030fe-a13e-4fcb-a13e-056d1bcb8683"/>
# MAGIC 
# MAGIC 
# MAGIC この新バージョンに説明を追加します。

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version is a random forest containing 300 decision trees and a max depth of 10 that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md <i18n value="039af76f-ed42-4d52-bd96-09f4853322be"/>
# MAGIC 
# MAGIC モデルの新バージョンを **`Staging`** に移行します。

# COMMAND ----------

time.sleep(10) # In case the registration is still pending

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md <i18n value="bba43d4f-da53-4f09-a324-a1fd6f3f7d74"/>
# MAGIC 
# MAGIC このモデルは現在ステージングであるため、 **`Production`** に移行する前に自動化CI/CDパイプラインを実行してモデルをテストすることができます。 テストが完了したら、そのモデルを **`Production`** に移行します。

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production",
    archive_existing_versions=True # Archive old versions of this model
)

# COMMAND ----------

# MAGIC %md <i18n value="2620b2ab-8fa7-4118-8212-5b9f19af7a8d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 削除 (Deleting)
# MAGIC 
# MAGIC モデルの古いバージョンを削除できます。

# COMMAND ----------

# MAGIC %md <i18n value="49c8fbc4-6b5d-4939-a8e7-2b6cc0945909"/>
# MAGIC 
# MAGIC バージョン1を削除します。 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> アーカイブされていないモデルのバージョンは削除できません。

# COMMAND ----------

client.delete_model_version(
    name=model_name,
    version=1
)

# COMMAND ----------

# MAGIC %md <i18n value="36e94f58-20c9-4a0e-bc57-3a5c258591a0"/>
# MAGIC 
# MAGIC モデルのバージョン2もアーカイブします。

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

# MAGIC %md <i18n value="967a313e-45ca-4d7d-b423-3c660e0f6136"/>
# MAGIC 
# MAGIC モデル全体を削除します。

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md <i18n value="cd34a8c0-ed7d-458b-ac28-55664acd3231"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC **質問:** MLflow trackingとモデル・レジストリはどこか違いますか?  
# MAGIC **答え:** トラッキングは、実験と開発のプロセスの追跡です。 モデル・レジストリは、トラッキングからモデルを取得し、ステージングを経て本番稼働に移行するように設計されています。 これは、データエンジニアや機械学習エンジニアがよくデプロイプロセスに担当することです。
# MAGIC 
# MAGIC **質問:** モデル・レジストリが必要なのはなぜですか?  
# MAGIC **答え:** MLflow　trackingが機械学習トレーニング・プロセスのエンドツーエンドの再現性を提供するのと同じように、モデル・レジストリはデプロイプロセスの再現性とガバナンスを提供します。 本番システムはミッションクリティカルなので、コンポーネントはACLで分離できるため、特定のユーザーしかが本番モデルを変更できません。 バージョン管理とCI/CDワークフロー統合は、モデルを本番環境にデプロイする際の重要な役割でもあります。
# MAGIC 
# MAGIC **質問:** UIを使用する場合と比較して、プログラム的に何を実行できますか?  
# MAGIC **答え:** ほとんどの操作はUIまたは純粋なPythonで行うことができます。 モデルのトラッキングはPythonで実施する必要がありますが、それ以降の処理はどちらの方法でも実行できます。 例えば、MLflow tracking APIを使用してログに記録されたモデルをUIで登録し、「Production」にプロモートすることができます。

# COMMAND ----------

# MAGIC %md <i18n value="9aeb78b7-1eba-4bd9-ae48-ab4cd792012f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ (Next Steps)
# MAGIC 
# MAGIC 次にレッスンに進みます。[Webhooks and Testing]($./03a-Webhooks-and-Testing) 

# COMMAND ----------

# MAGIC %md <i18n value="7cae8131-c410-449a-9581-f52899a6c799"/>
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q: ** MLflow modelの詳細情報はどこで入手できますか? 
# MAGIC **A: **  <a href="https://www.mlflow.org/docs/latest/models.html" target="_blank">MLflowのドキュメントをチェックしてください。</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
