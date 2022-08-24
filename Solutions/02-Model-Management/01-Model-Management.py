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
# MAGIC # モデル管理 (Model Management)
# MAGIC 
# MAGIC MLflow **`pyfunc`** を利用することで、完全にカスタマイズ可能なデプロイが実現できます。このレッスンでは、さまざまな環境で作成され、さまざまな環境にデプロイされる機械学習モデルを管理する一般的な方法を説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC  - モデル管理のベストプラクティスを紹介します。
# MAGIC  - 前処理ロジック、モデルのロード、周辺アーティファクト、トレーニングアルゴリズムを含めたモデルをカスタム環境にて構築します。
# MAGIC  - カスタムMLモデルを適用します。

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 機械学習モデルの管理 (Managing Machine Learning Models)
# MAGIC 
# MAGIC モデルがトレーニング・保存されたら、次はさまざまなサービスツールで使用できるようにパッケージ化します。**プラットフォームに依存しない方法でモデルをパッケージ化すれば、デプロイにおいて柔軟性が高まり、多くのプラットフォームでモデルを再利用できます。** 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **MLflow modelsは、機械学習モデルをパッケージングするための約束事のようなものであり、自己完結するコード、環境、モデルを提供します。** <br><br>
# MAGIC 
# MAGIC * このパッケージに主な抽象化されたのは**フレーバー**という概念です。
# MAGIC   - フレーバーとはモデルの異なる使い方です。
# MAGIC   - たとえば、TensorFlowモデルはTensorFlow DAG またはPython関数として読み込むことができます。
# MAGIC   - MLflow modelsとして使用すると、両方のフレーバーが利用可能になります。
# MAGIC * `python_function`または`pyfunc`というモデルのフレーバーは、モデルを使用する一般的な方法を提供します。
# MAGIC * これにより、モデルのフォーマットを気にせず、Python関数を使用してモデルをデプロイできます。
# MAGIC 
# MAGIC **MLflowは、プラットフォームに依存しないこれらの表現を用いることで、あらゆるトレーニングフレームワークを任意のデプロイソリューションにマッピングします**。推論の複雑さを大幅に軽減します。
# MAGIC 
# MAGIC 前処理と後処理ステップ、モデルのロード時に実行される任意のコード、周辺アーティファクトなどを含む任意のロジックをパイプラインに含めるので、必要に応じてパイプラインをカスタマイズできます。 これは、モデルだけでなく全パイプラインを、MLflowのエコシステムの他の部分と連携する単一のオブジェクトとして保存できることを意味します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models-enviornments.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 最も人気のあるビルトインフレーバーは、<br><br>
# MAGIC 
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.keras.html#module-mlflow.keras" target="_blank">mlflow.keras</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#module-mlflow.sklearn" target="_blank">mlflow.sklearn</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.spark.html#module-mlflow.spark" target="_blank">mlflow.spark</a>
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/python_api/index.html" target="_blank">ここですべてのフレーバーとモジュールを見ることができます。</a>
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### `pyfunc` を使ったモデルのカスタマイズ (Custom Models using `pyfunc`)
# MAGIC 
# MAGIC **`pyfunc`** は一般的なPython機能であり、トレーニングに使用されたライブラリに関係なく、任意のロジックを定義できます。**このオブジェクトは、MLflowの機能、特にスコアリングツールなどと相互運用ができます。** そのため、すべての依存関係を持つ関連するディレクトリ構造を持つクラスとして定義されます。 予測関数などのさまざまなメソッドを持つ「ただのオブジェクト」と言えます。 前提条件がほとんどないため、MLflow、SageMaker、Spark UDF、またはその他の環境にもデプロイできます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 詳細は<a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom" target="_blank"> **`pyfunc`** ドキュメント</a>をチェックください。<br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 一般的なコード例と  **`XGBoost`** との統合に関しては、こちらを<a href="https://github.com/mlflow/mlflow/blob/master/docs/source/models.rst#example-saving-an-xgboost-model-in-mlflow-format" target="_blank">このREADME</a>をチェックください。<br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> <a href="https://mlflow.org/docs/latest/models.html#example-creating-a-custom-add-n-model" target="_blank">入力値に **`n`** を足した簡単な例</a>はこちらになります。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 前処理済のトレーニングセットを使用してランダムフォレストモデルをトレーニングします。入力DataFrameは次のステップで処理されます。<br><br>
# MAGIC 
# MAGIC 1. データにあるいくつのレビュースコア列から集計して新たな特徴量 ( **`review_scores_sum`** ) を作成します。
# MAGIC 2. データ型を強制(enforce)する
# MAGIC 
# MAGIC 予測の際、モデルを使用するたびに同じ前処理を都度実行する必要があります。
# MAGIC 
# MAGIC この処理を簡素化するために、 **`preprocess_input(self, model_input)`** というヘルパーのメソッドを含んだ **`RFWithPreprocess`** モデルのクラスを定義します。このヘルパーメソッドでは、 **`fit()`** メソッドの実行前、またはモデルの **`.predict()`** にデータを渡す前に、生データを自動的に前処理します。そうすることで、将来アプリケーションでモデルを使用するときは、データのバッチごとに前処理ロジックを実装する必要がなくなります。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC データをインポートします。

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# Change column types to simulate not having control over upstream changes
X_train["latitude"] = X_train["latitude"].astype(str)
X_train["longitude"] = X_train["longitude"].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC デプロイする対象のモデルの作成スクリプトのバージョンを確認します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> このコードは**推奨しません**。

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Copy dataset
X_train_processed = X_train.copy()

# Feature engineer an aggregate feature
X_train_processed["review_scores_sum"] = (
    X_train["review_scores_accuracy"] + 
    X_train["review_scores_cleanliness"]+
    X_train["review_scores_checkin"] + 
    X_train["review_scores_communication"] + 
    X_train["review_scores_location"] + 
    X_train["review_scores_value"]
)
X_train_processed = X_train_processed.drop([
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
], axis=1)

# Enforce data types
X_train_processed["latitude_cleaned"] = X_train["latitude"].astype(float)
X_train_processed["longitude_cleaned"] = X_train["longitude"].astype(float)
X_train_processed = X_train_processed.drop(["latitude", "longitude"], axis=1)

## Repeat the same on the test datset
# Copy dataset
X_test_processed = X_test.copy()

# Feature engineer an aggregate feature
X_test_processed["review_scores_sum"] = (
    X_test["review_scores_accuracy"] + 
    X_test["review_scores_cleanliness"]+
    X_test["review_scores_checkin"] + 
    X_test["review_scores_communication"] + 
    X_test["review_scores_location"] + 
    X_test["review_scores_value"]
)
X_test_processed = X_test_processed.drop([
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
], axis=1)

# Enforce data types
X_test_processed["latitude_cleaned"] = X_test["latitude"].astype(float)
X_test_processed["longitude_cleaned"] = X_test["longitude"].astype(float)
X_test_processed = X_test_processed.drop(["latitude", "longitude"], axis=1)

# Fit and evaluate a random forest model
rf_model = RandomForestRegressor(n_estimators=15, max_depth=5)

rf_model.fit(X_train_processed, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 重複コードがたくさんありましたよね？　
# MAGIC そして、これを本番システムにデプロイする必要な場合はどうなるでしょうか？
# MAGIC 代わりに次のコードを見てください。

# COMMAND ----------

import mlflow

class RFWithPreprocess(mlflow.pyfunc.PythonModel):

    def __init__(self, params):
        """
        Initialize with just the model hyperparameters
        """
        self.params = params
        self.rf_model = None
        self.config = None
        
    def load_context(self, context=None, config_path=None):
        """
        When loading a pyfunc, this method runs automatically with the related
        context.  This method is designed to perform the same functionality when
        run in a notebook or a downstream operation (like a REST endpoint).
        If the `context` object is provided, it will load the path to a config from 
        that object (this happens with `mlflow.pyfunc.load_model()` is called).
        If the `config_path` argument is provided instead, it uses this argument
        in order to load in the config.
        """
        if context: # This block executes for server run
            config_path = context.artifacts["config_path"]
        else: # This block executes for notebook run
            pass

        self.config = json.load(open(config_path))
      
    def preprocess_input(self, model_input):
        """
        Return pre-processed model_input
        """
        processed_input = model_input.copy()
        processed_input["review_scores_sum"] = (
            processed_input["review_scores_accuracy"] + 
            processed_input["review_scores_cleanliness"]+
            processed_input["review_scores_checkin"] + 
            processed_input["review_scores_communication"] + 
            processed_input["review_scores_location"] + 
            processed_input["review_scores_value"]
        )
        processed_input = processed_input.drop([
            "review_scores_accuracy",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value"
        ], axis=1)

        processed_input["latitude_cleaned"] = processed_input["latitude"].astype(float)
        processed_input["longitude_cleaned"] = processed_input["longitude"].astype(float)
        processed_input = processed_input.drop(["latitude", "longitude"], axis=1)
        return processed_input
  
    def fit(self, X_train, y_train):
        """
        Uses the same preprocessing logic to fit the model
        """
        from sklearn.ensemble import RandomForestRegressor

        processed_model_input = self.preprocess_input(X_train)
        rf_model = RandomForestRegressor(**self.params)
        rf_model.fit(processed_model_input, y_train)

        self.rf_model = rf_model
    
    def predict(self, context, model_input):
        """
        This is the main entrance to the model in deployment systems
        """
        processed_model_input = self.preprocess_input(model_input.copy())
        return self.rf_model.predict(processed_model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **`context`** パラメータは、MLflowによって、下流ツールの中に自動的に提供されます。これは、簡単にシリアル化できないモデル (例えば、 **`keras`** モデル) やカスタム設定ファイルなどのカスタム依存オブジェクトを追加するために使用できます。
# MAGIC 
# MAGIC 以下の手順で設定ファイルを提供します。手順に注意してください。<br><br>
# MAGIC 
# MAGIC - クラスにロードしたい全てのファイルを保存します。
# MAGIC - キーと値のペアのアーティファクト辞書を作成します。値はそのオブジェクトへのパスです。
# MAGIC - モデルを保存すると、すべてのアーティファクトが下流で使用できるように同じディレクトリにコピーされます。
# MAGIC 
# MAGIC ここでは、いくつかのモデルのハイパーパラメーターをconfigとして保存します。

# COMMAND ----------

import json 
import os

params = {
    "n_estimators": 15, 
    "max_depth": 5
}

# Designate a path
config_path = f"{DA.paths.working_path}/data.json"

# Save the results
with open(config_path, "w") as f:
    json.dump(params, f)

# Generate an artifact object to saved
# All paths to the associated values will be copied over when saving
artifacts = {"config_path": config_path} 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC クラスをインスタンス化します。 **`load_context`** を実行してconfigをロードします。これは下流のサービングツールで自動実行される際に使われます。

# COMMAND ----------

model = RFWithPreprocess(params)

# Run manually (this happens automatically in serving integrations)
model.load_context(config_path=config_path) 

# Confirm the config has loaded
model.config

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルをトレーニングします。前処理ロジックが自動的に実行されることに注意してください。

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 予測を生成します。

# COMMAND ----------

predictions = model.predict(context=None, model_input=X_test)
predictions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルのSignatureを生成します。

# COMMAND ----------

from mlflow.models.signature import infer_signature

signature = infer_signature(X_test, predictions)
signature

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC conda環境を生成します。任意の複雑なものになる可能性があります。 **`mlflow.sklearn`** を使うときに必要なもので、 **`sklearn`** のバージョンが自動的にログに記録されます。 **`pyfunc`** を使う場合では、これでデプロイメント環境を手動で構築する必要があります。

# COMMAND ----------

from sys import version_info
import sklearn

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {"pip": ["mlflow",
                 f"scikit-learn=={sklearn.__version__}"]
        },
    ],
    "name": "sklearn_env"
}

conda_env

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルを保存します。

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "rf_preprocessed_model", 
        python_model=model, 
        artifacts=artifacts,
        conda_env=conda_env,
        signature=signature,
        input_example=X_test[:3] 
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルを **`python_function`** 形式でロードします。

# COMMAND ----------

mlflow_pyfunc_model_path = f"runs:/{run.info.run_id}/rf_preprocessed_model"
loaded_preprocess_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルを適用します。

# COMMAND ----------

loaded_preprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **`pyfunc` は任意の下流サービングツールと相互運用することができることに注意してください。これで任意のコード、適切なライブラリ、その他の情報が使用できるようになります。**

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC **質問:** MLflow　projectはモデルとどう違うのですか?<br>
# MAGIC **答え:** MLflow projectの役割は、runの再現性とコードのパッケージ化です。 MLflow modelsは、さまざまな環境へデプロイする役割を持っています。
# MAGIC 
# MAGIC **質問:** ML モデルフレーバーって何ですか?<br>
# MAGIC **答え:** フレーバーは、デプロイツールがモデルを理解するために使用する約束事です。これらフレーバーにより、各ライブラリを実装しなくても、任意のMLライブラリのモデルを適用できるデプロイツールを開発できます。 各トレーニング環境をデプロイ環境にマッピングする代わりに、MLモデルフレーバーがこのマッピングを管理・実施してくれます。
# MAGIC 
# MAGIC **質問:** どうやってモデルに前処理と後処理ロジックを追加すればいいですか?<br>
# MAGIC **答え:** **`mlflow.pyfunc.PythonModel`** を使って拡張したモデルクラスでは、データのロード、前処理、および後処理のロジックを実装することが可能です。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ
# MAGIC 
# MAGIC このレッスンのラボを実施します。 [Model Management Lab]($./Labs/01-Model-Management-Lab) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q:** MLflow modelsの詳細情報はどこで入手できますか?<br>
# MAGIC **A:**  <a href="https://www.mlflow.org/docs/latest/models.html" target="_blank">MLflow のドキュメントをチェックしてください。</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
