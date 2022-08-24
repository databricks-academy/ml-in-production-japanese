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
# MAGIC # 実験の追跡 (Experiment Tracking)
# MAGIC 
# MAGIC 機械学習のライフサイクルでは、さまざまなハイパーパラメーターとライブラリを使用して複数のアルゴリズムを学習し、それぞれパフォーマンスが異なる性能結果や学習済みモデルを得ることができます。 このレッスンでは、機械学習のライフサイクルを整理するために、これらの実験を追跡する方法を説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは、次のことを実施します。<br>
# MAGIC  - MLflowでML実験追跡の紹介
# MAGIC  - 実験のログを記録してUIで可視化
# MAGIC  - パラメータ、メトリック、モデルの記録
# MAGIC  - 過去のrun(コードの実行)をプログラム的に照会

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 機械学習のライフサイクルを通して...<br>
# MAGIC * データサイエンティストは、さまざまなモデルをテストし、
# MAGIC * 様々なライブラリを使い、
# MAGIC * それぞれ異なるハイパーパラメータを使います。
# MAGIC 
# MAGIC これらのさまざまな結果を追跡するには、次のような組織的なチャレンジがあります...<br>
# MAGIC * 実験の保存
# MAGIC * 結果の保存の保存
# MAGIC * モデルの保存
# MAGIC * 補足アーティファクトの保存
# MAGIC * コードの保存
# MAGIC * データスナップショットの保存

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### MLflowによる実験の追跡 (Tracking Experiments with MLflow)
# MAGIC 
# MAGIC MLflow Tracking とは...<br>
# MAGIC 
# MAGIC * 機械学習に特有のLogging API。
# MAGIC * トレーニングを行うライブラリや環境に縛られない。
# MAGIC * データサイエンスコードの実行である**run**の概念を中心に整備されている。
# MAGIC * Runは**実験(Experiments)**の中に集約され、一つの実験に多数のrunが含まれます。
# MAGIC * MLflow Serverは多数の実験をホストできます。
# MAGIC 
# MAGIC 各runでは、次の情報を記録できます。<br>
# MAGIC 
# MAGIC * **Parameters:** 入力パラメータのキーと値のペア。例：ランダムフォレストモデルの木の本数。
# MAGIC * **Metric:** 評価指標。例：RMSE、AUC。
# MAGIC * **Artifacts:** 任意形式の任意の出力ファイルで、実験の過程で出力された作成物。 例：画像、モデルのpickleファイル、データファイル。
# MAGIC * **Source:** runした時のソースコード。
# MAGIC 
# MAGIC Experiments(実験)は、Python、R、および Javaのライブラリを使用して追跡することも、CLIおよびREST callを使用して追跡することもできます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 実験のロギングとUI (Experiment Logging and UI)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC San Francisco Airbnbの物件データをロードしてください。これを使ってモデルをトレーニングします。

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **画面上部の「Experiment」ボタンをクリックしてMLflow UIに移動します。**
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/>Databricksワークスペース内のすべてのPythonノートブックには独自の実験があります。MLflowをノートブックに使用すると、全てのrunがノートブック実験に記録されます。ノートブックの実験は、対応するノートブックの名前とIDを使用します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 以下を実行して、基本的な実験のものをログします。<br>
# MAGIC 
# MAGIC 1. **`mlflow.start_run()`** を使って実験を開始し、その時にrunの名前をパラメータとして渡します。
# MAGIC 2. モデルをトレーニングします。
# MAGIC 3. **`mlflow.sklearn.log_model()`** を使用してモデルをログします。
# MAGIC 4. **`mlflow.log_metric()`** を使用してモデルの評価指標(メトリック)をログします。
# MAGIC 5. **`run.info.run_id`** を使ってrun IDを出力します。

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Run") as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Log metrics
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC UIで結果を確認します。 以下の項目を確認してください。<br>
# MAGIC 
# MAGIC 1. `Experiment ID`。
# MAGIC 2. runが実行された時刻。**Start Timeをクリックすると、 runに関する詳細情報が表示されます。**
# MAGIC 3. runを実行したコード。
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/mlflow/mlflow_exp_ui.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Start Timeをクリックしたら、以下の項目を確認します。<br>
# MAGIC 
# MAGIC 1. Run IDは上記出力されたものと一致しています。
# MAGIC 2. 保存したモデルには、モデルのpickleファイル、Conda環境、 **`MLmodel`** ファイルが含まれています。詳細については次のレッスンで説明します。
# MAGIC 
# MAGIC 
# MAGIC <div><img src="http://files.training.databricks.com/images/mlflow/mlflow_model_page.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### パラメータ、Metric、およびArtifact (Parameters, Metrics, and Artifacts)
# MAGIC 
# MAGIC しかし、待ってください、それだけではないです！ 最後の例では、runの名前、評価metric、およびモデル自体をArtifactsとしてログしました。 次に、パラメーター、複数のmetric、および特徴量の重要性を含むその他のArtifactをログしましょう。
# MAGIC 
# MAGIC まず、これらを実行する関数を作成します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Artifactをログするには、MLflowでログする前にどこかに保存する必要があります。 ここでは、一時ファイルを使用します。

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def log_rf(experiment_id, run_name, params, X_train, X_test, y_train, y_test):
  
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Create model, train it, and create predictions
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "mse": mean_squared_error(y_test, predictions), 
            "mae": mean_absolute_error(y_test, predictions), 
            "r2": r2_score(y_test, predictions)
        })

        # Log feature importance
        importance = (pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=False))
        importance_path = f"{DA.paths.working_path}/importance.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "feature-importance.csv")

        # Log plot
        fig, ax = plt.subplots()
        importance.plot.bar(ax=ax)
        plt.title("Feature Importances")
        mlflow.log_figure(fig, "feature_importances.png")
        display(fig)

        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 新しいパラメータを指定して実行します。

# COMMAND ----------

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

log_rf(experiment_id, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC UIでログをどのように表示されるかを確認します。Artifactからプロットファイルの保存場所を確認してください。
# MAGIC 
# MAGIC 次に、3回目のrunを実行します。

# COMMAND ----------

params_1000_trees = {
    "n_estimators": 1000,
    "max_depth": 10,
    "random_state": 42
}

log_rf(experiment_id, "Third Run", params_1000_trees, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 過去のrunを照会する (Querying Past Runs)
# MAGIC 
# MAGIC ログデータをPythonで再利用するために、過去のrunをプログラム的にクエリします。 クエリするには、 **`MlflowClient`** オブジェクトを利用します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> `client.set_tag(run.info.run_id, "tag_key", "tag_value")`を使ってrunのタグを設定することもできます。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 次に、 **`.list_run_infos()`** を使用し、実験のすべてのrunをリストします。これは **`experiment_id`** をパラメータとして渡します。

# COMMAND ----------

display(client.list_run_infos(experiment_id))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC いくつかの項目を引き出し、Spark DataFrameを作成します。

# COMMAND ----------

runs = spark.read.format("mlflow-experiment").load(experiment_id)
display(runs)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 最後のrunの情報を抽出し、関連するArtifactをリストします。

# COMMAND ----------

run_rf = runs.orderBy("start_time", ascending=False).first()

client.list_artifacts(run_rf.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 最後のrunの評価指標を抽出します。

# COMMAND ----------

client.get_run(run_rf.run_id).data.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC モデルをリロードし、特徴量の重要性を確認します。

# COMMAND ----------

model = mlflow.sklearn.load_model(f"runs:/{run_rf.run_id}/random_forest_model")
model.feature_importances_

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC **質問:** MLflow trackingでは何をログとして取得できますか？  
# MAGIC **答え:** MLflowは以下のものをログとして取得できます:
# MAGIC - **Parameter(パラメータ):** モデルへの入力
# MAGIC - **Metrixs(評価指標):** モデルのパフォーマンス
# MAGIC - **Artifact(作成物):** データ、モデル、画像などあらゆるオブジェクト
# MAGIC - **Source（ソースコード）:** runした時のコード。gitにリンクされている場合はコミットハッシュを含む
# MAGIC 
# MAGIC **質問:** 実験はどのようにLogをとることができますか?  
# MAGIC **答え:** 実験では、まずrunを作成し、そのrunオブジェクトにLogging method(例えば  **`run.log_param("MSE", .2)`** ) を使用することによってログを取得できます。
# MAGIC 
# MAGIC **質問:** ログとして出力されたartifactはどこに保存されますか?  
# MAGIC **回答:** ログとして出力されたartifactは、選択したディレクトリに保存されます。 Databricksでは、DBFS (Databricks ファイルシステム) に保存されます。
# MAGIC 
# MAGIC **質問:** 過去のrunを照会するにはどうしたらいいですか?  
# MAGIC **答え:** **`MlflowClient`** オブジェクトを使用して照会できます。 これでUIでできることはすべてプログラム的に実行できるため、プログラミング環境から出る必要はありません。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ
# MAGIC 
# MAGIC このレッスンのラボに進みます。 [実験の追跡Lab]($./Labs/02-Experiment-Tracking-Lab)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q:** MLflowのハイレベルな説明がありますか?  
# MAGIC **A:** <a href="https://databricks.com/session/accelerating-the-machine-learning-lifecycle-with-mlflow-1-0" target="_blank">2019年のSpark SummitでのSparkとMLflowのクリエイターであるMatei Zahariaの講演を見てください。</a>
# MAGIC 
# MAGIC **Q:** 様々な機械学習ツールのより大きな背景に関する情報ソースはありますか?  
# MAGIC **A:** <a href="https://roaringelephant.org/2019/06/18/episode-145-alex-zeltov-on-mlops-with-mlflow-kubeflow-and-other-tools-part-1/#more-1958" target="_blank">Roaring Elephantポッドキャストのこのエピソードをチェックしてください。</a>
# MAGIC 
# MAGIC **Q:** MLflowに関連するドキュメントはどこにありますか?  
# MAGIC **A:** <a href="https://www.mlflow.org/docs/latest/index.html" target="_blank">ドキュメントはここにあります。</a>
# MAGIC 
# MAGIC **Q:** 機械学習を学ぶために一般的な学習材料はありますか?  
# MAGIC **A:** <a href="https://www-bcf.usc.edu/~gareth/ISL/" target="_blank">_統計学習入門_</a>は、機械学習のテーマと基本的なアプローチを理解する入門書としてお薦めします。
# MAGIC 
# MAGIC **Q:** Sparkを使った機械学習に関する詳細情報はどこで入手できますか?  
# MAGIC **A:** <a href="https://databricks.com/blog/category/engineering/machine-learning" target="_blank">機械学習に特化したDatabricks Blog</a>をご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
