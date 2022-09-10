# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="4b6ee6d2-5ae5-4a95-bd96-ce6f3a1022ab"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # 高度な実験追跡 (Advanced Experiment Tracking)
# MAGIC 
# MAGIC このレッスンでは、さまざまなユースケースに適用できる　より高度なMLflow追跡オプションについて説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは、次のことを実施します。<br>
# MAGIC  - MLflow signatureと入力例を使用してモデルの入力と出力を管理します。
# MAGIC  - オートロギング(autologging)を利用してログを自動的に記録します。
# MAGIC  - MLflowの入れ子実行（nested run）でハイパーパラメーターのチューニングと反復トレーニングの結果を探索します。
# MAGIC  - MLflowとHyperOptを統合します。
# MAGIC  - SHAPの値とビジュアライゼーションをログとして記録します。

# COMMAND ----------

# MAGIC %md <i18n value="b9f1a52f-9e49-4efa-9d7c-23e6f524f079"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Signatureと入力例 (Signatures and Input Examples)
# MAGIC 
# MAGIC 以前は、MLflowでモデルをログに記録するときに、 **`.log_model(model, model_name)`** でモデルアーティファクトにモデルとモデルの名前のみをログに保存していました。
# MAGIC 
# MAGIC ただし、モデルのsignatureと入力例もログに保存することがベストプラクティスです。これにより、スキーマのチェックを容易になり、自動デプロイツールとの統合がしやすくなります。
# MAGIC 
# MAGIC **Signature**
# MAGIC * モデルのSignatureとは、モデルの入力と出力のスキーマです。
# MAGIC * 通常、 **`infer_schema`** 関数で取得します。
# MAGIC 
# MAGIC **入力例**
# MAGIC * モデルへの入力の例です。
# MAGIC * JSONに変換され、MLflowのrunに保存されます。
# MAGIC * MLflowモデルサービングとよく統合されています。
# MAGIC 
# MAGIC 一般に、 **`.log_model(model, model_name, signature=signature, input_example=input_example)`** のような形で、モデルをログに記録します。
# MAGIC 
# MAGIC 例を見てみましょう。ここでは、 **`sklearn`** ランダムフォレスト回帰モデルを作成し、それをsignatureと入力例とともにログに記録します。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="f3575bef-8818-418a-a47b-d010dda8ff33"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC データセットを読み込むことから始めましょう。

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# COMMAND ----------

# MAGIC %md <i18n value="4f202196-a36e-4ea0-83f0-d173452281c5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC それでは、モデルをトレーニングし、MLflowでモデルのログを記録しましょう。今回は、モデルをログに記録するときに **`signature`** と **`input_examples`** を追加します。

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Signature Example") as run:
    rf = RandomForestRegressor(random_state=42)
    rf_model = rf.fit(X_train, y_train)
    mse = mean_squared_error(rf_model.predict(X_test), y_test)
    mlflow.log_metric("mse", mse)

    # Log the model with signature and input example
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    input_example = X_train.head(3)
    mlflow.sklearn.log_model(rf_model, "rf_model", signature=signature, input_example=input_example)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="e9353a0e-59d4-4913-93f6-20ddb1a3bbcb"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflow UIでこのrunのモデルのsignatureと入力例を確認しましょう。
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/signature_example.gif" >

# COMMAND ----------

# MAGIC %md <i18n value="5b4669a8-168a-43e5-be30-ec45e83c75f8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### ネストされるRun (Nested run)
# MAGIC 
# MAGIC MLflowが **`Nested Run`** という便利なRunの整理機能を提供しています。Nested Runは、親Runと子Runをツリー構造で整理します。MLflow UIでは、親Runをクリックして展開したら、子Runが表示されます。
# MAGIC 
# MAGIC 応用例:
# MAGIC * **ハイパーパラメーターチューニング**の場合、すべてのモデルトレーニングのRunを一つの親Runの下にネストしておき、ハイパーパラメーターの整理および比較がしやすくなります。
# MAGIC * IoTデバイスなどのモデルを**並行してトレーニングする**場合、全てのモデルを一つの親Runの下に集約できます。これに関する詳細は、<a href="https://databricks.com/blog/2020/05/19/manage-and-scale-machine-learning-models-for-iot-devices.html" target="_blank">こちらをご覧ください。</a>
# MAGIC * ニューラルネットワークなどの**反復トレーニング**では、 **`n`** エポックの後にチェックポイントを設定し、モデルと関連するメトリックを保存できます。

# COMMAND ----------

with mlflow.start_run(run_name="Nested Example") as run:
    # Create nested run with nested=True argument
    with mlflow.start_run(run_name="Child 1", nested=True):
        mlflow.log_param("run_name", "child_1")

    with mlflow.start_run(run_name="Child 2", nested=True):
        mlflow.log_param("run_name", "child_2")

# COMMAND ----------

# MAGIC %md <i18n value="1357e920-239c-4e96-8163-af4b95b6e7cc"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflow UIを見て、ネストされたRunを確認します。

# COMMAND ----------

# MAGIC %md <i18n value="a6f66d2b-3b27-408a-beb7-71e107fd31a6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### オートロギング (Autologging)
# MAGIC 
# MAGIC これまでに、手動でモデル、パラメータ、メトリック、およびアーティファクトのログをMLflowに記録する方法でした。
# MAGIC 
# MAGIC ただし、場合によって、これを自動的にできると便利で、MLflow Autologgingが使えます。
# MAGIC 
# MAGIC オートロギングを使用すると、明示的なログステートメントを必要とせずに、**メトリック、パラメータ、およびモデルをログに記録できます。**
# MAGIC 
# MAGIC オートロギングを有効にするには、次の2つの方法があります。
# MAGIC 
# MAGIC 1. トレーニングコードの前にmlflow.autolog()を呼び出します。これにより、インストールしたサポート対象のライブラリをインポートするとすぐに、オートロギングが有効になります。<a href="https://www.mlflow.org/docs/latest/tracking.html#automatic-logging" target="_blank">サポートされているライブラリのリストはここにあります。</a>
# MAGIC 2. コードで使用するライブラリごとに、ライブラリ固有のオートロギング機能を使用します。例えば、 **`mlflow.sklearn.autolog()`** を使用してsklearnのMLflow Autologging 機能を有効にします。
# MAGIC 
# MAGIC 最初の例をもう一度試してみましょう。今回はオートロギングだけ使用します。すべてのライブラリでオートロギングを有効にします。
# MAGIC 
# MAGIC **注:** コードを **`mlflow.start_run()`** ブロックに入れる必要はありません。

# COMMAND ----------

mlflow.autolog()

rf = RandomForestRegressor(random_state=42)
rf_model = rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md <i18n value="c0771669-fd4f-4158-a710-48b2c41ed88c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflow UIを開いて、何が自動的に記録されたかを確認しましょう。

# COMMAND ----------

# MAGIC %md <i18n value="aa9b9ec6-9df3-4c2d-8b0b-6b08d3bea7b0"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### ハイパーパラメータチューニング (Hyperparameter Tuning)
# MAGIC 
# MAGIC Nested Runとオートロギングの最も一般的な使用例の1つは、ハイパーパラメーターチューニングです。
# MAGIC 
# MAGIC たとえば、DatabricksでSparkTrialsを利用して**HyperOpt**のrunを実行すると、MLflow UIで自動的に候補モデルやパラメーターなどを子runとして追跡されます。
# MAGIC 
# MAGIC Hyperoptはハイパーパラメーターチューニングを自動化するためのツールです。以下のものを介してApache Sparkと統合できるようになりました。
# MAGIC 
# MAGIC * **Trials:** シングルノードまたは分散のMLモデルの逐次トレーニング (例:MLlib)
# MAGIC * **SparkTrials:** シングルノードモデルの並列トレーニング (例: sklearn)。並列処理の数は、 **`parallelism`** パラメータで制御されます。
# MAGIC 
# MAGIC HyperOptでSparkTrialsを使って、最適なsklearnランダムフォレストモデルを見つけてみましょう。
# MAGIC 
# MAGIC Sean Owenのブログ<a href="https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html" target="_blank">“How (Not) to Tune Your Model With Hyperopt”</a>をチェックしてください。

# COMMAND ----------

# MAGIC %md <i18n value="58aed944-4244-45b5-b982-4a113c325ae7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC HyperoptのRunをセットアップします。 最小化したい目的関数と、Hyperopt Runのパラメーターの探索空間を定義する必要があります。
# MAGIC 
# MAGIC Hyperoptは目的関数を最小化するように働くので、ここでは単純に **`loss`** をmseとして返します。それが最小化しようとしているものだからです。
# MAGIC 
# MAGIC **注意**: 精度や決定係数などのメトリックを最大化しようとする場合は、Hyperoptが目的関数を最小化できるように、 **`-accuracy`** または **`-r2`** を渡す必要があります。

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials

# Define objective function
def objective(params):
    model = RandomForestRegressor(n_estimators=int(params["n_estimators"]), 
                                  max_depth=int(params["max_depth"]), 
                                  min_samples_leaf=int(params["min_samples_leaf"]),
                                  min_samples_split=int(params["min_samples_split"]))
    model.fit(X_train, y_train)
    pred = model.predict(X_train)
    score = mean_squared_error(pred, y_train)

    # Hyperopt minimizes score, here we minimize mse. 
    return score

# COMMAND ----------

# MAGIC %md <i18n value="7a7dcc3d-3def-43a8-9042-afe826d8804a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflow Hyperoptのrunを実行します。
# MAGIC 
# MAGIC **注:** このコードはオートロギングを使用します。Hyperoptでオートロギングを使用すると、使用されたハイパーパラメーターはログに記録されますが、モデル自体はログに記録されません。上の例とは異なり、ユーザーは最適なモデルを手動で記録する必要があります。

# COMMAND ----------

from hyperopt import SparkTrials

# Define search space
search_space = {"n_estimators": hp.quniform("n_estimators", 100, 500, 5),
                "max_depth": hp.quniform("max_depth", 5, 20, 1),
                "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
                "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1)}

# Set parallelism (should be order of magnitude smaller than max_evals)
spark_trials = SparkTrials(parallelism=2)

with mlflow.start_run(run_name="Hyperopt"):
    argmin = fmin(fn=objective,
                  space=search_space,
                  algo=tpe.suggest,
                  max_evals=16,
                  trials=spark_trials)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="9a09a629-3545-4f60-942e-7a0468c9b54d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC オートロギングされた結果については、MLflow UIで確認します。HyperoptでオートロギングがネストされたRunをどのように作成したかに注目してください！
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/HyperOpt.gif" >
# MAGIC 
# MAGIC すべてのネストされたRunを選択し、 **`Compare`** を選択すると、ハイパーパラメータのチューニングプロセスをよりよく理解するために、可視化されたチャートも作成できます。
# MAGIC 
# MAGIC 上の図のように`Compare`を選択し、次の画面で **`Parallel Coordinates Plot`** を選択して以下の画像を生成します。
# MAGIC 
# MAGIC **注意**:ビジュアライゼーションチャートを生成するにはパラメーターとメトリックを追加する必要があります。
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/Parallel Coordinates Plot.png" >

# COMMAND ----------

# MAGIC %md <i18n value="662b39d5-4255-4d8a-9930-024a9a79eccd"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 高度なアーティファクト追跡 (Advanced Artifact Tracking)
# MAGIC 
# MAGIC すでに見たアーティファクトのログ情報に加えて、いくつか高度なオプションがあります。
# MAGIC 
# MAGIC 次に、以下を見ていきます。
# MAGIC * <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.shap.html#mlflow.shap" target="_blank">mlflow.shap</a>:シャープレイ特徴量（Shapley feature）における重要度を自動的に計算して記録します。
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure" target="_blank">mlflow.log_figure</a>:matplotlibとplotlyを使い可視化したものをログに記録します。

# COMMAND ----------

import matplotlib.pyplot as plt

with mlflow.start_run(run_name="Feature Importance Scores"):
    # Generate and log SHAP plot for first 5 records
    mlflow.shap.log_explanation(rf.predict, X_train[:5])

    # Generate feature importance plot
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    fig, ax = plt.subplots()
    feature_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")

    # Log figure
    mlflow.log_figure(fig, "feature_importance_rf.png")

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="e112af2e-b3ea-4855-ae09-91aad0db27c0"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflow UIで確認します。
# MAGIC 
# MAGIC <img style="width:50%" src="https://files.training.databricks.com/images/mlpupdates/artifact_examples.gif" >

# COMMAND ----------

# MAGIC %md <i18n value="4eec3fff-d56d-40cf-85b9-6c82297de99d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：追加リソース (Additional Resources)
# MAGIC 
# MAGIC * <a href="http://hyperopt.github.io/hyperopt/" target="_blank">Hyperoptのドキュメント</a>
# MAGIC * <a href="https://databricks.com/blog/2019/06/07/hyperparameter-tuning-with-mlflow-apache-spark-mllib-and-hyperopt.html" target="_blank">ハイパーパラメータチューニングに関するブログ</a>
# MAGIC * <a href="https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html#how-to-use-hyperopt-with-sparktrials" target="_blank">Spark Trials Hyperoptのドキュメント</a>
# MAGIC * <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.shap.html" target="_blank">MLflow Shapのドキュメント</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
