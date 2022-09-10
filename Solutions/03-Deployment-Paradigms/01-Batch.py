# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b8f7f476-820c-470b-b998-ee49d8090103"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # バッチ・デプロイ(Batch Deployment)
# MAGIC 
# MAGIC バッチ推論は、機械学習モデルを展開する最も一般的な方法です。
# MAGIC このレッスンでは、JVM上で、Sparkと書き込みの最適化　を含むバッチを使用してモデルをデプロイするためのさまざまな戦略を紹介します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC 
# MAGIC  - バッチ・デプロイを検討する。
# MAGIC  - Spark DataFrameに **`sklearn`** モデルを適用し、結果を保存する。
# MAGIC  - パーティショニングやZ-orderingを含む書き込み最適化機能を利用する。

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="c69d373e-764d-4d90-89af-cd7ec988e8ce"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### バッチでの推論(Inference in Batch)
# MAGIC 
# MAGIC バッチ・デプロイは、機械学習モデルをデプロイするためのユースケースの大部分を占めます。<br><br> 
# MAGIC * これは通常、モデルにより予測を実行し、後で使用するためにどこかに保存することを意味します。
# MAGIC * ライブサービングの場合、結果は多くの場合、保存された予測をクイックに提供するデータベースに保存されます。
# MAGIC * それ以外のケースでは、電子メールのように、BLOBストアのようなパフォーマンスの低いデータストアに保存されることがあります。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/batch-predictions.png" width=800px /></img>
# MAGIC 
# MAGIC 推論の結果の書き込みは、さまざまな方法で最適化することができます。<br><br>
# MAGIC 
# MAGIC * 大容量データの場合、予測と書き込みは並列に行うべきです。
# MAGIC * **保存された予測結果へのアクセス・パターンは、データの書き込み方法にも留意する必要があります**。
# MAGIC   - 静的ファイルやデータウェアハウスの場合、パーティショニングによりデータの読み込みを高速化します。
# MAGIC   - データベースの場合、関連するクエリに対してインデックスを作成すると、一般的にパフォーマンスが向上します。
# MAGIC   - いずれの場合も、インデックスは本の索引に似た働きをします：　関連するコンテンツにスキップすることが可能です。
# MAGIC   

# COMMAND ----------

# MAGIC %md <i18n value="25510128-3adc-46f2-a87a-fd25250917b0"/>
# MAGIC 
# MAGIC モデルの精度を担保するためには、他にもいくつかの考慮事項があります。<br><br>
# MAGIC 
# MAGIC * 第一は、モデルが期待値と一致することを確認することです。
# MAGIC   - この点については、モデルドリフトセクションでさらに詳しく説明します。
# MAGIC * 第二に、**データセットの大部分でモデルを再トレーニングすること**です。
# MAGIC   - データセット全体、もしくは95%程度をトレーニングに使用します。
# MAGIC   - 訓練とテストの分割は、ハイパーパラメータを調整し、モデルが未知のデータでどのように動作するかを推定するのに有効な方法です。
# MAGIC   - データセットの大部分でモデルを再トレーニングすることで、可能な限り多くの情報をモデルに反映させることができます。

# COMMAND ----------

# MAGIC %md <i18n value="a6956622-b714-4906-b513-6c84e79399b8"/>
# MAGIC 
# MAGIC ### Sparkにおける推論(Inference in Spark)
# MAGIC 
# MAGIC 様々な機械学習ライブラリで学習したモデルは、Sparkを使用してスケールで適用することができます。 これを行うには、 **`mlflow.pyfunc.spark_udf`** を使用し、 **`SparkSession`**, モデル名, run IDを渡します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> SparkでUDFを使用するということは、クラスタの各ノードにサポートライブラリをインストールする必要があります。 **`sklearn`** の場合、これはデフォルトでDatabricksクラスタにインストールされています。 他のライブラリを使用する場合は、UDFとして動作するためにインストールする必要があります。 

# COMMAND ----------

# MAGIC %md <i18n value="a2a01839-d58e-480c-a827-a6fd16718b28"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC まず、 **`sklearn`** モデルのトレーニングから始めます。 **`mlflow`** で生成されたSpark UDFを使用して適用します。
# MAGIC 
# MAGIC データをインポートします。 **train/testの分割は行わないでください** 。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 最終的なモデルの学習では、トレーニングとテスト（train/test split）をスキップするのが一般的です。

# COMMAND ----------

import pandas as pd

df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X = df.drop(["price"], axis=1)
y = df["price"]

# COMMAND ----------

# MAGIC %md <i18n value="3110b46a-f08d-437b-a54b-99891bf3095a"/>
# MAGIC 
# MAGIC 
# MAGIC トレーニングとモデルのロギング（Train and log model）

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow.sklearn

with mlflow.start_run(run_name="Final RF Model") as run:
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf.fit(X, y)

    predictions = rf.predict(X)
    mlflow.sklearn.log_model(rf, "random_forest_model")

    mse = mean_squared_error(y, predictions) # This is on the same data the model was trained
    mlflow.log_metric("mse", mse)

# COMMAND ----------

# MAGIC %md <i18n value="4c5710a9-3cc1-49b6-81eb-9acf386afbad"/>
# MAGIC 
# MAGIC 
# MAGIC PandasデータフレームからSparkデータフレームの作成

# COMMAND ----------

spark_df = spark.createDataFrame(X)
display(spark_df)

# COMMAND ----------

# MAGIC %md <i18n value="0c892310-6d74-44df-a42e-cfea83296be4"/>
# MAGIC 
# MAGIC 
# MAGIC MLflowは簡単にSpark user defined function(UDF)を生成します。
# MAGIC これにより、Python環境と”Sparkを用いた大規模なモデル適用”とのギャップを埋めることができます。

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/random_forest_model")

# COMMAND ----------

# MAGIC %md <i18n value="edd6acae-9489-47c7-9d46-8a730ed48c3c"/>
# MAGIC 
# MAGIC 
# MAGIC 関数の入力として列名を使って、標準的なUDFとしてモデルを適用します。

# COMMAND ----------

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))

display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="1fa8ab51-0dc8-433c-92e2-ae76e661e060"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 書き込み最適化（Write Optimizations）
# MAGIC 
# MAGIC バッチ・デプロイのシナリオに応じて、多くの最適化が可能です。 特にSpark と Delta Lake において、次のような最適化が可能です:<br><br>
# MAGIC 
# MAGIC 
# MAGIC - **Partitioning:** 異なるカテゴリ値に関連するデータを異なるディレクトリに格納します。
# MAGIC - **Z-Ordering：** 関連する情報を、同じファイル群に配置します。
# MAGIC - **Data Skipping：** フィルタ（WHERE句）を含むクエリを高速化します。
# MAGIC - **Partition Pruning：** 読み込むデータ量を制限することでクエリーを高速化します。
# MAGIC 
# MAGIC 
# MAGIC その他の最適化を含む:<br><br>
# MAGIC 
# MAGIC - **Database indexing:** 特定のテーブル・カラムをより効果的にクエリーできるようにします。
# MAGIC - **Geo-replication:** データを異なる場所（geographical regions）に複製します。

# COMMAND ----------

# MAGIC %md <i18n value="5632d9fc-878e-4253-9911-505eb2ea8e2e"/>
# MAGIC 
# MAGIC 
# MAGIC 近い範囲で、パーティション（Partition）を設定しましょう。

# COMMAND ----------

dbutils.fs.rm(f"{DA.paths.working_dir}/batch-predictions-partitioned.delta", recurse=True)
delta_partitioned_path = f"{DA.paths.working_dir}/batch-predictions-partitioned.delta"

prediction_df.write.partitionBy("neighbourhood_cleansed").mode("OVERWRITE").format("delta").save(delta_partitioned_path)

# COMMAND ----------

# MAGIC %md <i18n value="daa95012-b4cb-4c40-8931-cf328b7b3e3d"/>
# MAGIC 
# MAGIC 
# MAGIC ファイルを見てみましょう。
# MAGIC Take a look at the files.

# COMMAND ----------

display(dbutils.fs.ls(delta_partitioned_path))

# COMMAND ----------

# MAGIC %md <i18n value="f3232c00-22de-4b36-97d6-1c3c76b30121"/>
# MAGIC 
# MAGIC 
# MAGIC Z-Orderingは、多次元クラスタリングの一種で、関連する情報を同じファイル群に配置するものです。 読み込む必要のあるデータ量を減らすことができます。 
# MAGIC <a href="https://docs.databricks.com/delta/optimizations/file-mgmt.html#z-ordering-multi-dimensional-clustering" target="_blank">詳しくはこちらをご覧ください。</a> 郵便番号でZ-0rder してみましょう。Z-Orderingは、多次元クラスタリングの一種で、関連する情報を同じファイル群に配置するものです。 読み込む必要のあるデータ量を減らすことができます。 <a href="https://docs.databricks.com/delta/optimizations/file-mgmt.html#z-ordering-multi-dimensional-clustering" target="_blank">詳しくはこちらをご覧ください。</a> 郵便番号でZ-Orderしてみましょう。

# COMMAND ----------

spark.sql(f"OPTIMIZE delta.`{delta_partitioned_path}` ZORDER BY (zipcode)")

# COMMAND ----------

# MAGIC %md <i18n value="f13dcb97-e937-4953-8753-ae6e79d50f43"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Feature Storeバッチスコアリング（Feature Store Batch Scoring）

# COMMAND ----------

# MAGIC %md <i18n value="661ad1cf-1389-4903-9d5c-68257d80637c"/>
# MAGIC 
# MAGIC 
# MAGIC Feature tableの作成
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> csvファイルからSparkを使ってDataFrameを読み込み、　Feature storeの **`data source`** を直接追跡します。

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table,FeatureLookup

## create a feature store client
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

## build feature dataframe, add index column and drop label
df = (spark.read
           .csv(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", header=True, inferSchema=True)
           .withColumn("index", monotonically_increasing_id()))

## feature data - all the columns except for the true label
features_df = df.drop("price")

## inference data - contains only index and label columns, if you have online features, it should be added to inference_df as well
inference_df = df.select("index", "price")

# COMMAND ----------

# MAGIC %md <i18n value="b3bbf60b-d884-4e1e-8cdb-a9de86fa2353"/>
# MAGIC 
# MAGIC 一意のテーブル名を付与します。
# MAGIC 
# MAGIC DBR 10.5 以降では、Feature Store テーブルを削除できますが、今のところ、このノートブックを再実行する場合に備えて一意の名前が必要です。

# COMMAND ----------

import uuid

# create unique table name
uid = uuid.uuid4().hex[:6]
feature_table_name = f"{DA.db_name}.airbnb_fs_{uid}"
print(f"Table: {feature_table_name}")

# create feature table
fs.create_table(
    name=feature_table_name,
    primary_keys=["index"],
    df=features_df,
    description="review cols of Airbnb data"
)

# COMMAND ----------

# MAGIC %md <i18n value="6ad326c5-036f-4c5e-950a-5ade154cc398"/>
# MAGIC 
# MAGIC 
# MAGIC Feature storeから、 **`fs.create_training_set`**　を使ってトレーニングセットを作成します。

# COMMAND ----------

## FeatureLoopup object
#feature_lookups = [FeatureLookup(feature_table_name, f, "index") for f in features_df.columns if f!="index"] ## exclude index colum
## uncomment the command below to create lookup features if using Runtime 9.1 ML
feature_lookups = [FeatureLookup(table_name = feature_table_name, feature_names = None, lookup_key = "index") ]

## fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
training_set = fs.create_training_set(inference_df, feature_lookups, label="price", exclude_columns="index")

# COMMAND ----------

# MAGIC %md <i18n value="1cb7cc5b-83cb-4a83-9bc6-027730a7fbfc"/>
# MAGIC 
# MAGIC 
# MAGIC Feature storeでのpackaged modelを記録します。
# MAGIC 一意のモデル名を付与する必要があります。

# COMMAND ----------

uid = uuid.uuid4().hex[:6]
model_name = f"{DA.unique_name}_airbnb-fs-model_{uid}"

print(f"Model Name: {model_name}")

# COMMAND ----------

from mlflow.models.signature import infer_signature
## log RF model as a feature store packaged model and register the packaged model in model registry as `model_name`
fs.log_model(
    model=rf,
    artifact_path="feature_store_model",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name=model_name,
    input_example=X[:5],
    signature=infer_signature(X, y)
)

# COMMAND ----------

# MAGIC %md <i18n value="79acd949-a75b-4b50-962f-49f612e56ce3"/>
# MAGIC 
# MAGIC 
# MAGIC ここで、Model Registry UI上でモデルを表示します。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_MLProd_model_registry.png" alt="step12" width="800"/>
# MAGIC 
# MAGIC </br>
# MAGIC </br>
# MAGIC 
# MAGIC 一方で、登録されたモデルは、Feature storeのUIにも表示されます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_MLProd_feature_store_UI.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md <i18n value="17e87bbe-1fff-4c46-8dcd-b358ad936dd8"/>
# MAGIC 
# MAGIC 
# MAGIC Feature store modelによるバッチ・スコア（Batch score）を行います。

# COMMAND ----------

## for simplicity sake, we will just predict on the same inference_data_df
batch_input_df = inference_df.drop("price") #exclude true label
with_predictions = fs.score_batch(f"models:/{model_name}/1", 
                                  batch_input_df, result_type='double')
display(with_predictions)

# COMMAND ----------

# MAGIC %md <i18n value="82dd28bc-1943-4273-b01d-c7293c6aa60c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー（Review）　
# MAGIC **質問:**　バッチ・デプロイにおけて、考慮する必要な事項はなんでしょうか？  
# MAGIC **答え:**　以下の事項か、推論結果をバッチ処理でデプロイする最適な方法を決定するのに役に立ちます。  
# MAGIC * データがどのようにクエリされるか 
# MAGIC * データがどのように書き込まれるか 
# MAGIC * トレーニングとデプロイ環境 
# MAGIC * 最終モデルがどのようなデータでトレーニングされるか 
# MAGIC 
# MAGIC **質問:**　推論の読み込みと書き込みはどのように最適化することができますか？  
# MAGIC **答え:**　書き込みは並列分散することで最適化することができます。 Sparkでは、これは、DataFrameのパーティションを管理することを意味しており、作業が均等に分散され、書き込み先のデータベースに最も効率的に接続します。　　

# COMMAND ----------

# MAGIC %md <i18n value="9a23fac7-e622-4c09-a389-b74c95f7efaf"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Lab<br>
# MAGIC 
# MAGIC このレッスンのラボを実施します。 [Batch Lab]($./Labs/01-Batch-Lab)

# COMMAND ----------

# MAGIC %md <i18n value="203cc214-4db6-4796-977e-eaf2bf16f33f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他のトピックとリソース（Additional Topics & Resources）
# MAGIC 
# MAGIC **Q:** MLflowを使ってUDFを作るための情報はどこにありますか？　  
# MAGIC **A:** こちら <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html" target="_blank">MLflow documentation for details</a>　をご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
