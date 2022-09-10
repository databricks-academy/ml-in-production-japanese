# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="5be07803-280c-44df-8e24-f546b3204f14"/>
# MAGIC 
# MAGIC 
# MAGIC # ストリーミングのデプロイメント ( Streaming Deployment)
# MAGIC 
# MAGIC バッチ処理に次いで多いのが、SparkのStructured Streamingのような技術を使った継続的なモデル推論です。 このレッスンでは、ストリームデータに対して推論を実行する方法を紹介します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、以下のことを実施します。<br>
# MAGIC  - ストリーミングデータに対する推論を行います。
# MAGIC  - ストリーミングデータに対して、 **`sklearn`** モデルを用いて予測します。
# MAGIC  - 予測結果を常に最新のデルタファイルにストリームします。

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="e572473d-49ff-4ce6-bdb9-5d50b0fad4e5"/>
# MAGIC 
# MAGIC 
# MAGIC Structured StreamsとStructured Streamsの処理方法に関する知識は、このレッスンの前提条件です。

# COMMAND ----------

# MAGIC %md <i18n value="cfc42f33-5726-4b6c-93f4-e2ce7b64f7e1"/>
# MAGIC 
# MAGIC 
# MAGIC ### ストリーミングデータに対する推論 (Inference on Streaming Data)
# MAGIC 
# MAGIC Spark Streamingでできること：<br>
# MAGIC 
# MAGIC * スケーラブルかつ耐障害性な運用により、入力データに対して継続的に推論を行います。
# MAGIC * ストリーミングアプリケーションは、ETLやその他のSparkの機能を組み込んで、リアルタイムにアクションを起こすことも可能です。
# MAGIC 
# MAGIC このレッスンは、本番機械学習ジョブに関連するストリーミングアプリケーションの入門編としています。 
# MAGIC 
# MAGIC ストリーミングには、特別な課題がいくつもあります。具体的には次のようなものがあります。<br>
# MAGIC 
# MAGIC * *エンドツーエンドの信頼性と正確性：* ネットワークの問題、トラフィックの急増、および/またはハードウェアの誤動作によって引き起こされるパイプラインの任意の要素の障害に対する回復力がなければなりません。
# MAGIC * *複雑な変換を処理する：* よく複雑なビジネスロジックかつ形式の異なるデータを入力として処理しなければなりません。
# MAGIC * *データの遅延と到着順序の外れ：* ネットワークの問題により、データの到着が遅れたり、意図した順序から外れたりすることがあります。
# MAGIC * *他のシステムとの統合：* アプリケーションは、データインフラストラクチャの他の部分と統合する必要があります。

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="6002f472-4e82-4d1e-b361-6611117e59dd"/>
# MAGIC 
# MAGIC 
# MAGIC Sparkでデータソースをストリーミングするのに...<br>
# MAGIC 
# MAGIC * バッチデータ処理と同じDataFrames APIを提供します。
# MAGIC * 重要な違いは、構造化ストリーミングの場合、DataFrameが無制限であることです。
# MAGIC * つまり、入力ストリームにデータが入ってきたら、新しいレコードが入力のDataFrameに追加されます。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ETL-Part-3/structured-streamining-model.png" style="height:400px; margin:20px"/></div>
# MAGIC 
# MAGIC Sparkは以下のユースケースに適用します。<br>
# MAGIC 
# MAGIC * バッチ推論
# MAGIC * データの着信ストリーム
# MAGIC 
# MAGIC しかし、低遅延の推論では、レイテンシ要求によりSparkが最適なソリューションになる場合もあれば、そうでない場合もあります。

# COMMAND ----------

# MAGIC %md <i18n value="1fd61935-d780-4322-80e4-cdc43c9ebcf7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### ストリームに接続 (Connecting to the Stream)
# MAGIC 
# MAGIC データ技術の成熟に伴い、業界では一連の技術に収斂されてきています。 Apache Kafkaや、AWS Kinesis、Azure Event Hubsといったクラウド専用の管理型代替手段は、多くのパイプラインの中心にあるデータ収集エンジンとなっています。 
# MAGIC 
# MAGIC この技術は、IoTデバイスなどデータを書き込むプロデューサと、Sparkクラスタなどデータを読み込んでリアルタイムで分析するコンシューマの間でのメッセージブローカーとなります。プロデューサーとコンシューマーの間に多対多の関係も対応可能であり、ブローカー自体がスケーラブルで耐障害性を持ちます。
# MAGIC 
# MAGIC ここでは、 **`maxFilesPerTrigger`** オプションを使用して、ストリームをシミュレートしてみます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> データをストリーミングする方法はいくつかあります。 もう1つの一般的なデザインパターンは、オブジェクトストアからストリームを生成し、新しいファイルが入ってきたらストリームで読み込むというパターンです。

# COMMAND ----------

airbnb_df = spark.read.parquet(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="ed0a634a-3aac-4225-a533-6e508e59e205"/>
# MAGIC 
# MAGIC データストリームのスキーマを作成します。 データストリームには、あらかじめ定義されたスキーマが必要です。

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType, StructType

schema = (StructType()
    .add("host_total_listings_count", DoubleType())
    .add("neighbourhood_cleansed", IntegerType())
    .add("zipcode", IntegerType())
    .add("latitude", DoubleType())
    .add("longitude", DoubleType())
    .add("property_type", IntegerType())
    .add("room_type", IntegerType())
    .add("accommodates", DoubleType())
    .add("bathrooms", DoubleType())
    .add("bedrooms", DoubleType())
    .add("beds", DoubleType())
    .add("bed_type", IntegerType())
    .add("minimum_nights", DoubleType())
    .add("number_of_reviews", DoubleType())
    .add("review_scores_rating", DoubleType())
    .add("review_scores_accuracy", DoubleType())
    .add("review_scores_cleanliness", DoubleType())
    .add("review_scores_checkin", DoubleType())
    .add("review_scores_communication", DoubleType())
    .add("review_scores_location", DoubleType())
    .add("review_scores_value", DoubleType())
    .add("price", DoubleType())
)

# COMMAND ----------

# MAGIC %md <i18n value="5488445f-162c-4e8a-9311-ac0bbfaa8b0e"/>
# MAGIC 
# MAGIC スキーマが一致することを確認します。

# COMMAND ----------

schema == airbnb_df.schema

# COMMAND ----------

# MAGIC %md <i18n value="98b3fcc8-4b65-43fa-9eb0-d3ef38171026"/>
# MAGIC 
# MAGIC 
# MAGIC シャッフルパーティションの数を確認します。

# COMMAND ----------

spark.conf.get("spark.sql.shuffle.partitions")

# COMMAND ----------

# MAGIC %md <i18n value="3545be89-e696-4c69-bf60-463925f861e6"/>
# MAGIC 
# MAGIC 
# MAGIC シャッフルパーティションの数を8に変更します。

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "8")

# COMMAND ----------

# MAGIC %md <i18n value="f1cde533-f5e0-47b0-a4b3-3d38b6d12354"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC  **`readStream`** と **`maxFilesPerTrigger`** を使用してデータストリームを作成します。
# MAGIC  

# COMMAND ----------

streaming_data = (spark
                 .readStream
                 .schema(schema)
                 .option("maxFilesPerTrigger", 1)
                 .parquet(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
                 .drop("price"))

# COMMAND ----------

# MAGIC %md <i18n value="1f097a74-075d-4616-9edb-e6b6f49aa949"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### ストリーミングデータへ`sklearn`モデルの適用
# MAGIC 
# MAGIC DataFrame APIを使用することで、Sparkはバッチデータ処理と同じように、受信データのストリームを処理することができるようになります。 

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run(run_name="Final RF Model") as run: 
    df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
    X = df.drop(["price"], axis=1)
    y = df["price"]

    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf.fit(X, y)

    mlflow.sklearn.log_model(rf, "random-forest-model")

# COMMAND ----------

# MAGIC %md <i18n value="9630f176-9589-4ab2-b9d2-98fe34676250"/>
# MAGIC 
# MAGIC Sparkで適用できるようにほど **`sklearn`** で学習したモデルからUDFを作成します。

# COMMAND ----------

import mlflow.pyfunc

pyfunc_udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run.info.run_id}/random-forest-model")

# COMMAND ----------

# MAGIC %md <i18n value="3e451958-fc3f-47a8-9c1d-adaff3fbc9b3"/>
# MAGIC 
# MAGIC ストリーム処理の前に、ストリーム名を作成します。

# COMMAND ----------

my_stream_name = "lesson03_stream"

# COMMAND ----------

# MAGIC %md <i18n value="6b46b356-b828-4032-b2f8-c7a60e7f70d6"/>
# MAGIC 
# MAGIC 次に、ストリームが実際に処理の「準備」が整うまでブロックするユーティリティ・メソッドを作成します。

# COMMAND ----------

import time

def until_stream_is_ready(name, progressions=3):  
    # Get the query identified by "name"
    queries = list(filter(lambda query: query.name == name, spark.streams.active))

    # We need the query to exist, and progress to be >= "progressions"
    while (len(queries) == 0 or len(queries[0].recentProgress) < progressions):
        time.sleep(5) # Give it a couple of seconds
        queries = list(filter(lambda query: query.name == name, spark.streams.active))

    print(f"The stream {name} is active and ready.")

# COMMAND ----------

# MAGIC %md <i18n value="6ff16efc-b40f-4386-b000-2f40edc19d2b"/>
# MAGIC 
# MAGIC 
# MAGIC これで、ストリームを予測で変換し、 **`display()`** コマンドで結果をプレビューできるようになります。

# COMMAND ----------

predictions_df = streaming_data.withColumn("prediction", pyfunc_udf(*streaming_data.columns))

display(predictions_df, streamName=my_stream_name)

# COMMAND ----------

until_stream_is_ready(my_stream_name)

# COMMAND ----------

# When you are done previewing the results, stop the stream.
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md <i18n value="0d285926-c3a5-40d5-a36f-c4963ab81f71"/>
# MAGIC 
# MAGIC 
# MAGIC ### デルタへストリーミング予測の書き出し (Write out Streaming Predictions to Delta)
# MAGIC 
# MAGIC 予測結果をストリーミングデータフレームとしてFeature Storeのテーブルに書き出すこともできます（一意のIDが必要）。

# COMMAND ----------

checkpoint_location = f"{DA.paths.working_dir}/stream.checkpoint"
write_path = f"{DA.paths.working_dir}/predictions"

(predictions_df
    .writeStream                                           # Write the stream
    .queryName(my_stream_name)                             # Name the query
    .format("delta")                                       # Use the delta format
    .partitionBy("zipcode")                                # Specify a feature to partition on
    .option("checkpointLocation", checkpoint_location)     # Specify where to log metadata
    .option("path", write_path)                            # Specify the output path
    .outputMode("append")                                  # Append new records to the output path
    .start()                                               # Start the operation
)

# COMMAND ----------

until_stream_is_ready(my_stream_name)

# COMMAND ----------

# MAGIC %md <i18n value="3657d466-7062-4c34-beaa-4452d19fe1af"/>
# MAGIC 
# MAGIC 実際のファイルを見てみましょう。 
# MAGIC 
# MAGIC これを数回リフレッシュして変化に注目してください。

# COMMAND ----------

spark.read.format("delta").load(write_path).count()

# COMMAND ----------

# When you are done previewing the results, stop the stream.
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md <i18n value="d297c548-624f-4ea6-b4d1-7934f9abc58e"/>
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC 
# MAGIC **質問:** 一般的なデータストリームは何でしょうか？ <br>
# MAGIC **答え：** Apache Kafkaや、AWS KinesisやAzure Event Hubsなどのクラウド管理型ソリューションが一般的なデータストリームです。 さらに、受信ファイルが入ってくるディレクトリを監視するのもよくあります。 新しいファイルが入ると、そのファイルをストリームに取り込んで処理します。
# MAGIC 
# MAGIC **質問:** SparkはどのようにExactly Once（厳密に1回）」のデータ配信を行い、ストリームのメタデータを維持するのでしょうか？ <br>
# MAGIC **答え:** チェックポイントは、クラスターの状態を維持する能力を通じ、耐障害性を実現します。
# MAGIC 
# MAGIC **質問:** Sparkのストリーミングは、他のSparkの機能とどのように統合されているのですか？ <br>
# MAGIC **答え:** Spark Streamingはバッチ処理と同じDataFrame APIを使用しているため、他のSpark機能との統合が容易に行えます。

# COMMAND ----------

# MAGIC %md <i18n value="0be2552c-9387-4d52-858e-b564c157c978"/>
# MAGIC 
# MAGIC 
# MAGIC ## その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q:** StreamingとKafkaの統合に関する詳しい情報はどこで得られますか？ <br>
# MAGIC **A:** <a href="https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html" target="_blank">Structured Streaming + Kafka Integration Guide</a>を確認してください。
# MAGIC 
# MAGIC **Q:** Spark 3.1のStructured Streamingの新機能を教えてください。 <br>
# MAGIC **A:** Databricksのブログ記事<a href="https://databricks.com/blog/2021/04/27/whats-new-in-apache-spark-3-1-release-for-structured-streaming.html" target="_blank">What’s New in Apache Spark™ 3.1 Release for Structured Streaming</a>をチェックしてみてください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
