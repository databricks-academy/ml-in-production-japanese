# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="eb86cd87-1efb-41e7-ab51-42e2519a7f7f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Feature Store
# MAGIC 
# MAGIC 本番機械学習ソリューションは、再現可能なデータ管理から始まります。このノートブックでは、<a href="https://docs.databricks.com/delta/versioning.html" target="_blank">Delta Tableのバージョン管理</a>と<a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target= "_blank">Databricks Feature Store</a>の利用をデータ管理の戦略としてカバーします。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC  - Deltaでテーブルのバージョンを管理する
# MAGIC  - プログラム的にFeature Tableのログに記録する

# COMMAND ----------

# MAGIC %md <i18n value="cdce9fac-cdce-4ab5-8fa3-2797972d57b6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## データ管理と再現性 (Data Management and Reproducibility)
# MAGIC 
# MAGIC 機械学習のライフサイクルを管理するということは...<br>
# MAGIC 
# MAGIC * データの再現性
# MAGIC * コードの再現性
# MAGIC * モデルの再現性
# MAGIC * 本番システムとの自動統合
# MAGIC 
# MAGIC **このコースではデータ管理から始めます。**これは次のようなさまざまな方法で実現できます。<br>
# MAGIC 
# MAGIC - データのスナップショットを保存する
# MAGIC - Deltaでテーブルのバージョン管理とTime Travelを行う
# MAGIC - Feature tableを使う

# COMMAND ----------

# MAGIC %md <i18n value="7f8a9e4b-a821-46cb-9fcd-fc58572769ce"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC 各レッスンを正しく実行するために、各レッスンの開始時に **`Classroom-Setup`** セルを実行してください。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="a1ba0678-b162-4513-a3af-675a169df7a6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC データを読み込んで、部件(リスティング)ごとに一意のIDを生成しましょう。

# COMMAND ----------

path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings.csv"
airbnb_df = spark.read.csv(path, header="true", inferSchema="true", multiLine="true", escape='"')

display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="3167a48a-b7d4-42cf-9197-1c6cd66c87bf"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Delta Tableのバージョン管理 (Versioning with Delta Tables)
# MAGIC 
# MAGIC 新しいDelta tableにデータを書き込むことから始めましょう。

# COMMAND ----------

(airbnb_df.write
          .format("delta")
          .save(DA.paths.airbnb))

# COMMAND ----------

# MAGIC %md <i18n value="2d9d25d2-df88-44a8-93eb-cfebfd8ca022"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC それでは、Delta tableを読み込んで編集し、 **`cancellation_policy`** と **`instant_bookable`** 列を削除しましょう。

# COMMAND ----------

delta_df = (spark.read
                 .format("delta")
                 .load(DA.paths.airbnb)
                 .drop("cancellation_policy", "instant_bookable"))
display(delta_df)

# COMMAND ----------

# MAGIC %md <i18n value="7ac1e0ec-785e-48f1-94f0-86b217432e69"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これで、 **`mode`** パラメータを使ってDeltaテーブルを **`overwrite`** できます。

# COMMAND ----------

(delta_df.write
         .format("delta")
         .mode("overwrite")
         .save(DA.paths.airbnb))

# COMMAND ----------

# MAGIC %md <i18n value="1fe2a536-7fc8-49aa-a7b2-e8322d81a86f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC おっと！実は **`cancellation_policy`** 列を残したかったのです。幸いなことに、データバージョニングを使用して、このテーブルの古いバージョンに戻すことができます。
# MAGIC 
# MAGIC まず、 **`DESCRIBE HISTORY`** SQL コマンドを使用します。

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY delta.`${DA.paths.airbnb}`

# COMMAND ----------

# MAGIC %md <i18n value="62f4bb05-4074-49e0-ac00-661940133a35"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC バージョン1の **`operationParameters`** 列に書いてあるように、テーブルを上書きしました。すべての元の列を取得するには、バージョン0をロードするために時間を遡る必要があります。その後、 **`instant_bookable`** 列のみを削除することができます。

# COMMAND ----------

delta_df = (spark.read
                 .format("delta")
                 .option("versionAsOf", 0)
                 .load(DA.paths.airbnb))
display(delta_df)

# COMMAND ----------

# MAGIC %md <i18n value="33bda953-e688-4729-a476-a3d46a2662a2"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC timestampを使って旧バージョンに戻すこともできます。 
# MAGIC 
# MAGIC **<a href="https://docs.databricks.com/delta/delta-batch.html#deltatimetravel" target="_blank">VACUUMコマンド</a>を実行すると、テーブルの古いスナップショットをクエリする機能 (time travel)が失効することに注意してください。**

# COMMAND ----------

timestamp = spark.sql(f"DESCRIBE HISTORY delta.`{DA.paths.airbnb}`").orderBy("version").first().timestamp

display(spark.read
             .format("delta")
             .option("timestampAsOf", timestamp)
             .load(DA.paths.airbnb))

# COMMAND ----------

# MAGIC %md <i18n value="82162788-2808-450a-95da-7b1387270814"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これで、 **`instant_bookable`** を削除してテーブルをoverwriteできます。

# COMMAND ----------

(delta_df.drop("instant_bookable")
         .write
         .format("delta")
         .mode("overwrite")
         .save(DA.paths.airbnb))

# COMMAND ----------

# MAGIC %md <i18n value="b34fd627-6426-42c7-b0d9-82e46fcd183e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC バージョン2は、最新かつ最も正確なテーブルのバージョンです。

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY delta.`${DA.paths.airbnb}`

# COMMAND ----------

# MAGIC %md <i18n value="ff976181-9c5e-4a87-8a2f-af9886791192"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Feature Store
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#databricks-feature-store" target="_blank">Feature Store</a>は、 **企業における全ての特徴量を管理する中央管理型リポジトリ** になります。これにより、組織内で **特徴量の共有と探索** が可能になり、モデルのトレーニングと推論をするために、特徴量を算出するための同じコードが使用されていることが確保できます。

# COMMAND ----------

import uuid

uid = uuid.uuid4().hex[:6]
table_name = f"{DA.db_name}.airbnb_{uid}"

print(f"Table: {table_name}")

# COMMAND ----------

# MAGIC %md <i18n value="bf78552e-18c0-4eda-8dcf-ed76bf38c998"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Feature Storeにデータを入力できるように、<a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store" target="_blank">Feature Store Client</a>を作成しましょう。

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

help(fs.create_table)

# COMMAND ----------

# MAGIC %md <i18n value="52d53b6d-b1c0-4f8e-bb0c-72df4ad7b71f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Feature Tableの作成 (Create Feature Table)
# MAGIC 
# MAGIC 次に、 **`create_table`** メソッドを使用してFeature Tableを作成します。
# MAGIC 
# MAGIC このメソッドは、いくつかのパラメーターをインプットとして受け取ります。
# MAGIC * **`name`** - **`<database_name>.<table_name>`** という形式のFeature table名
# MAGIC * **`primary_keys`** - 主キー。複数の列が必要な場合は、列名のリストを指定します
# MAGIC * **`df`** - このFeature tableに挿入するデータ。 **`features_df`** のスキーマがfeature tableのスキーマとして使用されます
# MAGIC * **`schema`** - Feature tableのスキーマ。 **`schema`** または **`features_df`** のどちらかが提供されなければならないことに注意してください
# MAGIC * **`description`** - Feature tableの説明
# MAGIC * **`partition_columns`**  - Feature tableをパーティション分割するために使用する列

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["id"],
    df=airbnb_df,
    partition_columns=["neighbourhood"],
    description="Original Airbnb data"
)

# COMMAND ----------

# MAGIC %md <i18n value="b5d563a7-36e5-4757-a54c-d622b4198ebc"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC あるいは、スキーマのみで ( **`df`** の指定なしで) **`create_table`** を実行し、 **`fs.write_table`** でfeature tableにデータを入力することもできます。 **`fs.write_table`** には **`overwrite`** と **`merge`** の2つのモードがあります。
# MAGIC 
# MAGIC 例:
# MAGIC 
# MAGIC ```
# MAGIC fs.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["index"],
# MAGIC     schema=airbnb_df.schema,
# MAGIC     description="Original Airbnb data"
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(
# MAGIC     name=table_name,
# MAGIC     df=airbnb_df,
# MAGIC     mode="overwrite"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md <i18n value="ff05da34-eb42-49e7-9a0e-8a6a03a37b7a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC それでは、UIで作成したテーブルがどのように追跡されるかを見てみましょう。最初に [Machine Learning] ワークスペースにいることを確認してから、ナビゲーションバーの左下にある [Feature Store] アイコンをクリックして、UIに移動します。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Nav.png" alt="step12" width="150"/>

# COMMAND ----------

# MAGIC %md <i18n value="de8630f3-d645-42c7-b966-6c2420854d46"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC このスクリーンショットでは、上記作成したFeature tableが表示されています。
# MAGIC <br>
# MAGIC <br>
# MAGIC  **`Producers`** のセクションに注目してください。このセクションは、Feature tableがどのノートブックから生成されたのかを示してあります。
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_details+(1).png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="744b350a-8786-4f6d-8997-fa9e967a0c2e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC また、FeatureStore clientの **`get_table()`** を使用してFeature Storeのメタデータを見ることもできます。

# COMMAND ----------

print(f"Feature table description : {fs.get_table(table_name).description}")
print(f"Feature table data source : {fs.get_table(table_name).path_data_sources}")

# COMMAND ----------

# MAGIC %md <i18n value="31c92288-4164-46a0-9763-7491e87eda53"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 各部件の平均レビュースコアを作成し、レビュー列の数を減らしましょう。

# COMMAND ----------

from pyspark.sql.functions import lit, expr

review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                  "review_scores_communication", "review_scores_location", "review_scores_value"]

airbnb_df_short_reviews = (airbnb_df
                           .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
                           .drop(*review_columns)
                          )

display(airbnb_df_short_reviews)

# COMMAND ----------

# MAGIC %md <i18n value="80156815-35e5-4f33-a4a1-b3c9d9390a39"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Overwrite（上書き）機能 (Overwrite Features)
# MAGIC 
# MAGIC モードを **`overwrite`** に設定し、dfから削除した特徴量列を最新のテーブルから削除します。

# COMMAND ----------

fs.write_table(name=table_name,
               df=airbnb_df_short_reviews,
               mode="overwrite")

# COMMAND ----------

# MAGIC %md <i18n value="61d7f79a-e7a9-4072-a694-587352484c5c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC UIに戻ると、変更日が変更され、新しい列がFeatureのリストに追加されたことが確認できます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png">削除された列はテーブルのスキーマにまだ存在してあるが、その値は全て **`null`** に置き換えられています。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_New_Feature.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md <i18n value="35489dba-e6a9-4bd0-913b-966e4613af3d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Feature Storeからデータを読み込みましょう。オプションとして、バージョンまたはタイムスタンプを指定し、Delta Time Travelを使用してFeature tableのスナップショットから読み取ることもできます。

# COMMAND ----------

# Display most recent table
feature_df = fs.read_table(name=table_name)
display(feature_df)

# COMMAND ----------

# MAGIC %md <i18n value="032776de-34bf-492c-857b-88caf2e3d34a"/>
# MAGIC 
# MAGIC Featureテーブルを削除しましょう。
# MAGIC 
# MAGIC Feature Storeの [`drop_table` API](https://opdhsblobprod04.blob.core.windows.net/contents/8dfe1f3273d94cadb148335e357e0036/5c18c2bcd4692b02fa58c32e632a169c?skoid=29100048-1fa1-4ada-b0e0-e2aa294fc66a&sktid=975f013f-7f24-47e8-a7d3-abc4752bf346&skt=2022-08-18T07%3A23%3A51Z&ske=2022-08-25T07%3A28%3A51Z&sks=b&skv=2020-10-02&sv=2020-08-04&se=2022-08-19T17%3A55%3A31Z&sr=b&sp=r&sig=VRno3dn0owAulPdQBuwhQmahDX8J785PBo%2BVxLhm9rY%3D)を利用してDBR 10.5+ MLのFeatuerテーブルを削除できます。 ただし、裏のDeltaテーブルも削除されます。そのため、裏のDeltaテーブルを削除したくない場合は、Feature Store UIの削除ボタンを利用ください。
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_delete.png">
# MAGIC 
# MAGIC 「削除」ボタンをクリックするとポップアップ画面が表示されます。「削除」ボタンをクリックして削除を確認します。
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_delete_window.png" width=600>

# COMMAND ----------

# MAGIC %md <i18n value="ba7fb23d-7ee0-453a-8bdc-fdf9e6c63dc4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Feature（特徴量）をリアルタイムサービングに使用する必要の場合は、<a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store" target="_blank">online store</a>にFeatureをpublishすることができます</a>。
# MAGIC 
# MAGIC また、Feature tableに対するアクセス権限を制御することもできます。テーブルを削除するには、UIの**`delete`** ボタンを使用します。 **データベースからDeltaテーブルを削除することも必要です。** 
# MAGIC 
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_permissions+(1).png" alt="step12" width="700"/>

# COMMAND ----------

# MAGIC %md <i18n value="b4480d8a-2aed-43dd-b52a-7e0c758afea4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC **質問:** なぜデータ管理が重要ですか?  
# MAGIC **答え:** データ管理は、エンドツーエンドMLの再現性において見過ごされがちな重要なことです。
# MAGIC 
# MAGIC **質問:** Delta tableでデータのバージョンをどのように管理するのですか?  
# MAGIC **答え:** Delta tableは、新しいデータが書き込まれるたびに自動的にバージョン管理されます。旧バージョンのテーブルへのアクセスは、 **`display(spark.sql(f"DESCRIBE HISTORY delta.{delta_path}"))`** を使って戻してほしいバージョンを見つけ、そのバージョンのデータをロードするぐらいに簡単にできます。タイムスタンプを使用して以前のバージョンに戻すこともできます。
# MAGIC 
# MAGIC **質問:** Feature Storeはどのような課題を解決するのに役立ちますか?  
# MAGIC **答え:** MLパイプラインの開発によく直面する重要な課題は、特徴量の再現性とデータの共有です。Feature Storeを利用することで、組織内のユーザーが同じ特徴量計算コードを使用できるようになります。
# MAGIC 
# MAGIC **質問:** データセットのハッシュ化は何に役立ちますか?  
# MAGIC **答え:** 2つのデータセットが同じかどうかを確認するのに役立ちます。これはデータの再現性の確認に役立ちます。ただし、2つのデータセットの完全な差分を示すことはできず、スケーラブルなソリューションでもありません。

# COMMAND ----------

# MAGIC %md <i18n value="09538a0b-08b7-4ace-8781-1b6e68bdd789"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ
# MAGIC 
# MAGIC 次のレッスンに進みます。 [実験の追跡]($./02-Experiment-Tracking)

# COMMAND ----------

# MAGIC %md <i18n value="ec52d384-7d1e-4fc3-a1db-50bb85224502"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 補足：その他トピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q:**  Delta tableの詳細を知りたいですが、どこを見ればいいですか？
# MAGIC **A:**  Himanshu Rajによるこの<a href="https://databricks.com/session_na21/intro-to-delta-lake" target="_blank">講演</a>をチェックしてください。
# MAGIC 
# MAGIC **Q:**  Feature Storeの詳細を知りたいですが、どこを見ればいいですか？
# MAGIC **A:**  <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">このドキュメント</a>では、Feature Storeを利用してパイプラインに対して何ができるかを詳しく説明しています。
# MAGIC 
# MAGIC **Q:**  再現性とその重要性についてもっと知りたいですが、どこを見ればいいですか？
# MAGIC **A:**  Mary Grace MoestaとSrijith Rajamohanによる<a href="https://databricks.com/blog/2021/04/26/reproduce-anything-machine-learning-meets-data-lakehouse.html" target="_blank">このブログ記事</a>は、再現可能なデータとモデルを作成するための基礎知識を提供します。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
