# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c83d51d6-428b-4691-82ea-778976cde46b"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # 本番における機械学習 (Machine Learning in Production)
# MAGIC ### MLflow、デプロイ、CI/CDで機械学習のライフサイクル全体管理###
# MAGIC ### (Managing the Complete Machine Learning Lifecycle with MLflow, Deployment and CI/CD)###
# MAGIC 
# MAGIC このコースでは、機械学習エンジニア、データエンジニア及びデータサイエンティストが、実験やモデルの管理からさまざまなデプロイ方式と本番環境に関連する様々な課題対応まで、機械学習のライフサイクル全体管理のベストプラクティスを学びます。MLflowを使用してデータ管理、実験追跡、モデル管理などを含むエンドツーエンドでの機械学習モデルの再現性から始め、バッチ、ストリーミング、リアルタイムでモデルのデプロイ、そして監視、アラート、CI/CDへの対応対策を学びます。サンプルコードはすべてのモジュールに付属しています。
# MAGIC 
# MAGIC このコースでは、
# MAGIC - まず、データ、モデル、実験の追跡などを含むエンドツーエンドでの再現性に焦点を当て、MLflowを使用した実験プロセスの管理について説明します。
# MAGIC - 次に、MLflowモデルレジストリへのモデルの登録、artifactsと環境の管理、モデルテストの自動化など、さまざまなダウンストリームのデプロイツールと統合することにより、モデルの運用を学びます。
# MAGIC - その次に、バッチ、ストリーミング、およびリアルタイム推論のためのモデルデプロイを実装します。
# MAGIC - 最後に、継続的なインテグレーション、継続的なデプロイメント、モニタリングおよびアラートを含めた本番環境で必要な取組もカバーします。
# MAGIC 
# MAGIC このコースを修了すると、機械学習モデルのログ記録、デプロイ、監視を含めたエンドツーエンドのパイプラインの構築が可能になります。
# MAGIC 
# MAGIC このコースは全てPythonで教えられています。
# MAGIC 
# MAGIC ## コーススケジュール (Lessons)
# MAGIC 
# MAGIC | 所要時間 &nbsp;| レッスン名 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | 説明 &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 30m  | **コースのご紹介 ＆ 環境設定**                               | *自己紹介、コース内容の概要紹介 ＆ Q&A* |
# MAGIC | 30m    | **本番における機械学習の全体像**    | エンドツーエンド機械学習のフルライフサイクルの紹介 |
# MAGIC | 10m  | **休憩**                                               ||
# MAGIC | 20m    | **[実験管理 - Feature Store]($./01-Experimentation)**    | [DeltaとDatabricks Feature Storeでのデータ管理]($./01-Experimentation/01-Feature-Store) |
# MAGIC | 40m  | **[実験管理 - 実験トラッキング ＆ ラボ]($./01-Experimentation)** | [MLflowでの機械学習（ML）実験の追跡]($./01-Experimentation/02-Experiment-Tracking) </br> [実験トラッキング ラボ]($./01-Experimentation/Labs/02-Experiment-Tracking-Lab) | 
# MAGIC | 10m  | **休憩**                                               ||
# MAGIC | 30m  | **[実験管理 - 高度な実験トラッキング & ラボ]($./01-Experimentation)** | [高度な実験トラッキング]($./01-Experimentation/03-Advanced-Experiment-Tracking) </br> [高度な実験トラッキング Lab (オプション)]($./01-Experimentation/Labs/03-Advanced-Experiment-Tracking-Lab) | 
# MAGIC | 30m    | **[モデル管理 - MLflow モデル & ラボ]($./02-Model-Management)**    | [MLflowでのモデル管理]($./02-Model-Management/01-Model-Management) </br> [モデル管理 lab]($./02-Model-Management/Labs/01-Model-Management-Lab) |
# MAGIC | 10m  | **休憩**                                               ||
# MAGIC | 35m  | **[モデル管理 - モデルレジストリ]($./02-Model-Management)**       | [MLflowでのモデル・バージョン管理、モデルのデプロイ]($./02-Model-Management/02-Model-Registry) |
# MAGIC | 25m  | **[モデル管理 - Webhooks]($./02-Model-Management)**      | [登録したモデルのテストジョブとWebhookの作成]($./02-Model-Management/03a-Webhooks-and-Testing) </br> [テストの自動化]($./02-Model-Management/03b-Webhooks-Job-Demo)|
# MAGIC | 10m  | **休憩**                                               ||
# MAGIC | 60m |**[モデルデプロイのパラダイム]($./03-Deployment-Paradigms)** | [バッチ推論]($./03-Deployment-Paradigms/01-Batch)</br> [リアルタイム推論]($./03-Deployment-Paradigms/02-Real-Time)</br> [ストリーミング推論 (Reference)]($./Reference/03-Streaming-Deployment)</br> [ラボ]($./03-Deployment-Paradigms/Labs)|
# MAGIC | 10m  | **休憩**                                               ||
# MAGIC | 60m  | **[アプリケーションの本番稼働]($./04-Production)**  | [監視]($./04-Production/01-Monitoring)</br> [監視ラボ]($./04-Production/Labs/01-Monitoring-Lab)</br>[アラート (Reference)]($./Reference/02-Alerting) </br>[パイプラインの例 (Reference)]($./Reference/04-Pipeline-Example/00-Orchestrate)|
# MAGIC 
# MAGIC 
# MAGIC ## 前提条件 (Prerequisites)
# MAGIC - Python経験 (**`pandas`**, **`sklearn`**, **`numpy`**)
# MAGIC - 機械学習とデータサイエンティストの基礎知識
# MAGIC 
# MAGIC ## Cluster設定 (Cluster Requirements)
# MAGIC - 講師にお問合せください。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_warn_24.png"/> **ノートブックAPIやモデルレジストリなど、このコースで利用されるいくつの機能は、Databricksの有料または試用版のサブスクリプションユーザーのみが利用できます。**
# MAGIC Databricks Community Editionを使用している場合は、ランディングページの **`Upgrade`** ボタンをクリックするか、<a href="https://accounts.cloud.databricks.com/registration.html#login" target="_blank">ここに移動して</a> 無料トライアルを開始してください。

# COMMAND ----------

# MAGIC %md <i18n value="35c71f4c-1ab2-4d02-a07f-c144d7fe7dfa"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC 各レッスンを正しく実行するには、各レッスンの開始時に必ず **`Classroom-Setup`** セルを実行してください（次のセルを参照）。

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="1464bb0e-c32c-4d92-b8a8-7d2e7767205f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### アジャイル・データサイエンス (Agile Data Science)
# MAGIC 
# MAGIC 機械学習モデルを本番環境にデプロイするには、データサイエンティストが最初にモデルを構築するときと状況が異なり、新たな複雑な問題が多く発生します。
# MAGIC 
# MAGIC 多くのチームでは、モノリシック型にカスタマイズした社内ソリューションを利用してこれらの課題を解決していますが、この場合 脆弱性が発生する可能性が高くかつ時間がかかるなどモデルの保守が難しくなります。
# MAGIC 
# MAGIC 機械学習モデルをデプロイするための体系的なアプローチにより、開発時間を最小限に抑え、データサイエンスから得られるビジネス価値を最大化するアジャイルソリューションを実現することが重要です。これを実現するには、データサイエンティストとデータエンジニアは、様々なデプロイ・ツールを利用するだけではなく、モデルを本番環境に移行した後に監視およびアラートを行うためのシステムの導入も必要です。
# MAGIC 
# MAGIC 主なデプロイメントのパラダイムは次のとおりです。<br>
# MAGIC 
# MAGIC 1. **バッチ推論:** 例えば、Webアプリケーションの中でリアルタイムで紹介可能なデータベースのようなものに、予測結果算出し後工程で使用するために保存しておくことを意味します。
# MAGIC 2. **ストリーミング推論:** データ・ストリームは、データがデータパイプラインに到着した後にすぐには変換されませんが、予測が必要な時に変換することを意味します。
# MAGIC 3. **リアルタイム推論:** 通常はモデルを提供するRESTエンドポイントに実装され、推論処理は、即座に低レイテンシーで実行されることを意味します。
# MAGIC 4. **モバイル/IoTデバイスに組込み:** 機械学習ソリューションを、モバイルまたはIoTデバイスに組み込むことを意味し、このコースの範囲外です。
# MAGIC <br>
# MAGIC 
# MAGIC モデルがこれらのパラダイムのいずれかにデプロイされたら、予測結果の品質、レイテンシー、スループット、およびその他の本番環境に関する考慮事項に対する全体のパフォーマンスの監視が必要となります。パフォーマンスが低下し始めると、これはモデルを再学習する必要があるか、モデルのサービングにより多くのリソースを割り当てる必要があるか、またはいくつかの改善が必要であることのサインと考えられます。これらのパフォーマンスの問題を把握するには、アラートシステムを整備する必要があります。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
