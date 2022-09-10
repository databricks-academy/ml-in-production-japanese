# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="61413ef3-42e5-4aa9-85c4-8c1d38cc46b5"/>
# MAGIC 
# MAGIC 
# MAGIC # モデルCI/CD(Model CI/CD)
# MAGIC 
# MAGIC 機械学習モデルのデプロイは挑戦的ですが、それは戦いの半分に過ぎません。機械学習エンジニアは、ソフトウェア開発のプロセスを機械学習の開発プロセスの一部として取り入れることができます。 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、以下を実施します。<br>
# MAGIC  - 継続的インテグレーション(統合)と継続的デプロイの定義
# MAGIC  - CI/CDと機械学習モデルの関連付け
# MAGIC  - 機械学習のためのCI/CDワークフローの利点
# MAGIC  - CI/CDワークフローで使用されるツール
# MAGIC  - 機械学習のためのCI/CDに役立つ学習リソース

# COMMAND ----------

# MAGIC %md <i18n value="04810706-c6b1-4e15-aa4e-f879e3c4144d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 継続的インテグレーションと継続的デプロイメント (Continuous Integration and Continuous Deployment)
# MAGIC 
# MAGIC 継続的インテグレーションと継続的デプロイは、一般的に**CI/CD**と呼ばれ、開発と運用の間のギャップを埋めるものです。
# MAGIC 
# MAGIC #### インテグレーション (Integration)
# MAGIC 
# MAGIC 一般的に、**インテグレーション**とは、次のようなものを指します。<br>
# MAGIC 
# MAGIC * 更新されたコードやその他のアーティファクト(作成物)をセントラル・リポジトリにプッシュします。
# MAGIC * これらのコードやアーティファクトに対する自動テストを実行します。
# MAGIC * テストが合格したら、コードやアーティファクトをシステムに統合することができます。
# MAGIC 
# MAGIC #### デリバリーとデプロイ
# MAGIC 
# MAGIC コードやアーティファクトが統合されたら、更新されたソフトウェアやアーティファクトは、**デリバリー**または**デプロイ**される必要があります。
# MAGIC 
# MAGIC * **デリバリー**：開発者の承認を得た前提でソフトウェア・アプリケーションが自動的にリリースされるプロセス
# MAGIC * **デプロイ**：完全に自動化されたバージョンのデリバリー
# MAGIC 
# MAGIC #### 継続的 (Continuous)
# MAGIC 
# MAGIC これらはなぜ**継続的**なのでしょうか？一般的に「継続的」とは、ソフトウェアの統合やデプロイを間断なく、効率よく、頻繁に行うことを意味します。

# COMMAND ----------

# MAGIC %md <i18n value="1b08a7f0-d9bb-4633-ae3d-e3a54903558f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 機械学習のためのCI/CD (CI/CD for Machine Learning)
# MAGIC 
# MAGIC CI/CDは様々なソフトウェア開発で一般的ですが、**機械学習エンジニアリング**にも欠かせません。
# MAGIC 
# MAGIC #### 機械学習のための統合 (Integration for Machine Learning)
# MAGIC 
# MAGIC 機械学習の統合にはプロジェクトのコードベースの更新が含まれますが、通常は更新されたアーティファクト(作成物)の統合を指します。具体的には、更新された**モデル**を統合したいのです。
# MAGIC 
# MAGIC このプロセスは、通常、次のようなワークフローで行われます。<br>
# MAGIC 
# MAGIC * 既存ソリューションの新モデルを構築します。
# MAGIC * モデルを集中管理しているモデル・リポジトリのテスト・ステージにプッシュします。
# MAGIC * モデルに対して、一連のテスト（ユニット、統合、回帰、パフォーマンスなど）を行います。
# MAGIC * モデル・リポジトリを利用して、モデルを本番機械学習システムに移行します。
# MAGIC 
# MAGIC #### 機械学習のためのデプロイメント (Deployment for Machine Learning)
# MAGIC 
# MAGIC このコースを通して、機械学習の様々なデプロイ方法について説明してきました。**バッチ**、継続的な**ストリーム**、または**リアルタイム**のいずれの方法でデプロイしたとしても、継続的なデプロイは、**エンドユーザーに提供するモデルや予測機能を頻繁に更新することになります**。
# MAGIC 
# MAGIC しかし、機械学習のためにCI/CDを使用する場合、デプロイした時点ではまだ終わりではありません。
# MAGIC 
# MAGIC * モデルの性能が時間とともに低下していないことを確認するために、モデルを継続的に評価する必要があります。
# MAGIC * 性能が低下することを「ドリフト」と呼びます。
# MAGIC * ドリフトが発生する理由やドリフトを検出する方法は様々ですが、次のレッスンで詳しく説明します。

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="91bbbcf5-dbc3-43d7-b309-a5aec7c1c22c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/Model-Staleness.png" style="height: 450px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="69237e1c-636c-4faa-8901-64eeb1240c2a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 機械学習におけるCI/CDのメリット (Benefits of CI/CD in Machine Learning)
# MAGIC 
# MAGIC 機械学習でCI/CDを実践すると、**完全に自動化された開発とデプロイのループを閉じる**ことができ、様々な利点が得られます。
# MAGIC 
# MAGIC #### 時間短縮 (Time Savings)
# MAGIC 
# MAGIC CI/CDがなければ、本番の機械学習アプリケーションの更新は、**時間がかかり、人に依存したプロセス**になります。<br>
# MAGIC 
# MAGIC 1. データサイエンティストが新しいモデルを構築することを決定します。
# MAGIC 2. データサイエンティストが手作業で新しいモデルを構築します。
# MAGIC 3. データサイエンティストによるモデルの性能評価を行います。
# MAGIC 4. データサイエンティストが新しいモデルを本番に稼動させるべきと判断します。
# MAGIC 5. データサイエンティストは、機械学習エンジニアに新しいモデルを本番稼動させるべきと伝えます。
# MAGIC 6. 機械学習エンジニアが手作業でモデルをテスト環境に移動させます。
# MAGIC 7. 機械学習エンジニアがテスト環境内にモデルをデプロイします。
# MAGIC 8. 機械学習エンジニアは、テスト環境内でモデルに対して一連の手動テストを実行します。
# MAGIC 9. 機械学習エンジニアが手動でモデルを本番環境に移行させます。
# MAGIC 10. 機械学習エンジニアが本番環境にモデルを展開します。
# MAGIC 11. データサイエンティストは、モデルのパフォーマンスを繰り返しテストし、アップデートが必要なタイミングを判断します。
# MAGIC 
# MAGIC これには、工数がかかります。CI/CDプロセスに従うことで、この時間のかかるワークフローを自動化することができます。これにより、より速いアップデートサイクルができるようになります。
# MAGIC 
# MAGIC * もっと多い最新モデル
# MAGIC * モデル不良などの不具合による悪影響を抑えることができます。
# MAGIC 
# MAGIC #### 一貫性 (Consistency)
# MAGIC 
# MAGIC 自動化された統合とデプロイに従うことで、プロセスの決定ポイントは、**一貫性**と**再現性**になります。これはつまり、<br>
# MAGIC 
# MAGIC * 各モデルは同じターゲットで作られています。
# MAGIC * 各モデルに全く同じテストを実施します。
# MAGIC * 各モデルは、ステージング環境と本番環境に同じように統合されます。
# MAGIC * 各モデルは全く同じ方法でデプロイされます。
# MAGIC * 各モデルは、ドリフトを検出するために同じ基準で継続的に評価されます。
# MAGIC 
# MAGIC これにより、担当のデータサイエンティストと機械学習エンジニアが異なる（または同じ担当者で工数が異なる）ことによる**バイアス**が、機械学習アプリケーションに悪影響を与えないようにすることができます。
# MAGIC 
# MAGIC #### ホットフィックスとロールバック (Hotfixes and Rollbacks)
# MAGIC 
# MAGIC 新コードの統合とデプロイを継続的に行うことのもう一つの利点は、ミスや問題を迅速に修正できることです。これには2つの方法があります。<br>
# MAGIC 
# MAGIC * **ホットフィックス**(**Hotfixes**): 実稼働中のソフトウェアアプリケーションのバグを迅速に修正するために書かれた小さなコードの断片。
# MAGIC * **ロールバック**(**Rollbacks**)：ソフトウェアアプリケーションを、正しく機能した最後のバージョンに戻すこと。

# COMMAND ----------

# MAGIC %md <i18n value="a77d37ff-6aa1-4a02-abb4-002ac0c638aa"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ![Azure ML Pipeline](https://files.training.databricks.com/images/ml-deployment/model-cicd.png)

# COMMAND ----------

# MAGIC %md <i18n value="241daf63-5043-4caf-99d1-a074412f45ec"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## モデルCI/CDのためのツール (Tools for Model CI/CD)
# MAGIC 
# MAGIC 機械学習にCI/CDを採用する場合、役立つツールがいろいろとあります。
# MAGIC 
# MAGIC 一般に、次のようなカテゴリに分類されます。
# MAGIC 
# MAGIC * **オーケストレーション**：アプリケーションのフローを制御します。
# MAGIC * **Git Hooks**: Gitリポジトリで特定のイベントが発生したときに自動的にコードを実行します。
# MAGIC * **アーティファクト管理**：パッケージソフトや機械学習モデルなどのアーティファクトを管理します。
# MAGIC * **環境管理**：アプリケーションで利用可能なソフトウェアリソースを管理します。
# MAGIC * **テスト**：モデルの妥当性と有効性を評価するためのテストを開発します。
# MAGIC * **アラート**：特定のイベントやテスト結果が発生したときに、該当のステークホルダーに通知します。
# MAGIC 
# MAGIC それぞれのカテゴリーに共通するツールは以下の通りです。<br>
# MAGIC 
# MAGIC 
# MAGIC |                        | OSS標準|Databricks|AWS|Azure|サードパーティ|
# MAGIC |------------------------|----------------------------------|--------------------------|-------------------------------------|---------------------|-----------------------------------|
# MAGIC | **オーケストレーション** |Airflow、Jenkins | ワークフロー、ジョブ、ノートブックワークフロー | CodePipeline、CodeBuild、CodeDeploy | DevOps、データファクトリー |                             |
# MAGIC | **Git Hooks** | | MLflow Webhooks | | | Github Actions、Gitlab、Travis CI |
# MAGIC | **アーティファクト管理** | PyPI、Maven | MLflow Model Registry | | | Nexus |
# MAGIC | **環境管理**| Docker、Kubernetes、Conda、pyenv | | Elastic Container Repository | コンテナ・レジストリ | DockerHub | 
# MAGIC | **Testing** | pytest | | | | Sonar | 
# MAGIC | **Alerting** | | Job | CloudWatch | モニター | PagerDuty、Slackとの統合 | 

# COMMAND ----------

# MAGIC %md <i18n value="d6203326-d8a9-4dab-bc9a-9ee5f6f0c93c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ML CI/CDリソース (ML CI/CD Resources)
# MAGIC 
# MAGIC 機械学習のためのCI/CDについて詳しく知りたい場合は、以下のリソースをご参考ください。
# MAGIC 
# MAGIC **ブログ**<br>
# MAGIC 
# MAGIC * <a href="https://databricks.com/blog/2017/10/30/continuous-integration-continuous-delivery-databricks.html" target="_blank">Continuous Integration and Continuous Delivery with Databricks</a>
# MAGIC * <a href="https://databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html" target="_blank">Productionizing Machine Learning: From Deployment to Drift Detection</a>
# MAGIC * <a href="https://databricks.com/blog/2020/10/13/using-mlops-with-mlflow-and-azure.html" target="_blank">Using MLOps with MLflow and Azure</a>
# MAGIC * <a href="https://ml-ops.org/content/mlops-principles" target="_blank">MLOps Principles</a>
# MAGIC 
# MAGIC **ドキュメント**<br>
# MAGIC 
# MAGIC * <a href="https://docs.microsoft.com/en-us/azure/databricks/dev-tools/ci-cd/ci-cd-azure-devops" target="_blank">Continuous integration and delivery on Azure Databricks using Azure DevOps</a>
# MAGIC * <a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" target="_blank">MLOps Continuous Delivery</a>
# MAGIC * <a href="https://docs.databricks.com/dev-tools/ci-cd.html" target="_blank">Continuous Integration and Delivery on Databricks using Jenkins</a>
# MAGIC 
# MAGIC **ツール**<br>
# MAGIC 
# MAGIC * <a href="https://github.com/databrickslabs/cicd-templates" target="_blank">CI/CD Templates</a>
# MAGIC * <a href="https://github.com/databrickslabs/dbx" target="_blank">DBX library</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
