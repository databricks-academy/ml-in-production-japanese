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
# MAGIC # アラート機能 (Alerting)
# MAGIC 
# MAGIC アラート機能により、異なるアプリケーションの進行状況を告知することができ、本番システムの自動化においてますます重要となっています。 このレッスンでは、電子メールとSlackやMicrosoft TeamsなどのツールとのREST統合を使用した基本的なアラート方法について探求します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、以下のことを実施します。<br>
# MAGIC  - アラートツールのランドスケイプの探索
# MAGIC  - SlackやMicrosoft Teamsと連携した基本的なRESTアラートの作成

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### アラートツールのランドスケイプ (The Alerting Landscape)
# MAGIC 
# MAGIC アラートツールには、さまざまなレベルのものがあります。
# MAGIC * PagerDuty 
# MAGIC   - 本番システムの停止を監視する最も人気なツールの1つです。
# MAGIC   - テキストメッセージや電話で問題をチーム間にエスカレーションすることができます。
# MAGIC * SlackまたはMicrosoft Teams
# MAGIC * Twilio   
# MAGIC * メール通知
# MAGIC 
# MAGIC ほとんどのアラートフレームワークは、REST統合によってカスタムアラートを出すことができます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 基本的なアラートの設定 (Setting Basic Alerts)
# MAGIC 
# MAGIC SlackまたはMicrosoft Teamsのエンドポイントを使用して、基本的なアラートを作成します。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC 
# MAGIC Microsoft Teamsのエンドポイントを設定するには、以下を実行します。<br>
# MAGIC 
# MAGIC 1. Teams設定後、**Teams**タブをクリックします。
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams1.png" style="height:200px; margin:20px"/></div>
# MAGIC 2. エンドポイントを関連付けたいチームの隣にあるドロップダウンをクリックします（チームがない場合は新規に作成します）。 次に、**Connectors**をクリックします。 <br></br> <br></br> <br></br> <br></br
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams2.png" style="height:350px; margin:20px"/></div>
# MAGIC 3. **Incoming Webhook**の隣にある**Configure**を選択します。 <br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams3.png" style="height:250px; margin:20px"/></div>
# MAGIC 4. Webhookに名前を付けて、**Create** をクリックします。
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams4.png" style="height:250px; margin:20px"/></div>
# MAGIC 5. URLをコピーして、下に貼り付けてください。
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams5.png" style="height:250px; margin:20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC SlackのWebhookを定義します。  
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> <a href="https://api.slack.com/incoming-webhooks#getting-started" target="_blank">以下の4つのステップを使用し、 </a>Slackのwebhookを定義します。<br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> この方法はPagerDutyにも同じように適用されます。
# MAGIC 
# MAGIC こちらの案内に従って、<a href="https://api.slack.com/incoming-webhooks#" target="_blank">incoming webhooks</a>を有効にしてください。

# COMMAND ----------

webhook_ml_production_api_demo = "" # FILL_IN

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC テストメッセージを送信し、Slackで確認します。

# COMMAND ----------

def post_api_endpoint(content, webhook=""):
    """
    Post message to Teams to log progress
    """
    import requests
    from requests.exceptions import MissingSchema
    from string import Template

    t = Template("{'text': '${content}'}")

    try:
        response = requests.post(webhook, data=t.substitute(content=content), headers={"Content-Type": "application/json"})
        return response
    except MissingSchema:
        print("Please define an appropriate API endpoint use by defining the `webhook` argument")

post_api_endpoint("This is my post from Python", webhook_ml_production_api_demo)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC * **質問:** アラートツールの代表的なものは何ですか？ 
# MAGIC * **回答:** PagerDutyは、本番環境で最も使用されるツールである傾向があります。 また、アラートをメールで送信するSMTPサーバーや、テキストメッセージでアラートを送信するTwilioも人気があります。 Slackのwebhookやbotも簡単に作ることができます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC * **Q:** このレッスンで紹介されたアラートツールはどこにありますか？ 
# MAGIC * **A:** <a href="https://www.twilio.com" target="_blank">Twilio</a> と <a href="https://www.pagerduty.com" target="_blank">PagerDuty</a> をチェックしてみてください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
