# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run "../../Includes/Drift-Monitoring-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="90df02e3-24c6-4bc6-99db-31417257255f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ラボ：ドリフト監視ラボ (Drift Monitoring Lab)
# MAGIC 
# MAGIC このラボでは、アイスクリームショップのシミュレーション・データを利用します。このデータには、前回のレッスンで見たような、2つの時間帯のデータが含まれています。2つの時間帯で発生する可能なドリフトを、先ほど学んだテクニックを使って特定するタスクです。
# MAGIC 
# MAGIC データには以下のカラムが含まれています。
# MAGIC 
# MAGIC **数値型：**
# MAGIC * **`temperature`** : その日の気温
# MAGIC * **`number_of_cones_sold`** : その日のアイスクリームのコーンの販売数
# MAGIC * **`number_bowls_sold`** : その日のアイスクリームのボウルの販売数
# MAGIC *  **`total_store_sales`** : お店のアイスクリーム以外の商品の売上金額の合計
# MAGIC * **`total_sales_predicted`** : その日のtotal_store_salesに対するモデルによる予測値
# MAGIC 
# MAGIC **カテゴリー型：**
# MAGIC * **`most_popular_ice_cream_flavor`** : その日一番人気のアイスクリームのフレーバー
# MAGIC * **`most_popular_sorbet_flavor`** : その日一番人気のシャーベットフレーバー
# MAGIC 
# MAGIC 
# MAGIC アイスクリーム以外の商品(例えばTシャツやその他の商品)の総売上高の予測に架空のモデルがあるとします。
# MAGIC 1回目と2回目のシミュレーション・データが与えられたら、潜在的なドリフトを特定し、それをどのように処理するかを分析します。

# COMMAND ----------

# MAGIC %md <i18n value="41653424-6d5b-4214-b16f-bf5c6fe8e284"/>
# MAGIC 
# MAGIC 
# MAGIC それでは、最初の時間帯のアイスクリームのデータフレームを見てみましょう。

# COMMAND ----------

df1.head()

# COMMAND ----------

# MAGIC %md <i18n value="e608da11-41af-42db-a6ef-95e26e17e530"/>
# MAGIC 
# MAGIC このデータセットから、シミュレーションのドリフトの形を特定してみてください。データセットは以下のように変更されています。
# MAGIC 
# MAGIC 1. 上流のデータ管理上のミスで華氏が摂氏に変換されていました。
# MAGIC 2. コーンの販売個数は一定値になっていました。
# MAGIC 3. アイスクリームの一番人気のフレーバーの分布が変わったが、欠損はありませんでした。
# MAGIC 4. ボウルが人気になって、販売数が増えています。
# MAGIC 5. 最も人気のあるシャーベットフレーバーには欠損があり、均等に分布しているものの、欠損の数が変化しています。
# MAGIC 2. アイスクリーム以外の商品の売上は上がりました。
# MAGIC 2. 店舗総売上高の予測値が下がりました。
# MAGIC 
# MAGIC これらの変化を念頭に置きながら、学んだ方法を使って、どのように変化を検知するかをやってみましょう。

# COMMAND ----------

# MAGIC %md <i18n value="132cbccb-ebd5-40fa-b0a0-f0c829fe2779"/>
# MAGIC 
# MAGIC 
# MAGIC それでは、2つ目の時間帯のアイスクリームのDataFrameを見てみましょう。

# COMMAND ----------

df2.head()

# COMMAND ----------

# MAGIC %md <i18n value="e9e9c8c7-8904-45d1-be3a-e8ff188656b8"/>
# MAGIC 
# MAGIC **`Monitor`** クラスを定義しています。このクラスを利用して、以下の質問に回答してください。

# COMMAND ----------

import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np 
from scipy.spatial import distance

class Monitor():
  
    def __init__(self, pdf1, pdf2, cat_cols, num_cols, alpha=.05, js_stat_threshold=0.2):
        """
        Pass in two pandas dataframes with the same columns for two time windows
        List the categorical and numeric columns, and optionally provide an alpha level
        """
        assert (pdf1.columns == pdf2.columns).all(), "Columns do not match"
        self.pdf1 = pdf1
        self.pdf2 = pdf2
        self.categorical_columns = cat_cols
        self.continuous_columns = num_cols
        self.alpha = alpha
        self.js_stat_threshold = js_stat_threshold
    
    def run(self):
        """
        Call to run drift monitoring
        """
        self.handle_numeric_js()
        self.handle_categorical()
        
        pdf1_nulls = self.pdf1.isnull().sum().sum()
        pdf2_nulls = self.pdf2.isnull().sum().sum()
        print(f"{pdf1_nulls} total null values found in pdf1 and {pdf2_nulls} in pdf2")
        
  
    def handle_numeric_ks(self):
        """
        Handle the numeric features with the Two-Sample Kolmogorov-Smirnov (KS) Test with Bonferroni Correction 
        """
        corrected_alpha = self.alpha / len(self.continuous_columns)

        for num in self.continuous_columns:
            ks_stat, ks_pval = stats.ks_2samp(self.pdf1[num], self.pdf2[num], mode="asymp")
            if ks_pval <= corrected_alpha:
                self.on_drift(num)
                
    def handle_numeric_js(self):
        """
        Handles the numeric features with the Jensen Shannon (JS) test using the threshold attribute
        """
        for num in self.continuous_columns:
            # Run test comparing old and new for that attribute
            range_min = min(self.pdf1[num].min(), self.pdf2[num].min())
            range_max = max(self.pdf1[num].max(), self.pdf2[num].max())
            base = np.histogram(self.pdf1[num], bins=20, range=(range_min, range_max))
            comp = np.histogram(self.pdf2[num], bins=20, range=(range_min, range_max))
            js_stat = distance.jensenshannon(base[0], comp[0], base=2)
            if js_stat >= self.js_stat_threshold:
                self.on_drift(num)
      
    def handle_categorical(self):
        """
        Handle the Categorical features with Two-Way Chi-Squared Test with Bonferroni Correction
        Note: null counts can skew the results of the Chi-Squared Test so they're currently dropped
            by `.value_counts()`
        """
        corrected_alpha = self.alpha / len(self.categorical_columns)

        for feature in self.categorical_columns:
            pdf_count1 = pd.DataFrame(self.pdf1[feature].value_counts()).sort_index().rename(columns={feature:"pdf1"})
            pdf_count2 = pd.DataFrame(self.pdf2[feature].value_counts()).sort_index().rename(columns={feature:"pdf2"})
            pdf_counts = pdf_count1.join(pdf_count2, how="outer")#.fillna(0)
            obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
            _, p, _, _ = stats.chi2_contingency(obs)
            if p < corrected_alpha:
                self.on_drift(feature)

    def generate_null_counts(self, palette="#2ecc71"):
        """
        Generate the visualization of percent null counts of all features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        return pd.concat([100 * self.pdf1.isnull().sum() / len(self.pdf1), 
                          100 * self.pdf2.isnull().sum() / len(self.pdf2)], axis=1, 
                          keys=["pdf1", "pdf2"]).style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
    
    def generate_percent_change(self, palette="#2ecc71"):
        """
        Generate visualization of percent change in summary statistics of numeric features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        summary1_pdf = self.pdf1.describe()[self.continuous_columns]
        summary2_pdf = self.pdf2.describe()[self.continuous_columns]
        percent_change = 100 * abs((summary1_pdf - summary2_pdf) / (summary1_pdf + 1e-100))
        return percent_change.style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
  
    def on_drift(self, feature):
        """
        Complete this method with your response to drift.  Options include:
          - raise an alert
          - automatically retrain model
        """
        print(f"Drift found in {feature}!")


# COMMAND ----------

# MAGIC %md <i18n value="b181cd42-1406-4bf8-8dc6-77a9e9f60cdd"/>
# MAGIC 
# MAGIC ドリフトを特定するために、2つの時間枠のアイスクリームのデータを基に、 **`Monitor`** オブジェクトを作成します。

# COMMAND ----------

drift_monitor = Monitor(
  df1,
  df2, 
  cat_cols = ["most_popular_ice_cream_flavor", "most_popular_sorbet_flavor"], 
  num_cols = ["temperature", "number_of_cones_sold", "number_bowls_sold", "total_store_sales", "total_sales_predicted"],
  alpha=.05, 
  js_stat_threshold=0.2
)

# COMMAND ----------

# MAGIC %md <i18n value="24755f69-2a0e-45ba-a1f3-b45871e25dbb"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 要約統計 (Summary Statistics)
# MAGIC 
# MAGIC 2つデータセットのデータとその要約統計量に確認し、比較してみてください。 **`drift_monitor`** クラスを使用して欠損の数を出します。何か印象に残っていることはありますか？

# COMMAND ----------

# ANSWER
# most_popular_sorbet_flavor has a 20% null count in pdf2!
drift_monitor.generate_null_counts()

# COMMAND ----------

# MAGIC %md <i18n value="52effbfd-a185-4d1e-a711-fe5997db94ed"/>
# MAGIC 
# MAGIC 
# MAGIC **`drift_monitor`** クラスを使用して変化をパーセント形式で出します。何か印象に残ることはありますか？

# COMMAND ----------

# ANSWER
# temperature, number_bowls_sold, total_store_sales, and total_sales_predicted seemed to change a bit!
drift_monitor.generate_percent_change()

# COMMAND ----------

# MAGIC %md <i18n value="ee2a0b06-5a3f-4e1c-b255-3a2f59db70d5"/>
# MAGIC 
# MAGIC Investigate why `temperature` has such a big percent change! If you compare `df1.describe()` and `df2.describe()`, what differences do you see? `df1` uses Fahrenheit whereas `df2` uses Celsius! In this case, it was relatively easy to find out the root cause of the drift; however, in real use cases, it might be much harder! 
# MAGIC 
# MAGIC  `temperature` の変化がこれほど大きい理由を調べてください! `df1.describe()` と `df2.describe()` を比較すると、どのような違いが見られますか? `df1` は華氏を使用しますが、`df2` は摂氏を使用します! 今回のケースではドリフトの根本原因を見つけるのは比較的簡単でした。ただし、実際のケースでは、はるかに難しい場合があります。

# COMMAND ----------

# MAGIC %md <i18n value="29aaae05-cbd2-4515-b483-3b7224bf6187"/>
# MAGIC 
# MAGIC 
# MAGIC ### 統計検定 (Statistical Tests)
# MAGIC 
# MAGIC では、Jensen-Shannonとカイ二乗の分割表検定（Bonferroni 補正付き）を試してみましょう。
# MAGIC 
# MAGIC いずれも **`drift_monitor.run()`** を呼び出すと実装されます。それぞれの検定で統計的に有意なp値が見つかった場合、またはJS統計があらかじめ設定した閾値以上である場合に、その特徴量の名前を出力されます。
# MAGIC 
# MAGIC 結果を確認し、上記の変更点と比較します。

# COMMAND ----------

# ANSWER
# Note the chi-squared test filters nulls so this solution simply prints out the total null values
drift_monitor.run()

# COMMAND ----------

# MAGIC %md <i18n value="063891c2-6b81-47ec-8a01-76511bb52349"/>
# MAGIC 
# MAGIC 
# MAGIC ### 精査 (Closer Look)
# MAGIC 
# MAGIC ***これらの要約統計量と統計的検定を用いて、ドリフトをすべて捕らえることができましたでしょうか？***
# MAGIC 
# MAGIC あなたがこのアイスクリーム屋を経営していると想像してください。
# MAGIC * ***それぞれの状況にどのように対処しますか？***
# MAGIC * ***モデルやビジネスにどのような影響を与えますか？***

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
