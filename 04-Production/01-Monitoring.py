# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="11677a04-117e-48ac-82d0-fe478df33360"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # ドリフト監視 (Drift Monitoring)
# MAGIC 
# MAGIC モデルを時間の経過とともに監視することは、モデルパフォーマンスのドリフトや破壊的な変更に対して、保護することに繋がります。 このレッスンでは、ドリフトへの対応策を探り、ドリフトを特定するための統計的手法を実装します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを実施します。<br>
# MAGIC  - ドリフトの種類と関連の統計的手法を分析します。
# MAGIC  - コルモゴロフ–スミルノフ検定とジェンセン–シャノン検定を使用してドリフトを判別します。
# MAGIC  - 要約統計を使用してドリフトを監視します。
# MAGIC  - 包括的な監視ソリューションを適用します。
# MAGIC  - ドリフトの監視におけるアーキテクチャ上の考慮事項を探ります。

# COMMAND ----------

# MAGIC %md <i18n value="438a1fd4-30c3-4362-92e0-df5e77f3060d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ドリフト監視 (Drift Monitoring)
# MAGIC 
# MAGIC 大多数の機械学習モデルは、データが定常確率分布に従って生成されることを前提としています。しかし、人間の活動を反映しているほとんどのデータセットは時間とともに変化するため、機械学習モデルはしばしば陳腐化します。
# MAGIC 
# MAGIC たとえば、COVID-19が大流行する前にレストラン売上を予測するモデルは、パンデミック下のレストラン売上の予測に使えなくなるでしょう。データの分布は、時間の経過とともに変化またはドリフトをしています。
# MAGIC 
# MAGIC ドリフトは、いくつかの種類があります。
# MAGIC 
# MAGIC * **データのドリフト**
# MAGIC   * **データ変化**
# MAGIC     * 実際には、上流でのデータ変更に伴うドリフトが、最も一般的な原因の1つです。
# MAGIC     * 例えば、変更されたETLタスクからの発生したNullレコードなどです
# MAGIC   * **特徴量のドリフト** 
# MAGIC     * 入力用特徴量の分布の変化
# MAGIC     *  \\(P(X)\\)の変化
# MAGIC   * **ラベルのドリフト**
# MAGIC     * データ中のラベルの分布の変化
# MAGIC     *  \\(P(Y)\\)の変化
# MAGIC   * **予測値のドリフト** 
# MAGIC       * モデルから得られる予測結果ラベルの分布の変化
# MAGIC       * \\(P(\hat{Y}| X)\\の変更 
# MAGIC * **コンセプトのドリフト** 
# MAGIC   * 入力変数と目的変数ラベルの相関の変化
# MAGIC   * \\(P(Y| X)\\)の分布の変化
# MAGIC   * 現行モデルが無効になる可能性が高い
# MAGIC 
# MAGIC **ドリフトを監視するためには、ドリフトの原因毎にを監視する必要があります。**

# COMMAND ----------

# MAGIC %md <i18n value="4b7fc32c-42d4-430b-8312-93e67efdfeb5"/>
# MAGIC 
# MAGIC それぞれの状況に応じて異なる対処が必要があり、ドリフトが生じたからと言って直ちに現在のモデルを更新する必要はないことに注意してください。
# MAGIC 
# MAGIC 例えば、
# MAGIC * 気温を入力変数として、雪見だいふくの売上を予測するように設計されたモデルを想像してみてください。最近のデータは気温の値が高く、雪見だいふくの売上が高い場合、特徴量とラベルの両方のドリフトが発生していますが、モデルがうまく機能している限り、問題はありません。しかし、このような変化があると、他のビジネスアクションを取る可能性があるので、とにかく変化を監視することが重要です。
# MAGIC 
# MAGIC * ただし、気温が上昇して売上が増加したが、予測がこの変化と一致しなかった場合は、コンセプト・ドリフトが発生している可能性があり、モデルを再トレーニングする必要があります。
# MAGIC 
# MAGIC * いずれの場合も、他のビジネスプロセスに影響を与える可能性があるので、潜在的なドリフトをすべて追跡することが重要です。
# MAGIC 
# MAGIC **起こり得る変化に最適に適応するために、データと予測結果を時間軸で比較し、発生しうるあらゆる種類のドリフトを特定します。** 

# COMMAND ----------

# MAGIC %md <i18n value="eb4e7ab9-9d0c-4d59-9eaa-2db2da05b5a9"/>
# MAGIC 
# MAGIC ドリフト監視の本質は、**データの時間窓（Time Windows)に対して統計検定を実行すること**です。これにより、ドリフトを検出し、特定の根本原因にたどり着くことができます。ここにいくつかの解決策を説明します。
# MAGIC 
# MAGIC **数値型特徴量**
# MAGIC * 要約統計量
# MAGIC   * 平均、中央値、分散、欠損値の数、最大値、最小値
# MAGIC * 検定テスト
# MAGIC   * <a href="https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence" target="_blank">ジェンセン・シャノン</a>
# MAGIC     -このメソッドは、平滑化および正規化されたMetricを提供します。
# MAGIC   * <a href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test" target="_blank">2標本コルモゴロフ・スミルノフ (KS)、</a> <a href="https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test" target="_blank">マン・ホイットニー</a>、または <a href="https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test" target="_blank">ウィルコクソン</a>
# MAGIC     - **注意**:これらの検定は、正規性の仮定と処理可能なデータサイズにより大きく異なります。
# MAGIC     - 正規性をチェックし、結果に基づいて適切な検定を選択します（例えば、マン・ホイットニーはスキューに対してより寛容でです）。 
# MAGIC   * <a href="https://ja.wikipedia.org/wiki/ワッサースタイン計量" target="_blank">ワッサースタイン計量</a>
# MAGIC   * <a href="https://ja.wikipedia.org/wiki/カルバック・ライブラー情報量" target="_blank">カルバック・ライブラー情報量</a>
# MAGIC     - Jensen-Shannonダイバージェンスに関連しています。
# MAGIC 
# MAGIC     
# MAGIC **カテゴリー型特徴量**
# MAGIC * 要約統計量
# MAGIC   * 最頻値、固有値の数、欠損値の数
# MAGIC * 検定テスト
# MAGIC   * <a href="https://ja.wikipedia.org/wiki/カイ二乗検定" target="_blank">一元カイ二乗検定</a>
# MAGIC   * <a href="https://en.wikipedia.org/wiki/Chi-squared_test" target="_blank">カイ二乗の分割表検定</a>
# MAGIC   * <a href="https://ja.wikipedia.org/wiki/フィッシャーの正確確率検定" target="_blank">フィッシャーの正確確率検定</a>
# MAGIC 
# MAGIC また、入力変数と出力ラベルの関係を記録したい場合があります。その場合、ラベル変数のデータ型によって異なる方法で処理します。
# MAGIC 
# MAGIC **数値型**
# MAGIC * <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" target="_blank">ピアソン係数</a>
# MAGIC 
# MAGIC **カテゴリ型** 
# MAGIC * <a href="https://ja.wikipedia.org/wiki/分割表" target="_blank">Contingency Tables</a>分割表</a>
# MAGIC 
# MAGIC 興味深い代替案として、次の方法があります。ドリフトの監視を教師あり学習問題として捉え、特徴量とラベルをモデルへの入力データとして使用し、ラベルは与えられた行がトレーニングセットからかるおか推論セットから来るのかを示すことです。モデルの精度が向上すると、モデルがドリフトしたことを意味します。
# MAGIC 
# MAGIC やってみましょう！

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="232b2c47-e056-4adf-8f74-9515e3fc164e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Kolmogorov-Smirnov 検定 (Kolmogorov-Smirnov Test) 
# MAGIC 
# MAGIC 数値型特徴量には**2標本コルモゴロフ-スミルノフ (KS) 検定**を使用します。この検定は、2つの標本の分布が異なるかどうかを判定します。この検定では、以下を実施します。<br><br>
# MAGIC 
# MAGIC - 2つの異なる分布を持つ確率が高い場合に、より高いKS統計値を返します。
# MAGIC - 統計的有意性が高いほど、低いP値を返します。
# MAGIC 
# MAGIC 実際には、p値の閾値が必要です。P値が0.05（5%）を下回った場合、サンプルの分布が異なると見なします。通常、このしきい値（アルファレベル）は0.05に設定します。

# COMMAND ----------

import seaborn as sns
from scipy.stats import gaussian_kde, truncnorm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance

def plot_distribution(distibution_1, distibution_2):
    """
    Plots the two given distributions 

    :param distribution_1: rv_continuous 
    :param distribution_2: rv_continuous 

    """
    sns.kdeplot(distibution_1, shade=True, color="g", label=1)
    sns.kdeplot(distibution_2, shade=True, color="b", label=2)
    plt.legend(loc="upper right", borderaxespad=0)

def get_truncated_normal(mean=0, sd=1, low=0.2, upp=0.8, n_size=1000, seed=999):
    """
    Generates truncated normal distribution based on given mean, standard deviation, lower bound, upper bound and sample size 

    :param mean: float, mean used to create the distribution 
    :param sd: float, standard deviation used to create distribution
    :param low: float, lower bound used to create the distribution 
    :param upp: float, upper bound used to create the distribution 
    :param n_size: integer, desired sample size 

    :return distb: rv_continuous 
    """
    np.random.seed(seed=seed)

    a = (low-mean) / sd
    b = (upp-mean) / sd
    distb = truncnorm(a, b, loc=mean, scale=sd).rvs(n_size, random_state=seed)
    return distb

def calculate_ks(distibution_1, distibution_2):
    """
    Helper function that calculated the KS stat and plots the two distributions used in the calculation 

    :param distribution_1: rv_continuous
    :param distribution_2: rv_continuous 

    :return p_value: float, resulting p-value from KS calculation
    :return ks_drift: bool, detection of significant difference across the distributions 
    """
    base, comp = distibution_1, distibution_2
    p_value = np.round(stats.ks_2samp(base, comp)[1],3)
    ks_drift = p_value < 0.05

    # Generate plots
    plot_distribution(base, comp)
    label = f"KS Stat suggests model drift: {ks_drift} \n P-value = {p_value}"
    plt.title(label, loc="center")
    return p_value, ks_drift

def calculate_probability_vector(distibution_1, distibution_2):
    """
    Helper function that turns raw values into a probability vector 

    :param distribution_1: rv_continuous
    :param distribution_2: rv_continuous 

    :return p: array, probability vector of distribution_1
    :return q: array, probability vector of distribution_2
    """
    global_min = min(min(distibution_1), min(distibution_2))
    global_max = max(max(distibution_1), max(distibution_2))
    
    p = np.histogram(distibution_1, bins=20, range=(global_min, global_max))
    q = np.histogram(distibution_2, bins=20, range=(global_min, global_max))
    
    return p[0], q[0]
    
def calculate_js_distance(p, q, raw_distribution_1, raw_distribution_2, threshold=0.2):
    """
    Helper function that calculated the JS distance and plots the two distributions used in the calculation 

    :param p: array, probability vector for the first distribution
    :param q: array, probability vector for the second distribution 
    :param raw_distribution_1: array, raw values used in plotting
    :param raw_distribution_2: array, raw values used in plotting
    :param threshold: float, cutoff threshold for the JS statistic

    :return js_stat: float, resulting distance measure from JS calculation
    :return js_drift: bool, detection of significant difference across the distributions 
    """
    js_stat = distance.jensenshannon(p, q, base=2)
    js_stat_rounded = np.round(js_stat, 3)
    js_drift = js_stat > threshold

    # Generate plot
    plot_distribution(raw_distribution_1, raw_distribution_2)
    label = f"Jensen Shannon suggests model drift: {js_drift} \n JS Distance = {js_stat_rounded}"
    plt.title(label, loc="center")

    return js_stat, js_drift

# COMMAND ----------

# MAGIC %md <i18n value="f3740dad-ea94-4fcc-9577-7ac36398b1ee"/>
# MAGIC 
# MAGIC サンプルサイズ　50　で始めてみましょう。

# COMMAND ----------

calculate_ks(
  get_truncated_normal(upp=.80, n_size=50), 
  get_truncated_normal(upp=.79, n_size=50) 
)

# COMMAND ----------

# MAGIC %md <i18n value="8d7321cf-8bc9-48ab-be02-6e78ac8276a5"/>
# MAGIC 
# MAGIC 
# MAGIC 分布がかなり似ていて、p値が高いことがわかります。では、サンプルサイズを増やしてp値への影響を見てみましょう。 **`N = 10,000`** に設定します。

# COMMAND ----------

calculate_ks(
  get_truncated_normal(upp=.80, n_size=1000), 
  get_truncated_normal(upp=.79, n_size=1000)
)

# COMMAND ----------

# MAGIC %md <i18n value="4971d477-a582-46f0-8d3a-a3416d52e118"/>
# MAGIC 
# MAGIC サンプルサイズを大きくすると、p値が大幅に下がりました。サンプルサイズをさらに10倍にしましょう: **`N = 100,000`** 

# COMMAND ----------

calculate_ks(
  get_truncated_normal(upp=.80, n_size=100000), 
  get_truncated_normal(upp=.79, n_size=100000) 
)

# COMMAND ----------

# MAGIC %md <i18n value="8f4ca19a-53ed-40ea-ad4a-3bb9e0ca7ee8"/>
# MAGIC 
# MAGIC 
# MAGIC サンプルサイズが大きくなると、 **`P_value`** はゼロ近くまで低下し、2つのサンプルが有意に異なることを示しています。しかし、2つの分布のプロットを視覚的に見るだけでは、かなり似ています。KSの検定では、2つの分布の類似性を判断するのに役立ちますが、サンプルサイズに制限に直面することになります。では、サンプルサイズが*大きい*場合、分布の類似性をどのように検定できるでしょうか？

# COMMAND ----------

# MAGIC %md <i18n value="e58287d8-9bf3-43cd-a686-20ec4e497da4"/>
# MAGIC 
# MAGIC ## Jensen Shannon
# MAGIC 
# MAGIC Jensen Shannon (JS) 距離は、**2つの確率分布間の距離を測定し、平滑化および正規化されるため、大きなデータセットでのドリフト検出に適しています。** 対数底が2の場合、JS統計量は0から1の範囲になります。
# MAGIC 
# MAGIC - 0は分布が同一であることを意味します。
# MAGIC - 1は分布に類似性がないことを意味します。
# MAGIC 
# MAGIC JS距離は <a href="https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence" target="_blank">JSダイバージェンスの平方根として定義されます</a>。
# MAGIC 
# MAGIC ![Jensen Shannon Divergence](https://miro.medium.com/max/1400/1*viATYZeg9SiT-ZdzYGjKYA.png)
# MAGIC 
# MAGIC ここで、*M*は*P*と*Q*の点ごとの平均として定義され、*H(P)*はエントロピー関数として定義されます。
# MAGIC 
# MAGIC ![JS Entropy](https://miro.medium.com/max/1400/1*NSIn8OVTKufpSlvOOoXWQg.png)
# MAGIC 
# MAGIC p値を提供するKS統計とは異なり、JS統計はスカラー値のみを提供します。したがって、**カットオフ閾値を手動で指定**する必要があります。この値を超えると、2つのデータセットがドリフトしていると特定されます。

# COMMAND ----------

# MAGIC %md <i18n value="f68d1cd1-a4da-4400-a52b-92360baf4f42"/>
# MAGIC 
# MAGIC 
# MAGIC 同一の分布を持つ場合、JS統計量が0になることを検証します。ここでの **`p`** と **`q`** 引数は確率ベクトルであり、生の値ではないことに注意してください。

# COMMAND ----------

distance.jensenshannon(p=[1.0, 0.0, 1.0], q=[1.0, 0.0, 1.0], base=2.0)

# COMMAND ----------

# MAGIC %md <i18n value="8cae5f7f-adf6-43d6-bfb4-7a50b45dfce0"/>
# MAGIC 
# MAGIC 
# MAGIC この例をもう一度 **`N=1,000`** で試してみよう。

# COMMAND ----------

raw_distribution_1 = get_truncated_normal(upp=.80, n_size=1000)
raw_distribution_2 = get_truncated_normal(upp=.79, n_size=1000)

p, q = calculate_probability_vector(raw_distribution_1, raw_distribution_2)

calculate_js_distance(p, q, raw_distribution_1, raw_distribution_2, threshold=0.2) 

# COMMAND ----------

# MAGIC %md <i18n value="20eb1618-d5ff-4bd7-b772-6d342497326f"/>
# MAGIC 
# MAGIC そして、 **`N = 10,000`** 

# COMMAND ----------

raw_distribution_1 = get_truncated_normal(upp=.80, n_size=10000)
raw_distribution_2 = get_truncated_normal(upp=.79, n_size=10000)

p, q = calculate_probability_vector(raw_distribution_1, raw_distribution_2)

calculate_js_distance(p, q, raw_distribution_1, raw_distribution_2, threshold=0.2) 

# COMMAND ----------

# MAGIC %md <i18n value="4858cfcd-903e-4839-9eba-313a923e1a16"/>
# MAGIC 
# MAGIC 最後に、 **`N = 100,000`** 

# COMMAND ----------

raw_distribution_1 = get_truncated_normal(upp=.80, n_size=100000)
raw_distribution_2 = get_truncated_normal(upp=.79, n_size=100000)

p, q = calculate_probability_vector(raw_distribution_1, raw_distribution_2)

calculate_js_distance(p, q, raw_distribution_1, raw_distribution_2, threshold=0.2) 

# COMMAND ----------

# MAGIC %md <i18n value="db1e429a-8590-4658-b234-13aea4800a81"/>
# MAGIC 
# MAGIC 
# MAGIC 上に示したように、JS距離は平滑化および正規化されているため、サンプルサイズの増加に対してはるかに強い耐性があります。

# COMMAND ----------

# MAGIC %md <i18n value="c76599da-4b09-4e6f-8826-557347429af8"/>
# MAGIC 
# MAGIC 実際には、ある一定期間のデータを取得し、それを時間に基づいてグループ（例：週単位など）に分割します。2つのグループに対して検定を実行し、統計的に有意な変化があったかどうかを判断します。これらの監視ジョブの頻度は、トレーニング・ウィンドウ、推論データのサンプルサイズ、およびユースケースによって異なります。これを実際のデータセットでシミュレーションsてみましょう。

# COMMAND ----------

# Load Dataset
airbnb_pdf = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")

# Identify Numeric & Categorical Columns
num_cols = ["accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews", "review_scores_rating", "price"]
cat_cols = ["neighbourhood_cleansed", "property_type", "room_type"]

# Drop extraneous columns for this example
airbnb_pdf = airbnb_pdf[num_cols + cat_cols]

# Split Dataset into the two groups
pdf1 = airbnb_pdf.sample(frac = 0.5, random_state=1)
pdf2 = airbnb_pdf.drop(pdf1.index)

# COMMAND ----------

# MAGIC %md <i18n value="e9d3aad2-2af9-4deb-84a9-a393211eaf2b"/>
# MAGIC 
# MAGIC ドリフトをシミュレートするには、 **`pdf2`** を変更します。次の現実的な変更を加えてみてください。
# MAGIC 
# MAGIC * ***Airbnbsの需要が急増したため、Airbnbの価格は2倍になりました***。
# MAGIC   * *ドリフトの種類*: コンセプト、ラベル 
# MAGIC * ***上流のデータ管理上のミスにより、`neighbourhood_cleansed`に欠損が発生しました。***
# MAGIC   * *ドリフトの種類*: 特徴量
# MAGIC * ***上流のデータ変更により、`review_score_rating`は、以前の100ポイントシステムではなく、5つ星の評価システムに移行しました。***
# MAGIC   * *ドリフトの種類*: 特徴量

# COMMAND ----------

pdf2["price"] = 2 * pdf2["price"]
pdf2["review_scores_rating"] = pdf2["review_scores_rating"] / 20
pdf2["neighbourhood_cleansed"] = pdf2["neighbourhood_cleansed"].map(lambda x: None if x == 0 else x)

# COMMAND ----------

# MAGIC %md <i18n value="75862f88-d5f4-4809-9bb6-c12e22755360"/>
# MAGIC 
# MAGIC 
# MAGIC ## 要約統計の適用 (Apply Summary Stats)
# MAGIC 
# MAGIC まず、 **`dbutils.data.summarize`** を使用し、2つのデータセットのデータ分布の要約統計量を確認することから始めます。

# COMMAND ----------

dbutils.data.summarize(pdf1)

# COMMAND ----------

dbutils.data.summarize(pdf2)

# COMMAND ----------

# MAGIC %md <i18n value="90a1c03a-c124-43bb-8083-2abf0fd778a9"/>
# MAGIC 
# MAGIC 要約プロットからは分布違いの箇所を見つけるのが難しいので、要約統計量の変化率を見える化しましょう。

# COMMAND ----------

# Create visual of percent change in summary stats
cm = sns.light_palette("#2ecc71", as_cmap=True)
summary1_pdf = pdf1.describe()[num_cols]
summary2_pdf = pdf2.describe()[num_cols]
percent_change = 100 * abs((summary1_pdf - summary2_pdf) / (summary1_pdf + 1e-100))
percent_change.style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)

# COMMAND ----------

# MAGIC %md <i18n value="2e4a5ada-393f-47cd-a9e6-d2f8cf8e570e"/>
# MAGIC 
# MAGIC **`review_scores_rating`** と **`price`** では、多くの統計量が大きく変更しているようなので、それらを調べてみたいと思います。次に、データの2つのサブセットに対してKSテストを実行します。ただし、一連のテストを実行しているため、この状況ではデフォルトのアルファレベル=0.05を使用できません。なぜなら、検定グループで少なくとも1つの偽陽性 (特徴量の分布が変化しなかったのに、変化したと結論付ける)の確率が、グループ内の検定数とともに増加するためです。
# MAGIC 
# MAGIC この問題を解決するために、 **ボンフェローニ補正(Bonferroni correction)** を採用します。これで、アルファ水準はグループ内の検定数あたり0.05に変更されます。これは一般的な方法であり、偽陽性の可能性を減らします。
# MAGIC 
# MAGIC 詳細については、<a href="http://ja.wikipedia.org/wiki/ボンフェローニ補正" target="_blank">こちらをご覧ください</a>。

# COMMAND ----------

# Set the Bonferroni Corrected alpha level
alpha = 0.05
alpha_corrected = alpha / len(num_cols)

# Loop over all numeric attributes (numeric cols and target col, price)
for num in num_cols:
    # Run test comparing old and new for that attribute
    ks_stat, ks_pval = stats.ks_2samp(pdf1[num], pdf2[num], mode="asymp")
    if ks_pval <= alpha_corrected:
        print(f"{num} had statistically significant change between the two samples")

# COMMAND ----------

# MAGIC %md <i18n value="37037a08-09a1-41ad-a876-a919c8895b25"/>
# MAGIC 
# MAGIC 上記のように、Jensen-Shannon距離はKS距離に比べていくつかの利点があるので、その検定も実行してみましょう。
# MAGIC 
# MAGIC p値がないため、ボンフェローニ補正は必要ありませんが、データセットに関する知識に基づいて手動でしきい値を設定する必要があります。

# COMMAND ----------

# Set the JS stat threshold
threshold = 0.2

# Loop over all numeric attributes (numeric cols and target col, price)
for num in num_cols:
    # Run test comparing old and new for that attribute
    range_min = min(pdf1[num].min(), pdf2[num].min())
    range_max = max(pdf1[num].max(), pdf2[num].max())
    base = np.histogram(pdf1[num], bins=20, range=(range_min, range_max))
    comp = np.histogram(pdf2[num], bins=20, range=(range_min, range_max))
    js_stat = distance.jensenshannon(base[0], comp[0], base=2)
    if js_stat >= threshold:
        print(f"{num} had statistically significant change between the two samples")

# COMMAND ----------

# MAGIC %md <i18n value="19ccfb17-b34c-4a70-b01a-11f1e2661507"/>
# MAGIC 
# MAGIC それでは、カテゴリ型の特徴量を見てみましょう。まず欠損値の割合を確認します。

# COMMAND ----------

# Generate missing value counts visual 
pd.concat(
  [100 * pdf1.isnull().sum() / len(pdf1), 100 * pdf2.isnull().sum() / len(pdf2)], 
  axis=1, 
  keys=["pdf1", "pdf2"]
).style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)

# COMMAND ----------

# MAGIC %md <i18n value="4bb159b0-c70f-45ab-a81f-e01ef41d66cd"/>
# MAGIC 
# MAGIC **`neighbourhood_cleansed`** にはpdf1にはなかったいくつかの欠損値があります。それでは、この例の **`カイ二乗の分割表検定`** を実行してみましょう。このテストは<a href="https://ja.wikipedia.org/wiki/分割表" target="_blank">分割表</a>を作成し、特定のカテゴリ型特徴量に対して各特徴カテゴリ値の数の列と、 **`pdf1`** と **`pdf2`** の行があります。
# MAGIC 
# MAGIC 次に、データの時間窓とその特徴量の分布との間に関連性があるかどうかを判断するp値を返します。それが有意であれば、分布は時間とともに変化し、ドリフトがあったと結論付けます。

# COMMAND ----------

alpha = 0.05
corrected_alpha = alpha / len(cat_cols) # Still using the same correction
    
for feature in cat_cols:
    pdf_count1 = pd.DataFrame(pdf1[feature].value_counts()).sort_index().rename(columns={feature:"pdf1"})
    pdf_count2 = pd.DataFrame(pdf2[feature].value_counts()).sort_index().rename(columns={feature:"pdf2"})
    pdf_counts = pdf_count1.join(pdf_count2, how="outer").fillna(0)
    obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
    _, p, _, _ = stats.chi2_contingency(obs)
    if p < corrected_alpha:
        print(f"{feature} statistically significantly changed")
    else:
        print(f"{feature} did not statistically significantly change")

# COMMAND ----------

# MAGIC %md <i18n value="770b3e78-3388-42be-8559-e7a0c1e345b0"/>
# MAGIC 
# MAGIC **注意:** カイ二乗の分割表検定は、欠損値が導入されたためではなく、あるneighbourhoodに欠損値が特別に導入され流ことで不均一な分布につながったためにドリフトを検出しました。欠損値が全体を通して一様であれば、この検定では依存性の変化としてフラグを立てず、低いカウントのままで同様の分布として見ることができます。

# COMMAND ----------

# MAGIC %md <i18n value="71d4c070-91ff-4314-986a-d9c799ca221f"/>
# MAGIC 
# MAGIC 
# MAGIC カイ二乗検定に関する追加の説明です。
# MAGIC 
# MAGIC カイ二乗検定では、ビン数が少ない分布の場合は検定を無効にし、偽陽性をもたらす可能性があります。 
# MAGIC 
# MAGIC また、カイ二乗検定には次の2種類があります。一元カイ二乗検定およびカイ二乗の分割表検定です。一元カイ二乗検定は適合度検定です。単一特徴量分布と母集団分布を取り、その母集団から単一特徴量分布をランダムに引いた場合の確率を報告します。ドリフト監視のコンテキストでは、古い時間窓を母集団分布として使用し、新しい時間窓を単一特徴量分布として使用します。p値が低い場合は、ドリフトが発生し、新しいデータが古い分布に似ていないという判断に繋がります。この検定は度数を比較するため、最近の時間窓の分布が似ていても、データが少ない場合は、おそらくそうではない場合、低いp値が返します。そのような場合は、カイ二乗の分割表検定を試してください。
# MAGIC 
# MAGIC 上記で使用したカイ二乗の分割表検定は、むしろ独立性の検定です。行が時間窓1と2を表し、列が特定の特徴量のカテゴリ毎の数を表すテーブルを取り込みます。これは、時間窓と特徴量の分布の間に関係があるかどうか、言い換えれば、分布が時間窓から独立しているかどうかを決定します。この検定では、分布の総計数の減少などの差は検出されないことに注意してください。これは、データ量が等しくない時間窓を比較する場合に便利ですが、必要に応じて欠損の数の変化や数の差を別途チェックするようにしてください。
# MAGIC 
# MAGIC いずれにしても、正しく機能するために、各カテゴリのビン数が多いこと(通常は5を超える)を想定しています。この例では、カテゴリの数が多いため、一部のビン数がこれらの検定で必要な数よりも低かったのです。幸いなことに、カイ二乗の分割表検定のscipy実装では、この状況では二次元が一次元よりも好ましい低カウントの補正を利用しますが、理想的にはさらに高いビン数が必要です。
# MAGIC 
# MAGIC Fisher Exact検定は、カウントが少なすぎる状況では良い代替手段ですが、現在、2x2より大きい分割表では、この検定に対するPythonの実装はありません。この検定を実行したい場合は、Rを使用する必要があります。
# MAGIC 
# MAGIC これらは考慮すべき微妙な違いですが、いずれにしても、p値が低い場合は、時間窓の間では分布が大きく異なること、一元カイ二乗または二元カイ二乗ではドリフトしていることを示しています。

# COMMAND ----------

# MAGIC %md <i18n value="d5a348a1-e123-4560-b1e3-09b29b9d4e28"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 一つのクラスにまとめる (Combine into One Class)
# MAGIC 
# MAGIC ここでは、これまで見てきた検定とコードを **`Monitor`** クラスに入れ、上記のコードを実際にどのように実装できるかを示します。

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
    
drift_monitor = Monitor(pdf1, pdf2, cat_cols, num_cols)
drift_monitor.run()

# COMMAND ----------

drift_monitor.generate_percent_change()

# COMMAND ----------

drift_monitor.generate_null_counts()

# COMMAND ----------

# MAGIC %md <i18n value="7dd9c6a3-8b89-46f4-a041-790fe2895ffc"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ドリフト監視アーキテクチャ (Drift Monitoring Architecture)
# MAGIC 
# MAGIC モデルのデプロイとドリフトの監視の参考ワークフローです。
# MAGIC 
# MAGIC ![Azure ML Pipeline](https://files.training.databricks.com/images/monitoring.png)
# MAGIC 
# MAGIC **ワークフロー**
# MAGIC * ***MLflowとDeltaを使用してモデルとデータをログに記録し、モデルを本番環境にデプロイします。***
# MAGIC * ***次のタイム・ステップのデータが到着したら：***
# MAGIC   * 現在の本番モデルからログされた入力データを取得します。
# MAGIC   * 観測 (真の) 値を取得します。
# MAGIC   * 観測値と予測値の評価指標 (例:RMSE) を比較します。
# MAGIC   * 上記の統計検定を実行し、潜在的なドリフトを特定します。
# MAGIC * ***ドリフトが見つからない場合：***
# MAGIC   * 監視を続けます。
# MAGIC * ***ドリフトが見つかった場合:***
# MAGIC   * 状況を分析してアクションを取ります。
# MAGIC   * モデルの再トレーニング/デプロイが必要な場合：
# MAGIC     * 新しいデータで候補モデルを構築します。
# MAGIC     * 最近のデータで本番モデルよりもパフォーマンスが良い場合は、候補モデルをデプロイします。

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="fecb11d3-918a-4449-8c94-1319dc74bc7f"/>
# MAGIC 
# MAGIC このレッスンでは、ドリフトを特定するための統計的手法に焦点を当てました。
# MAGIC 
# MAGIC ただし、他の方法もあります。
# MAGIC 
# MAGIC <a href="https://scikit-multiflow.github.io/" target="_blank">`skmultiflow`パッケージ には</a>、ドリフト検出アルゴリズムのいくつかの良いオプションがあります。DDMメソッドを試してみてください。
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/drift.png" style="height:400px; margin:20px"/></div>
# MAGIC 
# MAGIC 検出閾値は、`(pi+si)` が最小の場合に取得される2つの統計量の関数として計算されます。
# MAGIC 
# MAGIC  * `pmin`:記録された最低エラー率
# MAGIC  * `smin`:記録された最小標準偏差
# MAGIC 
# MAGIC インスタント`i`に、検出アルゴリズムは以下を使用します。
# MAGIC 
# MAGIC  * `pi`:インスタントiのエラー率
# MAGIC  * `si`:インスタントiでの標準偏差
# MAGIC 
# MAGIC 警告ゾーンに入り、変化を検出するためのデフォルトの条件は次のとおりです。
# MAGIC 
# MAGIC  *  `pi + si >= pmin + 2 * smin`の場合 -> 警告ゾーン
# MAGIC  * `pi + si >= pmin + 3 * smin`の場合 -> 変更検出
# MAGIC 
# MAGIC #### モデルベースのアプローチ (Model Based Approaches)
# MAGIC 
# MAGIC 直感的ではないが、より強力なアプローチを構築するためには機械学習ベースの監視ソリューションを考えられます。
# MAGIC 
# MAGIC 一般的な例をいくつか示します。
# MAGIC 
# MAGIC 1. 正常または異常として分類されたデータセットを使って教師ありアプローチを作成します。ただし、そのようなデータセットを見つけるのは容易ではありません。
# MAGIC 2. 回帰法を使用して経時的に入力データの将来の値を予測し、強い予測誤差がある場合はドリフト検出とします。

# COMMAND ----------

# MAGIC %md <i18n value="c5f29222-00d9-4b74-8842-aef5264dbdec"/>
# MAGIC 
# MAGIC 
# MAGIC 詳細については、Chengyin EngとNiall Turbittによる講演をご参考ください：<a href="https://databricks.com/session_na21/drifting-away-testing-ml-models-in-production" target="_blank">Drifting Away:Testing ML Models in Production</a>。
# MAGIC 
# MAGIC このレッスンは、この講演の内容から多く引用しています。
# MAGIC 
# MAGIC 2022年8月の時点で、Databricksには、分布の変化を監視し、モデルのパフォーマンスを追跡するモデルモニタリングサービスがあります。プライベート・プレビュー中です。
# MAGIC モデル・アサーションを使用してモデルを監視することに関心がある方は、この[ブログ投稿](https://www.databricks.com/blog/2021/07/22/monitoring-ml-models-with-model-assertions.html) を参考してください。
# MAGIC .

# COMMAND ----------

# MAGIC %md <i18n value="1074438b-a67b-401d-972d-06e70542c967"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 次のステップ
# MAGIC 
# MAGIC このレッスンのラボを実施します。 [Monitoring Lab]($./Labs/01-Monitoring-Lab) 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
