# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d4db55e1-8962-4f27-b122-eeedd7089ae7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # ラボ: 高度な実験追跡 (Lab: Advanced Experiment Tracking)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このラボでは、次のことを実施します。<br>
# MAGIC  - <a href="https://www.mlflow.org/docs/latest/tracking.html" target="_blank">MLflow Tracking</a>を使って手動でハイパーパラメーターのチューニングのrunを入れ子実行（nested run）としてログに記録します。
# MAGIC  - <a href="http://hyperopt.github.io/hyperopt/" target="_blank">hyperopt</a> を入れ子実行（nested run）を自動的にログに記録します。

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="b0833c9c-68a1-4193-9740-2f4e02d299a5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 手動でハイパーパラメータチューニング (Manual Hyperparameter Tuning)
# MAGIC 
# MAGIC 次の方法でmlflowのrun構造を作成します。
# MAGIC 
# MAGIC * **`parent`** という名前の親runを作成します。
# MAGIC * この親runでは:
# MAGIC   * sklearn RandomForestRegressorを **`X_Train`** と **`y_train`** でトレーニングします。
# MAGIC   * Signatureと入力例を取得します。( **`infer_signature`** でsignatureを取得します)
# MAGIC * **`child_1`** という名前の入れ子実行（nested run）を作成します。
# MAGIC   * **`child_1`** では、　
# MAGIC     * sklearn RandomForestRegressorを **`X_TRAIN`** と **`y_train`** でmax_depth = 5を設定してトレーニングします。
# MAGIC     * 「max_depth」パラメータをログに記録します。
# MAGIC     * mseをログに記録します。
# MAGIC     * モデルと一緒に入力例とsignatureをログに記録します。 
# MAGIC * **`child_2`** という名前の別の入れ子実行（nested run）を作成します。
# MAGIC   * **`child_2`** では、
# MAGIC     * sklearn RandomForestRegressorを **`X_TRAIN`** と **`y_train`** でmax_depth=10を設定してトレーニングします。
# MAGIC     * 「max_depth」パラメータをログに記録します。
# MAGIC     * mseをログに記録します。
# MAGIC     * モデルと一緒に入力例とsignatureをログに記録します。 
# MAGIC     * モデルの特徴量重要度プロットを生成してログに記録します。 (ヒントが必要な場合はデモを見てください)

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

pdf = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
X = pdf.drop("price", axis=1)
y = pdf["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# TODO
import mlflow 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import numpy as np 
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="parent") as run:
  
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    input_example = X_train.head(3)
  
    with mlflow.start_run(run_name="child_1", nested=#FILL_IN):
        max_depth = 5
        rf = RandomForestRegressor(random_state=42, max_depth=max_depth)
        rf_model = rf.fit(X_train, y_train)
        mse = mean_squared_error(rf_model.predict(X_test), y_test)
        ## log the max_depth parameter                  
        # FILL_IN
        ## log the mse metric
        # FILL_IN
        ## log model
        # FILL_IN

    with mlflow.start_run(run_name="child_2", nested=#FILL_IN):
        max_depth = 10
        rf = RandomForestRegressor(random_state=42, max_depth=max_depth)
        rf_model = rf.fit(X_train, y_train)
        mse = mean_squared_error(rf_model.predict(X_test), y_test)
        ## log the max_depth parameter                  
        # FILL_IN
        ## log the mse metric
        # FILL_IN
        ## log model
        # FILL_IN

        # Generate feature importance plot
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")

        # Log figure
        # FILL_IN

# COMMAND ----------

# MAGIC %md <i18n value="ef7fafd4-ac6f-4e20-8ae0-e427e372ce92"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Hyperoptでオートロギング (Autologging with Hyperopt)
# MAGIC 
# MAGIC この演習では、HyperOptを使用してsklearnランダムフォレストモデルのハイパーパラメーターをチューニングします。
# MAGIC 
# MAGIC 以下のステップで操作します。
# MAGIC 
# MAGIC 1. HyperOptを使用して、sklearnランダムフォレストで **`n_estimators`** と **`max_depth`** をチューニングします。
# MAGIC   * **`n_estimators`** :50～500に、間隔を10に設定します。
# MAGIC     * <a href="http://hyperopt.github.io/hyperopt/getting-started/search_spaces/" target="_blank">このドキュメント</a>を参考して適当な範囲を見つけてください。
# MAGIC   * **`max_depth`** :5～15に、間隔を1に設定します。 
# MAGIC     * <a href="http://hyperopt.github.io/hyperopt/getting-started/search_spaces/" target="_blank">このドキュメント</a>を参考して適当な範囲を見つけてください。
# MAGIC   * **`parallelism`** :2
# MAGIC   * **`max_evals`** :16
# MAGIC 2. MLflow UIで入れ子実行（nested run）を見つけます。
# MAGIC 3. 複数nested runの“Parallel Coordinates Plot”を生成します。
# MAGIC 
# MAGIC **注:** 
# MAGIC - MLflow UIですべての入れ子実行（nested run）を選択し、「compare」を押す必要があります。
# MAGIC - 一番下の入れ子実行（nested run）を選択し、Shiftキーを押しながら一番上の入れ子実行（nested run）をクリックすると、すべての入れ子実行（nested run）が選択されます。

# COMMAND ----------

# TODO 
from hyperopt import fmin, tpe, hp, SparkTrials 

# Define objective function
def objective(params):
    # build a Random Forest Regressor with hyperparameters
    model = RandomForestRegressor(#FILL_IN)

    # fit model with training data
    model.fit(X_train, y_train)

    # predict on testing data
    pred = model.predict(#FILL_IN)

    # compute mean squared error
    score = #FILL_IN
    return score

# COMMAND ----------

# TODO
# Define search space
search_space = #FILL_IN

# Set algorithm type
algo = tpe.suggest
# Create SparkTrials object
spark_trials = SparkTrials(parallelism=#FILL_IN)

# start run
with mlflow.start_run(run_name="Hyperopt"):
    argmin = fmin(#FILL_IN)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
