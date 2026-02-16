# Kaggle_Predicting-Heart-Disease

本プロジェクトでは、提供された `train.csv` を用いて\
**Heart Disease（心疾患）の発症確率を予測するベースラインモデル**
を構築します。

評価指標は **ROC-AUC** であり、`Presence`
の予測確率に基づいて評価されます。

------------------------------------------------------------------------

## 📌 概要

-   目的変数: `Heart Disease`
    -   `Presence` → 1
    -   `Absence` → 0
-   ID列: `id`
-   モデル: **ロジスティック回帰 (Logistic Regression)**
-   前処理:
    -   数値特徴量 → 標準化
    -   カテゴリ特徴量 → OneHotEncoding
-   評価:
    -   ホールドアウト ROC-AUC (80/20 split)
    -   5-Fold Stratified Cross Validation (OOF ROC-AUC)

------------------------------------------------------------------------

## 📂 想定ディレクトリ構成

    project/
    ├── data
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── model1.ipynb
    └── README.md

Notebook環境で実行する場合は、`train.csv`
が正しく配置されていることを確認してください。

------------------------------------------------------------------------

## 🚀 実行

実行すると以下が行われます。

1.  `train.csv` の読み込み
2.  数値列・カテゴリ列の自動判定
3.  ロジスティック回帰モデルの学習
4.  以下の指標を出力
    -   Holdout ROC-AUC
    -   5-Fold OOF ROC-AUC
5.  OOF予測結果を保存

保存先:

    /project/submission.csv

------------------------------------------------------------------------

## 📊 出力フォーマット

出力ファイルは以下の形式になります。

    id,Heart Disease
    630000,0.20
    630001,0.30
    630002,0.10
    ...

-   `id` : サンプルID
-   `Heart Disease` : Presence である確率（0〜1）

------------------------------------------------------------------------

## 🧠 モデル詳細

### 特徴量の自動判定ルール

-   dtype が `object` → カテゴリ扱い
-   数値型かつユニーク数 ≤ 12 → カテゴリ扱い
-   それ以外 → 数値特徴量

### 前処理パイプライン

**数値特徴量** - `SimpleImputer(strategy="median")` - `StandardScaler()`

**カテゴリ特徴量** - `SimpleImputer(strategy="most_frequent")` -
`OneHotEncoder(handle_unknown="ignore")`

------------------------------------------------------------------------

## 📈 評価方法

### ① ホールドアウト検証

80/20 の stratified split を使用します。

    ROC-AUC = roc_auc_score(y_valid, predicted_probabilities)

### ② 5-Fold Stratified CV

Out-of-Fold (OOF) 予測を生成し、全体AUCを計算します。

    5-Fold OOF ROC-AUC = roc_auc_score(y, oof_predictions)

より信頼性の高いベースライン指標となります。

------------------------------------------------------------------------

## 📦 必要ライブラリ

    pip install pandas numpy scikit-learn

------------------------------------------------------------------------

## 🎯 今後の改善案

本ベースラインはシンプルなモデルです。\
性能向上のためには以下が有効です。

-   LightGBM / XGBoost / CatBoost の導入
-   特徴量エンジニアリング
-   ハイパーパラメータチューニング
-   クラス不均衡対策
-   Target Encoding など高度なカテゴリ処理

------------------------------------------------------------------------

## 📄 ライセンス

本コードは学習・実験用途を目的としたベースライン実装です。
