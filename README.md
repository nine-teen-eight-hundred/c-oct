# c-oct-dataset-converter

画像セグメンテーション応用研究のためのプログラム一式です。

# 簡単な説明

1. VOTT で作成したデータを準備します。-> (a)
2. `python vott-json-to-segmentation-dataset-6class.py vott-target-dir vott-source-dir center` で (a) を変換します。
3. `python pspnet-6-class.py train` でモデルをトレーニングします。
4. `python pspnet-6-class.py predict weights_file output_suffix` で推論を実行します。
5. `python pspnet-6-class.py evaluate weights_file bootstrap_repeats` でモデルの性能を評価します。

# 含まれているもの

### データセット生成ツール

画像セグメンテーションタスクの学習データセットを生成するツール。

- vott-json-to-segmentation-dataset-6class.py
- vott-json-to-segmentation-dataset-4class.py
- vott-json-to-segmentation-dataset-2class.py

### モデル並びに学習・評価プログラム

PSPNet ベースの画像セグメンテーションモデルと、学習・評価プログラムの実装。

- pspnet-6-class.py
- pspnet-4-class.py (WIP)

### 画像連結ツール

推論結果画像を視覚的に比較しやすいよう連結するツール。

- concat-images.py

# 含まれていないもの

### 画像データセット

別途 VOTT で作成したデータが必要です。
