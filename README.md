# daicwoz_voice

DAIC-WOZ で提供されている生音声データの前処理と特徴量抽出

## 実行手順

1. `download.sh`を実行して、DAIC-WOZデータをダウンロードする
2. `python voice.py`を実行して、生音声の前処理と特徴量抽出を行う
3. `python daicwoz_label.py`を実行して、ラベルを作成する

## 生音声の前処理と特徴量抽出 (`voice.py`)

音声書き起こしファイルに記載されている秒数をもとに、被験者の音声区間を特定しそれ以外の区間を無音にした音声を作成します。

音声書き起こしファイルの秒数がズレているデータがあるため、[adbailey1/daic_woz_process](https://github.com/adbailey1/daic_woz_process)を参考に秒数を補正しています。

その後、この前処理を行った音声からOpenSMILEの1秒ごとの特徴量とVGGishの特徴量を抽出します。VGGishはPyTorchでの実装である[harritaylor/torchvggish](https://github.com/harritaylor/torchvggish)を利用しています。


## ラベルの作成 (`daicwoz_label.py`)

DAIC-WOZで提供されているラベルに関する各CSVを結合し、モデル学習のために必要なラベルを作成します。