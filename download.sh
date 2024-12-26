#!/bin/bash

# DAIC-WOZをダウンロード
wget -r -l inf -A zip -nd -P ./data/raw -nc https://dcapswoz.ict.usc.edu/wwwdaicwoz/
cd ./data/raw
# 解凍するファイルをループで処理
for zip_file in *.zip; do
    # ZIPファイルを解凍する
    unzip "$zip_file" -d "${zip_file%.zip}"

    # 解凍したディレクトリ内の不要ファイルを削除
    find "${zip_file%.zip}" -type f ! \( -name '*.wav' -o -name '*_TRANSCRIPT.csv' \) -exec rm {} \;
done
cd ../..

echo "ダウンロードと解凍が完了しました。"