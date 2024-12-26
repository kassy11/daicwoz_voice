#!/bin/bash

wget -r -l inf -A zip -nd -P ./data/raw -nc https://dcapswoz.ict.usc.edu/wwwdaicwoz/
cd ./data/raw
for zip_file in *.zip; do
    unzip "$zip_file" -d "${zip_file%.zip}"
    find "${zip_file%.zip}" -type f ! \( -name '*.wav' -o -name '*_TRANSCRIPT.csv' \) -exec rm {} \;
done
cd ../..

echo "Finish downloading DAIC-WOZ dataset (only wav and transcript files are kept)"