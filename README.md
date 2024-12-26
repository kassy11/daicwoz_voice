# daicwoz_voice

Preprocessing and feature extraction for raw voice data of DAIC-WOZ

## How to use

1. Run `download.sh` to download the DAIC-WOZ data
2. Run `python main.py` to preprocess the raw voice and extract features
3. Run `python daicwoz_label.py` to create labels

## Voice processing (`main.py`)

Based on the number of seconds listed in the audio transcript file, the participant's voice sections are identified and other sections are silenced to create audio.

Because the number of seconds in the audio transcript file is out of sync, correcting the number of seconds by referring to [adbailey1/daic_woz_process](https://github.com/adbailey1/daic_woz_process).

After this, OpenSMILE features per second and VGGish features are extracted from the preprocessed audio. For VGGish, using [harritaylor/torchvggish](https://github.com/harritaylor/torchvggish), a PyTorch implementation.


## Label processing (`daicwoz_label.py`)

Combine each CSV of labels provided by DAIC-WOZ to create the labels for model training.