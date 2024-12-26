import opensmile
import librosa
from logzero import logger
import pandas as pd
import os
from utils import save_feature

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,  # LLDを計算する
)


def _get_lld_per_sec(voice_file_path):
    """
    D-Vlogの元論文に記載されている方法でLLDを抽出する
    1秒毎にLLDsを抽出・平均化し、全てのLLDsを連結したものを特徴量とする
    frameStep, frameSizeの合わせ方は不明なので無視する
    """
    y, sr = librosa.load(voice_file_path, sr=None)
    duration = int(librosa.get_duration(y=y, sr=sr))

    lld_data = []
    # 1秒ごとに音声を処理する
    for i in range(duration):
        start_sample = int(i * sr)
        end_sample = int((i + 1) * sr)
        y_segment = y[start_sample:end_sample]
        # 音声セグメントが1秒に満たない場合はループを終了
        if len(y_segment) < sr:
            break
        lld_result = smile.process_signal(y_segment, sampling_rate=sr)
        lld_mean = lld_result.mean(axis=0)
        lld_data.append(lld_mean)
    lld_df = pd.DataFrame(lld_data)
    if duration != lld_df.shape[0]:
        logger.warning(f"duration: {duration} != lld feature column: {lld_df.shape[0]}")
    return lld_df


def extract_opensmile_lld_feature(voice_file_path, output_data_dir, data_id):
    """
    音声からopenSMILEのLLD特徴量を抽出する
    """
    logger.info(f"Extracting openSMILE LLD feature per sec from {voice_file_path}....")
    os.makedirs(output_data_dir, exist_ok=True)
    feature_per_sec = _get_lld_per_sec(voice_file_path)
    save_feature(feature_per_sec, output_data_dir, f"{data_id}.csv")
