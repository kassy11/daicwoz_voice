import glob
import os
import numpy as np
import pandas as pd
from logzero import logger
from pydub import AudioSegment
import re

# Copy from adbailey1/daic_woz_process
daicwoz_misaligned = {318: 34.32, 321: 3.84, 341: 6.19, 362: 16.86}


def get_participant_voice(
    audio_file_path,
    participant_segments,
    output_file_path,
):
    """
    Extract participant voice from audio file
    """
    logger.info(f"Extracting participant voice from {audio_file_path}....")
    audio = AudioSegment.from_file(audio_file_path)
    result_audio = AudioSegment.empty()

    for start_sec, end_sec in participant_segments:
        start_millisec = int(start_sec * 1000)
        end_millisec = int(end_sec * 1000)
        result_audio += audio[start_millisec:end_millisec]

    result_audio.export(output_file_path, format="wav")
    logger.info(f"Participant voice extracted to {output_file_path}")
    return


def get_participant_segments(transcript_csv_file_path, data_id: int):
    """
    Get participant segments from transcript file
    """
    try:
        transcript_df = pd.read_csv(transcript_csv_file_path, delimiter="\t")
    except Exception as e:
        logger.error(f"Error reading transcript file: {e}")
        raise

    if "speaker" not in transcript_df.columns:
        logger.error("'speaker' column not found in daicwoz transcript.")
        raise ValueError("'speaker' column missing in transcript.")
    participant_segments_df = transcript_df[
        transcript_df["speaker"].str.lower() == "participant"
    ]

    for col in ["start_time", "stop_time"]:
        if col not in participant_segments_df.columns:
            logger.error(f"Required column '{col}' not found in transcript.")
            raise ValueError(f"Column '{col}' missing in transcript.")

    if data_id in daicwoz_misaligned.keys():
        # Misaligned data
        logger.info(f"Misalignment found in {data_id}. Correcting time...")
        correction_time = daicwoz_misaligned[data_id]
        participant_segments_df.loc[:, "start_time"] += correction_time
        participant_segments_df.loc[:, "stop_time"] += correction_time
    return participant_segments_df[["start_time", "stop_time"]].values.tolist()


def get_voice_files(input_data_dir):
    """
    Get voice files from input data directory
    """
    pattern = os.path.join(input_data_dir, "*", "*_AUDIO.wav")
    file_paths = glob.glob(pattern, recursive=True)
    result = []
    for file_path in file_paths:
        data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
        match = re.match(r"(\d+)", data_id)
        if match:
            data_id = match.group(1)
        result.append((data_id, file_path))
    result.sort()
    return result


def get_transcript_files(input_data_dir):
    """
    Get  transcript files from input data directory
    """
    patterns = [
        os.path.join(input_data_dir, "*", "*_TRANSCRIPT.csv"),
        os.path.join(input_data_dir, "*", "*_Transcript.csv"),
    ]
    result = []
    for pattern in patterns:
        file_paths = glob.glob(pattern, recursive=True)
        for file_path in file_paths:
            data_id = os.path.relpath(file_path, input_data_dir).split(os.sep)[0]
            match = re.match(r"(\d+)", data_id)
            if match:
                data_id = match.group(1)
            result.append((data_id, file_path))
    result.sort()
    return result


def _save_as_npy(csv_file_path, output_dir):
    """
    Convert csv file to npy file
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1)
    npy_file_path = os.path.basename(os.path.splitext(csv_file_path)[0]) + ".npy"
    np.save(os.path.join(output_dir, npy_file_path), data)


def save_feature(feature: pd.DataFrame, output_dir, output_file_name):
    """
    Save feature to csv and npy file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, output_file_name)
    feature.to_csv(csv_file_path, index=False)
    _save_as_npy(csv_file_path, output_dir)
