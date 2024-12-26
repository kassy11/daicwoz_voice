import argparse
from logzero import logger
from utils import (
    get_transcript_files,
    get_voice_files,
    get_participant_segments,
    get_participant_voice,
)
from voice_opensmile import extract_opensmile_lld_feature
from voice_vggish import extract_vggish_feature
import os


def main(data_dir, no_extract_feature):
    raw_dir = os.path.join(data_dir, "raw")
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    feature_dir = os.path.join(data_dir, "feature")
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    logger.info(
        f"raw dir: {raw_dir}, preprocessed dir: {preprocessed_dir}, feature dir: {feature_dir}"
    )

    voice_files = get_voice_files(raw_dir)
    transcript_files = get_transcript_files(raw_dir)
    if len(voice_files) == 0 or len(transcript_files) == 0:
        logger.error("No data found. Please check the raw data directory.")
        raise ValueError("No raw data found. Please check the raw data directory.")
    elif len(voice_files) != len(transcript_files):
        logger.error("The number of voice files and transcript files are not equal.")
        raise ValueError(
            "The number of voice files and transcript files are not equal."
        )

    for voice_file, transcript_file in zip(voice_files, transcript_files):
        logger.info(
            f"Preprocess and feature extraction from {voice_file} and {transcript_file}"
        )
        voice_data_id = voice_file[0]
        voice_file_path = voice_file[1]
        transcript_data_id = transcript_file[0]
        transcript_file_path = transcript_file[1]
        logger.info(f"voice_data_id: {voice_data_id}")
        logger.info(f"transcript_data_id: {transcript_data_id}")
        logger.info(f"voice_file_path: {voice_file_path}")
        logger.info(f"transcript_file_path: {transcript_file_path}")

        if voice_data_id != transcript_data_id:
            logger.error(
                f"voice_data_id: {voice_data_id} != video_data_id: {transcript_data_id}"
            )
            raise ValueError("voice data_id and transcript data_id are not equal.")

        data_id = voice_data_id
        participant_segments = get_participant_segments(
            transcript_file_path, int(data_id)
        )
        voice_output_file_path = os.path.join(preprocessed_dir, f"{data_id}_AUDIO.wav")
        get_participant_voice(
            voice_file_path,
            participant_segments,
            voice_output_file_path,
        )
        if not no_extract_feature:
            extract_opensmile_lld_feature(
                voice_output_file_path,
                os.path.join(feature_dir, "opensmile"),
                data_id,
            )
            extract_vggish_feature(
                voice_output_file_path,
                os.path.join(feature_dir, "vggish"),
                data_id,
            )
        else:
            logger.info("Skip feature extraction.")
    logger.info("Preprocessing and feature extraction completed!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        help="Path to data dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--no_extract_feature", action="store_true", dest="no_extract_feature"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    no_extract_feature = args.no_extract_feature
    logger.info(f"Start preprocessing and feature extraction from {data_dir}")
    main(data_dir, no_extract_feature)
