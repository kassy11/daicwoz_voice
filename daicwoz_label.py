import argparse
from logzero import logger
import pandas as pd

PHQ8_CUTOFF_POINT = 10


def main(train_csv_path, dev_csv_path, test_csv_path, output_csv_path):
    train_df = pd.read_csv(train_csv_path)
    dev_df = pd.read_csv(dev_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df = train_df[["Participant_ID", "PHQ8_Score"]]
    dev_df = dev_df[["Participant_ID", "PHQ8_Score"]]
    test_df = test_df[["Participant_ID", "PHQ_Score"]]

    train_df.columns = ["index", "label"]
    dev_df.columns = ["index", "label"]
    test_df.columns = ["index", "label"]
    # label列を10以上で1、それ以外は0に変換
    train_df["label"] = train_df["label"].apply(
        lambda x: 1 if x >= PHQ8_CUTOFF_POINT else 0
    )
    dev_df["label"] = dev_df["label"].apply(
        lambda x: 1 if x >= PHQ8_CUTOFF_POINT else 0
    )
    test_df["label"] = test_df["label"].apply(
        lambda x: 1 if x >= PHQ8_CUTOFF_POINT else 0
    )

    train_df["fold"] = "train"
    dev_df["fold"] = "valid"
    test_df["fold"] = "test"

    result_df = pd.concat([train_df, dev_df, test_df])
    result_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv_path",
        help="Path to train csv",
        type=str,
        default="./data/label/train_split_Depression_AVEC2017.csv",
    )
    parser.add_argument(
        "--dev_csv_path",
        help="Path to train csv",
        type=str,
        default="./data/label/dev_split_Depression_AVEC2017.csv",
    )
    parser.add_argument(
        "--test_csv_path",
        help="Path to train csv",
        type=str,
        default="./data/label/full_test_split.csv",
    )
    parser.add_argument(
        "--output_csv_path",
        help="Path to train csv",
        type=str,
        default="./data/label/daicwoz_label.csv",
    )
    args = parser.parse_args()
    train_csv_path = args.train_csv_path
    dev_csv_path = args.dev_csv_path
    test_csv_path = args.test_csv_path
    output_csv_path = args.output_csv_path
    logger.info(
        f"Creating label from train_csv_path: {train_csv_path}, dev_csv_path: {dev_csv_path}, test_csv_path: {test_csv_path}"
    )
    main(train_csv_path, dev_csv_path, test_csv_path, output_csv_path)
