import pandas as pd
import torch
import os
from logzero import logger
from utils import save_feature

# VGGishのPyTorch実装
# See: https://github.com/harritaylor/torchvggish
model = torch.hub.load("harritaylor/torchvggish", "vggish")
model.eval()


def extract_vggish_feature(voice_file_path, output_data_dir, data_id):
    """
    音声からVGGishの特徴量を抽出する
    """
    logger.info(f"Extracting VGGish feature from {voice_file_path}....")
    os.makedirs(output_data_dir, exist_ok=True)
    # VGGishの特徴量を取得
    feature = model.forward(voice_file_path)
    save_feature(
        pd.DataFrame(feature.detach().cpu().numpy()),
        output_data_dir,
        f"{data_id}.csv",
    )
