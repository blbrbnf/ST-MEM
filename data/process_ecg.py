# Copyright (c) VUNO Inc. All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


_LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        required=True,
                        help="Path to the WFDB ECG database directory.")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_path',
                        type=str,
                        default="./index.csv",
                        help="Path to the index file.")
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories."""
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records


def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window."""
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i:i + crop_length] for i in start_idx]


# 【新增】辅助函数：从 wfdb 的 comments 中提取诊断标签
def extract_labels(comments):
    """
    从 .hea 文件的注释中提取诊断标签。
    PTB-XL 和 CPSC2018 等数据集通常使用 'Dx: label1, label2' 的格式记录。
    """
    if not comments:
        return []
    
    for comment in comments:
        # 寻找以 Dx: 开头的注释行
        if comment.startswith("Dx:"):
            # 提取 Dx: 后面的字符串，并按逗号分割
            dx_str = comment.split("Dx:")[1].strip()
            if not dx_str:
                return []
            # 清理空格并返回标签列表
            labels = [label.strip() for label in dx_str.split(",")]
            return labels
    return []


def run(args):
    # Identify the header files
    record_rel_paths = find_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} records.")

    # Prepare an index dataframe
    # 【新增】在索引中加入 LABEL 列，方便后续训练核对
    index_df = pd.DataFrame(columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE", "LABEL"])

    # Save all the cropped signals
    num_saved = 0
    num_skipped_multilabel = 0  # 【新增】用于统计因为多标签被跳过的样本数量

    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = os.path.join(args.output_dir, record_rel_dir)
        os.makedirs(save_dir, exist_ok=True)
        source_name = record_rel_dir.split("/")[0]
        
        # 读取信号和头文件信息
        signal, record_info = wfdb.rdsamp(os.path.join(args.input_dir, record_rel_path))
        
        # 【新增】获取该记录的 comments 并提取标签
        comments = record_info.get("comments", [])
        labels = extract_labels(comments)
        
        # 【核心新增】如果标签数量大于 1，则将其剔除（跳过本次循环）
        if len(labels) > 1:
            num_skipped_multilabel += 1
            continue
            
        # 如果没有标签或只有 1 个标签，我们提取出这个唯一的标签（用于保存到 index.csv 中）
        single_label = labels[0] if len(labels) == 1 else "UNKNOWN"

        lead_idx = np.array([record_info["sig_name"].index(lead_name) for lead_name in _LEAD_NAMES])
        signal = signal[:, lead_idx]
        fs = record_info["fs"]
        signal_length = record_info["sig_len"]
        
        if signal_length < 10 * fs:  # Exclude the ECGs with lengths of less than 10 seconds
            continue
            
        cropped_signals = moving_window_crop(signal.T, crop_length=10 * fs, crop_stride=10 * fs)
        
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                continue
            pd.to_pickle(cropped_signal.astype(np.float32),
                         os.path.join(save_dir, f"{record_name}_{idx}.pkl"))
            
            # 【修改】将单标签也保存到 csv 中
            index_df.loc[num_saved] = [f"{record_rel_path}_{idx}.pkl",
                                       f"{record_name}_{idx}.pkl",
                                       fs,
                                       source_name,
                                       single_label]
            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")
    print(f"Skipped {num_skipped_multilabel} records due to multiple labels.") # 【新增】打印剔除的样本数
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    index_df.to_csv(args.index_path, index=False)


if __name__ == "__main__":
    run(get_parser())