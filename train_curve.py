import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# 读取训练日志
log_path = r'C:\Code\My_GitHub_project\ST-MEM\output\ptbxl\raw_100_16-64_1e3_5\log.txt'
data = []

if not os.path.exists(log_path):
    print(f"找不到日志文件: {log_path}")
else:
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    df = pd.DataFrame(data)

    # 绘图设置 - 改为 2x2 布局
    plt.figure(figsize=(12, 10))

    # 子图1: Loss
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid Loss', marker='s')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2: Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['MulticlassAccuracy'], label='Accuracy', color='green', marker='^')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 子图3: AUROC
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['MulticlassAUROC'], label='AUROC', color='orange', marker='v')
    plt.title('Multiclass AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True)

    # 子图4: F1 Score (新增)
    plt.subplot(2, 2, 4)
    # 使用日志中的真实 key: MulticlassF1Score
    plt.plot(df['epoch'], df['MulticlassF1Score'], label='F1 Score', color='red', marker='D')
    plt.title('Multiclass F1 Score (Macro)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()