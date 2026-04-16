import pandas as pd
import numpy as np
import os
import ast

def get_superclass_label(scp_dict_str):
    """
    从官方 scp_codes 中提取 5 大类 (Superclass) 标签:
    0: NORM (正常) | 1: MI (心肌梗死) | 2: STTC (ST段改变) | 3: CD (传导异常) | 4: HYP (肥厚)
    """
    try:
        scp_dict = ast.literal_eval(scp_dict_str)
        
        # 定义 5 大类的主要细分子类
        mi_list = ['AMI', 'IMI', 'LMI', 'PMI', 'ASMI', 'ALMI', 'IPLMI', 'IPMI', 'INJAS', 'INJAL', 'INJIN', 'INJLA', 'INJLP', 'INJPM']
        sttc_list = ['STTC', 'NST_AXL', 'NST_ISC', 'ISCAS', 'ISCAL', 'ISCAN', 'ISCLA', 'ISCLP', 'ISCPM', 'STRAS', 'STRAL', 'STRIN', 'STRLA', 'STRLP', 'STRPM']
        cd_list = ['CD', 'LBBB', 'RBBB', 'LAFB', 'LPFB', 'IVCD', 'WPW', '1AVB', '2AVB', '3AVB']
        hyp_list = ['HYP', 'LVH', 'RVH', 'SEHYP']
        
        # 优先检测异常
        for code in scp_dict.keys():
            if code in mi_list: return 1
            if code in sttc_list: return 2
            if code in cd_list: return 3
            if code in hyp_list: return 4
            
        # 如果没有上述异常，且包含 NORM，则判定为正常
        if 'NORM' in scp_dict: 
            return 0
            
        return -1 # 无法归为这 5 类的样本将被标记为 -1 并剔除
    except:
        return -1

def prepare_data():
    # --- 1. 路径配置 ---
    base_dir = r'C:\data\PTB_processed_500'       
    index_csv = r'C:\data\index_500.csv'          
    db_csv = r'C:\dataset\PTB-XL\ptbxl_database.csv' 
    
    print("Step 1: 正在扫描硬盘上的 .pkl 文件位置...")
    path_map = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pkl'):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                path_map[file] = rel_path

    # --- 2. 读取数据并提取标签 ---
    print("Step 2: 正在关联患者信息并提取 LABEL...")
    df = pd.read_csv(index_csv)
    official_df = pd.read_csv(db_csv)
    
    # 提取 ecg_id 
    df['ecg_id'] = df['FILE_NAME'].apply(lambda x: int(x.split('_')[0]))
    
    # 核心修复：在这里生成 LABEL 列！
    official_df['LABEL'] = official_df['scp_codes'].apply(get_superclass_label)
    
    # 提取我们需要的列：ecg_id (用于匹配), patient_id (用于拆分隔离), LABEL (用于训练)
    patient_data = official_df[['ecg_id', 'patient_id', 'LABEL']]
    
    # 将 patient_data 合并进我们的 index 总表
    merged_df = pd.merge(df, patient_data, on='ecg_id', how='left')
    
    # 修正文件相对路径
    merged_df['RELATIVE_FILE_PATH'] = merged_df['FILE_NAME'].map(path_map)
    
    # 剔除无法匹配到 5 大类的无效数据
    initial_len = len(merged_df)
    merged_df = merged_df[merged_df['LABEL'] != -1]
    print(f"-> 剔除无效标签后，数据量从 {initial_len} 变为 {len(merged_df)}")
    
    # 打印标签分布检查
    print("-> 标签分布统计:")
    print(merged_df['LABEL'].value_counts().sort_index())

    # --- 3. 按照患者 ID 划分 (70-10-20) ---
    print("Step 3: 正在按患者ID进行 70-10-20 划分...")
    unique_patients = merged_df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    n = len(unique_patients)
    train_pts = unique_patients[:int(n * 0.7)]
    valid_pts = unique_patients[int(n * 0.7):int(n * 0.8)]
    test_pts = unique_patients[int(n * 0.8):]
    
    # --- 4. 保存文件 ---
    splits = {
        'train.csv': merged_df[merged_df['patient_id'].isin(train_pts)],
        'valid.csv': merged_df[merged_df['patient_id'].isin(valid_pts)],
        'test.csv': merged_df[merged_df['patient_id'].isin(test_pts)]
    }
    
    for name, split_df in splits.items():
        # 移除训练不需要的辅助列，保留 RELATIVE_FILE_PATH, FILE_NAME, SAMPLE_RATE, LABEL 等
        final_df = split_df.drop(columns=['ecg_id', 'patient_id'])
        save_path = os.path.join(base_dir, name)
        final_df.to_csv(save_path, index=False)
        print(f"✅ 已生成 {name} | 样本数: {len(final_df)}")

if __name__ == "__main__":
    prepare_data()