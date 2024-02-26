import pandas as pd
import os
import random

# 定义类别对应的文件数量
required_samples = {0: 31, 1: 33} # 0是野生，1是突变

# 定义报告文件夹的路径
reports_folder = '/data1/houhaotian/IDH/TXT/'

# 初始化DataFrame的列
test_set = {'label': [], 'check_number': [], 'description': []}

# 对每个类别进行操作
for label in required_samples:
    # 获取指定类别的所有文件
    class_files = [f for f in os.listdir(reports_folder) if f.startswith(str(label) + '_')]
    # 随机选择需要的数量
    selected_files = random.sample(class_files, required_samples[label]) # 从class_files中选required_samples[label]个文件
    # 读取文件内容并填充DataFrame
    for file in selected_files:
        check_number = file.split('_')[1].replace('.txt', '')
        with open(os.path.join(reports_folder, file), 'r', encoding='utf-8') as f:
            description = f.read().strip()
        test_set['label'].append(label)
        test_set['check_number'].append(check_number)
        test_set['description'].append(description)

# 创建DataFrame
test_set_df = pd.DataFrame(test_set)

# 随机打乱数据
test_set_df = test_set_df.sample(frac=1).reset_index(drop=True)

# 输出路径
output_excel_path = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/test_data_idh.xlsx'
test_set_df.to_excel(output_excel_path, index=False)

print(f"Test set saved to {output_excel_path}")
