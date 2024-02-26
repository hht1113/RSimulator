import torch
import os
from data.dataset import BrainTumorDataset # ..表示上一级目录
from models.multimodal_model import Multimodal_Net 
from tqdm import tqdm

# 设置文件路径和模型路径
root_path = '/data1/houhaotian/IDH/'  # 数据集路径
best_model_path = f"../checkpoints_teacher/img+rep_attention/experiment_5/best/best_model_auc.pth"
device = 'cuda:0' # 设置gpu
test_path = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/test_data_idh.xlsx' # 测试集表格

# 加载模型
model = Multimodal_Net(fc_type = 'img+rep') # 在这里调整fc_type
model.load_state_dict(torch.load(best_model_path)) # 选用分类效果最好的模型
model.to(device)
model.eval() # 设置为评估模式

# 创建数据集实例
dataset = BrainTumorDataset(img_dir=root_path, report_dir=root_path)
train_set, _, _, _, _, _, _ = dataset.split_dataset(test_path, preprocessed_data_24_128_128_dir='/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/preprocessed_data_24_128_128')
print(f"训练集的数目是:{len(train_set)}")

# 创建目录来存储影像特征和报告特征 保存到两个文件夹下
img_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/IMG'
txt_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/TXT'
fusion_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/FUSION'
os.makedirs(img_feature_dir, exist_ok=True)
os.makedirs(txt_feature_dir, exist_ok=True)
os.makedirs(fusion_feature_dir, exist_ok=True)

# 遍历数据集并保存特征
for idx in tqdm(range(len(train_set)), desc="Extracting features"): 
    combined_volume_tensor, tokenized_report, label_tensor, filename = train_set[idx]
    
    img_feature_path = os.path.join(img_feature_dir, f"{filename}_IMG.pt")
    txt_feature_path = os.path.join(txt_feature_dir, f"{filename}_TXT.pt")
    fusion_feature_path = os.path.join(fusion_feature_dir, f"{filename}_FUSION.pt")
    
    # 检查是否已经存在，如果是，则跳过
    if os.path.exists(img_feature_path) and os.path.exists(txt_feature_path):
        continue
    # 后期可能把上两行代码注释掉，每次覆盖得到新的特征 因为可能用不同的教师模型提取特征

    # 加载数据和提取特征
    with torch.no_grad():
        combined_volume, tokenized_report, _, _ = train_set[idx] # 加载预处理过后的数据
        # 这个预处理对影像来说是维度统一与堆叠（3，24，256，256），对报告来说是tokenize
        # 输入数据移动到GPU
        combined_volume = combined_volume.to(device)
        combined_volume = combined_volume.unsqueeze(0) # 添加批数维度
        # 将字典中的每个张量移动到GPU
        tokenized_report = {k: v.to(device) for k, v in tokenized_report.items()}
        # 对于 tokenized_report，确保每个张量都添加了批次维度
        tokenized_report = {key: value.unsqueeze(0) for key, value in tokenized_report.items()}
    
        # 提取特征
        # 需要添加一个批数维度  对于单样本来说，通常需要用squeeze(0)补充批次维度
        report_features, _, _ = model.Rep_model(tokenized_report)  # 输出CLS_embedding (1，512)
        if model.img_aggregation_type=='ATTENTION':
            image_features, _, _ = model.Img_model(combined_volume, report_features) # 这里是加权后的影像特征 shape:(1,512)
        else:
            image_features = model.Img_model(combined_volume)  # 输出(1，512)
        fusion_features = torch.cat([image_features, report_features], dim=-1) # shape:(1,1024)

        # 移除批次的维度W
        image_features = image_features.squeeze(0) # 删除大小为1的维度，未指定dim，所以全部删除，这里是删除批次
        report_features = report_features.squeeze(0)
        fusion_features = fusion_features.squeeze(0)

        # 保存特征到磁盘 都是512维(fusion_features是1024)的特征，便于求余弦相似性
        torch.save(image_features.cpu(), img_feature_path)  
        torch.save(report_features.cpu(), txt_feature_path)
        torch.save(fusion_features.cpu(), fusion_feature_path)
