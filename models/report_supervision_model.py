import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.image_model import ImageEncoder  # 假设这是您的ImageEncoder类
from sentence_transformers import SentenceTransformer, util
from models.multimodal_model import Multimodal_Net  
import random

teacher_model_path = f"./checkpoints_teacher/img+rep_attention/experiment_5/best/best_model_auc.pth"
class ReportSupervisionModel(nn.Module):
    def __init__(self, img_feature_dir, rep_feature_dir, aggregation_type, device, retrieval_type):
        super(ReportSupervisionModel, self).__init__()
        self.img_feature_dir = img_feature_dir
        self.rep_feature_dir = rep_feature_dir
        self.fc = nn.Linear(1024, 2)  # 全连接层用于分类
        self.device = device
        self.retrieval_type = retrieval_type

        # 加载教师模型的影像编码器
        teacher_model = Multimodal_Net(fc_type='img+rep')
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        teacher_model.eval()
        self.teacher_img_model = teacher_model.Img_model
        self.teacher_img_model.eval()

        # 创建学生模型实例
        self.student_img_model = ImageEncoder(
            image_model='resnet18',
            aggregation_type=aggregation_type,
            H=128, W=128, D=24,       # 根据您的数据集来设置
            channels=5,               # 使用5通道图像
            mm_dim=512,               # 设置多模态维度
            num_class=2,              # 分类的类别数
            num_heads=8,              # 注意力机制的头数
            bias=False,               # 是否使用偏置
            dropout_rate=0.1,         # Dropout率
            mask_columns=2            # 遮罩列
        )
        
        # 获取教师模型的影像编码器的状态字典
        teacher_img_model_state_dict = teacher_model.Img_model.state_dict()
        
        # 由于结构可能不同，只加载匹配的键
        student_img_model_state_dict = self.student_img_model.state_dict()
        filtered_teacher_state_dict = {k: v for k, v in teacher_img_model_state_dict.items() if k in student_img_model_state_dict}
        student_img_model_state_dict.update(filtered_teacher_state_dict)

        # 加载过滤后的状态字典到学生模型
        self.student_img_model.load_state_dict(student_img_model_state_dict)
        self.student_img_model.train()

        self.aggregation_type = aggregation_type

        # 小型网络
        self.simulation_network = nn.Sequential(
            nn.Linear(512, 512),  # 第一个全连接层
            nn.ReLU(),            # 激活函数
            nn.Dropout(0.1),      # Dropout防止过拟合
            nn.Linear(512, 512)   # 第二个全连接层，输出维度为512
        )

    def retrieve_and_process(self, rep_features_dict, img_features_dict, image_features_batch, filenames):
        # 将所有影像特征转换为张量列表，并检查每个元素确实是一个张量
        img_features_list = [img_features_dict[key] for key in img_features_dict.keys() if isinstance(img_features_dict[key], torch.Tensor)]
        rep_features_list = [rep_features_dict[key] for key in rep_features_dict.keys() if isinstance(rep_features_dict[key], torch.Tensor)]

        # 将列表转换为2D tensor
        img_features_tensor = torch.stack(img_features_list)  # 2D tensor of image features
        rep_features_tensor = torch.stack(rep_features_list)  # 报告特征的2D张量

        # 对批次中的每个影像特征计算与其最相似的前10个影像特征
        batch_topk_rep_filenames = []
        batch_simulated_report_features = []

        img_features_tensor = img_features_tensor.to(self.device)
        rep_features_tensor = rep_features_tensor.to(self.device)
        image_features_batch = image_features_batch.to(self.device)

        for image_feature, filename in zip(image_features_batch, filenames):
            if self.retrieval_type == 'cos_sim':
                # 计算与单个影像特征最相似的前10个影像特征
                similar_indices = util.semantic_search(image_feature.unsqueeze(0), img_features_tensor, top_k=10)[0]
                topk_img_filenames = [list(img_features_dict.keys())[idx['corpus_id']] for idx in similar_indices]
                topk_img_features = [img_features_dict[fname] for fname in topk_img_filenames]
                topk_rep_filenames = [fname.replace("_IMG.pt", "_TXT.pt") for fname in topk_img_filenames]
                topk_rep_features = [rep_features_dict[fname] for fname in topk_rep_filenames]
                simulated_report_feature = self.attention(image_feature.unsqueeze(0), topk_img_features, topk_rep_features).squeeze(0)
            elif self.retrieval_type == 'random':
                # 随机检索10个影像和对应报告
                topk_img_filenames = random.sample(list(img_features_dict.keys()), 10)
                topk_img_features = [img_features_dict[fname] for fname in topk_img_filenames]
                topk_rep_filenames = [fname.replace("_IMG.pt", "_TXT.pt") for fname in topk_img_filenames]
                topk_rep_features = [rep_features_dict[fname] for fname in topk_rep_filenames]
                simulated_report_feature = self.attention(image_feature.unsqueeze(0), topk_img_features, topk_rep_features).squeeze(0)
            elif self.retrieval_type == 'average':
                # 使用所有报告特征的均值作为模拟报告特征
                simulated_report_feature = torch.mean(rep_features_tensor, dim=0)
                topk_rep_filenames = [None] * 10  # 使用None作为占位符
            else:
                raise ValueError(f"Unknown retrieval type: {self.retrieval_type}")

            # 添加模拟报告特征和文件名到相应的批次列表
            batch_simulated_report_features.append(simulated_report_feature)
            batch_topk_rep_filenames.append(topk_rep_filenames)

        # 将批次列表转换为张量
        batch_simulated_report_features = torch.stack(batch_simulated_report_features)
        return batch_simulated_report_features, batch_topk_rep_filenames
        
    def attention(self, query, keys, values): #(1,512);列表，每个元素是(512);列表，每个元素是(512)
        # 确保所有张量都在同一设备上
        query = query.to(self.device)
        keys = [k.to(self.device) for k in keys]
        values = [v.to(self.device) for v in values]

        # 堆叠它们
        keys = torch.stack(keys)  # shape: (10, 512)
        values = torch.stack(values)  # shape: (10, 512)

        query = query.unsqueeze(1)  # shape: (1, 1, 512)
        attention_scores = torch.bmm(query, keys.unsqueeze(0).transpose(1, 2))  # shape: (1, 1, 10)
        # 使用批处理矩阵乘法，转换顺序以对齐尺寸
   
        # 应用softmax得到权重 
        attention_weights = F.softmax(attention_scores, dim=-1)  # shape: (1, 1, 10)

        # 用权重加权values得到模拟报告特征 shape: (16,1,512)
        weighted_sum = torch.bmm(attention_weights, values.unsqueeze(0))  # shape: (1, 1, 512)

        # 移除添加的维度 得到模拟报告特征
        simulated_report_feature = weighted_sum.squeeze(0).squeeze(0)  # shape: (512,)
        return simulated_report_feature  # shape: (512,)

    def forward(self, combined_volume, filename): # 传入当前预处理后的影像和影像名称 'AVG', '3D', 'COS_SIM', 'ATTENTION'
        # 提取影像特征
        image_feature_teacher = self.teacher_img_model(combined_volume) # 得到(16,512)的影像特征 第二个参数(报告特征)没有时，就是切片平均
        image_feature_student = self.student_img_model(combined_volume)

        # 加载并处理特征
        img_features_dict = load_features(self.img_feature_dir)
        rep_features_dict = load_features(self.rep_feature_dir)
        # 模拟报告特征 shape: (16, 512)
        simulated_report_feature, batch_topk_rep_filenames = self.retrieve_and_process(rep_features_dict, img_features_dict, image_feature_teacher, filename)

        # 使用小型网络处理模拟报告特征
        simulated_report_feature = self.simulation_network(simulated_report_feature)

        # 特征拼接 shape: (16, 512*2)
        if self.aggregation_type == 'ATTENTION':
            image_feature_student, _, _ = self.student_img_model(combined_volume, simulated_report_feature) # 第二个参数是模拟的报告特征，可以得到加权后的影像特征 shape:(16,512)
            fused_feature = torch.cat((image_feature_student, simulated_report_feature), dim=1)
        elif self.aggregation_type == 'AVG':
            fused_feature = torch.cat((image_feature_student, simulated_report_feature), dim=1)

        # 全连接层
        logits = self.fc(fused_feature) 
        # logits shape: (16, 3) 模拟报告特征 shape: (16, 512) 
        return logits, simulated_report_feature, rep_features_dict, batch_topk_rep_filenames, fused_feature # 嵌套列表,(16,10)

# 读入特征值保存为字典{特征名：特征值}
def load_features(feature_dir):
    feature_dict = {}
    # 排序
    sorted_filenames = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
    # 导入教师模型提取的特征
    for filename in sorted_filenames:
        path = os.path.join(feature_dir, filename)
        feature = torch.load(path)
        feature_dict[filename] = feature
    return feature_dict
