import torch
import torch.nn.functional as F

# 计算蒸馏损失，教师模型交互后的特征和学生模型的特征（单影像、用模拟报告特征交互后的）
# teacher_features和student_features均是attention融合之后的 shape:(16,512) 
def margin_aware_distillation(teacher_features, student_features, teacher_logits):
    # 计算教师和学生特征的余弦相似度
    # cos_sim((batch_size, 1, feature_dim),(1, batch_size, feature_dim))
    # 输出均为 batchsize*batchsize的矩阵,主对角元为1的对称矩阵
    sim_teacher = F.cosine_similarity(teacher_features.unsqueeze(1), teacher_features.unsqueeze(0), dim=2)
    sim_student = F.cosine_similarity(student_features.unsqueeze(1), student_features.unsqueeze(0), dim=2)
    
    # 计算两个相似度矩阵之间的差异并按行求和 shape:(16,)
    sim_diff = (sim_teacher - sim_student).sum(dim=1)
    
    # 计算教师模型的softmax概率和熵
    # shape: (16,3)
    probs_teacher = F.softmax(teacher_logits, dim=1)
    # 手动实现熵计算 sum(-pi*log(pi)) shape:(16,)
    entropy_teacher = -torch.sum(probs_teacher * torch.log(probs_teacher + 1e-5), dim=1)
    
    # 计算加权的知识蒸馏损失 每个样本差异的均值
    mad_loss = (entropy_teacher * sim_diff).mean()
    
    return mad_loss

# 不加权熵值（即直接使用特征相似度差异的均值作为损失）
def mad_loss_unweighted(teacher_features, student_features, teacher_logits=None):
    sim_teacher = F.cosine_similarity(teacher_features.unsqueeze(1), teacher_features.unsqueeze(0), dim=2)
    sim_student = F.cosine_similarity(student_features.unsqueeze(1), student_features.unsqueeze(0), dim=2)
    sim_diff = (sim_teacher - sim_student).sum(dim=1)
    mad_loss = sim_diff.mean()
    return mad_loss

# 加权熵值：熵值越大的样本权重越大
def mad_loss_entropy_weighted(teacher_features, student_features, teacher_logits):
    sim_teacher = F.cosine_similarity(teacher_features.unsqueeze(1), teacher_features.unsqueeze(0), dim=2)
    sim_student = F.cosine_similarity(student_features.unsqueeze(1), student_features.unsqueeze(0), dim=2)
    sim_diff = (sim_teacher - sim_student).sum(dim=1)
    probs_teacher = F.softmax(teacher_logits, dim=1)
    entropy_teacher = -torch.sum(probs_teacher * torch.log(probs_teacher + 1e-5), dim=1)
    mad_loss = (entropy_teacher * sim_diff).mean()
    return mad_loss

# 加权熵值：熵值越小的样本权重越大
def mad_loss_confidence_weighted(teacher_features, student_features, teacher_logits, max_entropy=torch.log(torch.tensor(3.0))):
    sim_teacher = F.cosine_similarity(teacher_features.unsqueeze(1), teacher_features.unsqueeze(0), dim=2)
    sim_student = F.cosine_similarity(student_features.unsqueeze(1), student_features.unsqueeze(0), dim=2)
    sim_diff = (sim_teacher - sim_student).sum(dim=1)
    probs_teacher = F.softmax(teacher_logits, dim=1)
    entropy_teacher = -torch.sum(probs_teacher * torch.log(probs_teacher + 1e-5), dim=1)
    # 由于广播机制，max_entropy会扩展到batchsize维度
    confidence_weights = max_entropy - entropy_teacher
    mad_loss = (confidence_weights * sim_diff).mean()
    return mad_loss
