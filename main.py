import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"
from models.multimodal_model import Multimodal_Net
from train import train, multiple_train_experiments
from test import test
import torch
import torch.nn as nn
import torch.optim as optim
from models.report_supervision_model import ReportSupervisionModel

torch.cuda.empty_cache()
train_bool = True
test_bool = False
is_student = False # Ture表示使用知识蒸馏，False表示不使用
use_mad = False # True表示使用MAD_loss，False表示不使用MAD_loss(使用软标签KL_loss)
teacher_model_path = f"./checkpoints_teacher/img+rep_attention/experiment_5/best/best_model_auc.pth"
def main():
    # 定义训练参数
    root_path = '/data1/houhaotian/IDH/'  # 数据集路径ViT
    test_path = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/test_data_idh.xlsx' # 测试集表格
    batch_size = 4  # 批量大小 基本用满了显存
    transform = None  # 转换函数（如果有的话）
    
    num_epochs = 20  # 训练轮数 要注意修改
    num_experiments = 1 # 五次重复训练
    fc_type = 'img_only' # 要改输入模态的时候在这里修改 'img_only', 'rep_only', 'img+rep', 'rep_supervise'
    vit = True
    aggregation_type = 'ATTENTION' # 'ATTENTION, AVG ATTENTION表示用模拟报告特征再一次加权影像，再拼接。AVG就是直接拼接
    retrieval_type = 'cos_sim' # cos_sim, random, average
    loss_weights = {
    'classification': 1,  # 分类损失的权重
    'distillation': 0,    # 蒸馏损失的权重
    'report': 0        # 报告对照损失的权重
    }
    # 加载在验证集上表现最优的模型
    best_model_path = f"/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/checkpoints_teacher/img_only_vit/experiment_5/best/best_model_auc.pth"  # 根据实际情况调整路径 best_model_loss.pth,best_model_auc.pth,best_model_acc.pth
    # best_model_path = '/data1/houhaotian/PyCharmCode/SimulatorNet/checkpoints_teacher_attention/student/weights_classification0.2_distillation0.8_report0/best/best_model_acc.pth' # 复制路径
    # best_model_path = "/data1/houhaotian/PyCharmCode/SimulatorNet/checkpoints_detailedimg_only/best/best_model_loss.pth
    device = 'cuda:0'
    num_runs = 1 # 测试集执行次数
    
    report_criterion = torch.nn.MSELoss() 
    classification_criterion = nn.CrossEntropyLoss() # 分类损失
    distillation_criterion = nn.KLDivLoss(reduction='batchmean') # 蒸馏损失
    # num_folds = 5
    # test_size = 0.1 测试集是固定的
    
    # 调用训练函数
    if train_bool == True:
        print("Begin to train")
        print(f"fc_type is {fc_type}, is_student = {is_student}, use_mad = {use_mad}, loss_weights = {loss_weights}")
        # 调用多次实验的函数
        multiple_train_experiments(num_experiments, root_path, test_path, fc_type, num_epochs, batch_size, transform, classification_criterion, distillation_criterion, report_criterion, device, aggregation_type, teacher_model_path, is_student, loss_weights, use_mad, retrieval_type, vit)
        # train(root_path, test_path, num_epochs, batch_size, transform, classification_criterion, distillation_criterion, report_criterion, device, fc_type, aggregation_type, teacher_model_path, is_student, loss_weights, use_mad)
    if test_bool == True:
        print("Begin to test")
        print(f"fc_type is {fc_type}, is_student = {is_student}, use_mad = {use_mad}, loss_weights = {loss_weights}")
        test(root_path, test_path, best_model_path, batch_size, loss_weights, retrieval_type, device, aggregation_type, fc_type, is_student, num_runs, vit)

if __name__ == "__main__":
    main()
