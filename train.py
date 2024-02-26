import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from data.dataset import BrainTumorDataset
from models.multimodal_model import Multimodal_Net  
from transformers import AutoTokenizer
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from data.dataset import BrainTumorDataset
from tqdm import tqdm
from utils.metrics import calculate_metrics, plot_losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from models.report_supervision_model import ReportSupervisionModel
from utils.losses import mad_loss_unweighted, mad_loss_entropy_weighted, mad_loss_confidence_weighted
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

def multiple_train_experiments(num_experiments, root_path, test_path, fc_type, num_epochs, batch_size, transform, classification_criterion, distillation_criterion, report_criterion, device, aggregation_type, teacher_model_path, is_student, loss_weights, use_mad, retrieval_type,vit):
    start_experiment = 1  # 默认从第一个实验开始

    # 基于损失权重定义weight_str
    weight_str = "_".join(f"{key}{value}" for key, value in loss_weights.items()) if is_student or fc_type=='rep_supervise' else ""
    # 检查每个实验的检查点目录以确定从哪个实验开始或重新开始
    for i in range(1, num_experiments + 1):
        if is_student==False:
            checkpoint_dir = os.path.join(f"./checkpoints_teacher/{fc_type}", f"weights_{weight_str}", f"experiment_{i}")
        else:
            if use_mad:
                checkpoint_dir = os.path.join(f"./checkpoints_teacher/student_madloss", f"weights_{weight_str}", f"experiment_{i}")
            else:
                checkpoint_dir = os.path.join(f"./checkpoints_teacher/student", f"weights_{weight_str}", f"experiment_{i}")
        print(f"Checking for the existence of {checkpoint_dir}...")  # Debug print
        if not os.path.exists(checkpoint_dir):
            # 找到了未开始的实验，从这里开始
            start_experiment = i
            break
        else:
            # 检查该实验是否已完成
            epoch_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch") and os.path.isdir(os.path.join(checkpoint_dir, d))]
            if not epoch_dirs:
                # 没有epoch子目录，需要从此实验开始或重新开始
                print(f"No checkpoint directory found for experiment {i}, will start this experiment.")  # Debug print
                start_experiment = i
                break
            else:
                # 如果有epoch目录，检查是否已经达到预定的epoch数
                epoch_numbers = sorted([int(d.replace("epoch", "")) for d in epoch_dirs])
                if epoch_numbers[-1] != num_epochs - 1:
                    # 未达到预定的epoch数，从此实验开始或重新开始
                    start_experiment = i
                    break

    # 开始或继续剩余的实验
    for i in range(start_experiment, num_experiments + 1):
        print(f"Starting training for experiment {i}/{num_experiments}")
        train(root_path, test_path, fc_type, num_epochs, batch_size, transform, classification_criterion, distillation_criterion, report_criterion, device, aggregation_type, teacher_model_path, is_student, loss_weights, use_mad, retrieval_type, experiment_number=i, vit=vit)

def train(root_path, test_path, fc_type, num_epochs, batch_size, transform, classification_criterion, distillation_criterion, report_criterion, device, aggregation_type, teacher_model_path, is_student, loss_weights, use_mad, retrieval_type, experiment_number, vit):
    # 函数实现...
    # torch.cuda.empty_cache()
    # 保存特征的路径
    img_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/IMG'  # 替换为实际的影像特征文件夹路径
    rep_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/TXT'   # 替换为实际的文本特征文件夹路径
    fusion_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/FUSION'   # 替换为实际的文本特征文件夹路径
    # 初始化数据集
    dataset = BrainTumorDataset(img_dir=root_path, report_dir=root_path, transform=transform)
    # 预处理和保存数据
    dataset.save_preprocessed_data()
    # 划分数据集
    train_set, val_set, test_set, total_train_samples, total_val_samples, train_label_counts, val_label_counts = dataset.split_dataset(test_path) # 另一个参数取默认值
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # 输出数据集信息
    print(f"Total training samples: {total_train_samples}")
    print(f"Training samples per label: {train_label_counts}")
    print(f"Total validation samples: {total_val_samples}")
    print(f"Validation samples per label: {val_label_counts}")

    # 为每个实验创建唯一的检查点基础路径
    checkpoint_base_path = f"./checkpoints_teacher"
    os.makedirs(checkpoint_base_path, exist_ok=True)

    teacher_model = Multimodal_Net(fc_type='img+rep').to(device)
    # 初始化模型并移动到GPU 报告监督模型用ReportSupervisionModel
    if fc_type == 'rep_supervise' and is_student:
        model = ReportSupervisionModel(img_feature_dir, rep_feature_dir, aggregation_type, device, retrieval_type).to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_path))
        teacher_model.eval()
        checkpoint_sub_path  = "student+rep_supervise"
    elif fc_type == 'rep_supervise':
        model = ReportSupervisionModel(img_feature_dir, rep_feature_dir, aggregation_type, device, retrieval_type).to(device)
        if retrieval_type == 'cos_sim':
            checkpoint_sub_path = "rep_supervise_cos_sim"
        elif retrieval_type == 'random':
            checkpoint_sub_path = "rep_supervise_random"
        elif retrieval_type == 'average':
            checkpoint_sub_path = "rep_supervise_avg"
    elif is_student and use_mad:
        # model = Multimodal_Net(fc_type='img_only').to(device)
        # teacher_model.load_state_dict(torch.load(teacher_model_path))
        # teacher_model.eval()
        # checkpoint_sub_path = "student_madloss"
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        teacher_model.eval()
        model = Multimodal_Net(fc_type='img_only').to(device) # 创建单影像学生模型
        # 提取教师模型的影像编码器的参数
        teacher_image_encoder_params = {k: v for k, v in teacher_model.Img_model.state_dict().items()}
        # 加载教师模型影像编码器的参数到学生模型中
        model.load_state_dict(teacher_image_encoder_params, strict=False)
        # 设置学生模型为训练模式
        model.train()
        checkpoint_sub_path = "student_madloss"
    elif is_student:
        model = Multimodal_Net(fc_type='img_only').to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_path))
        teacher_model.eval()
        checkpoint_sub_path = "student"
    else:
        model = Multimodal_Net(fc_type=fc_type).to(device)
        if model.img_aggregation_type == 'AVG':
            checkpoint_sub_path = fc_type + '_' + model.img_aggregation_type + '-' + 'non_interaction'
        elif model.img_aggregation_type == 'ATTENTION':
            checkpoint_sub_path = fc_type + '_' + model.img_aggregation_type+ '-'+'interaction'
        elif model.img_aggregation_type == '3D':
            checkpoint_sub_path = fc_type + '_'+ model.img_aggregation_type + '_'+ 'non_interaction'
        elif model.img_aggregation_type == 'ViT':
            model = Multimodal_Net(fc_type=fc_type,
                        input_W=224,  # width of slice
                        input_H=224,  # height of slice
                        input_D=24,  # slice number
                        ).to(device)
            checkpoint_sub_path = fc_type + '_vit'
    # 如果有权重字符串，则加入路径中
    weight_str = "_".join(f"{key}{value}" for key, value in loss_weights.items()) if is_student or fc_type=='rep_supervise' else ""
    weight_folder = f"weights_{weight_str}" if weight_str else ""

    # 创建最终的检查点路径
    checkpoint_path = os.path.join(checkpoint_base_path, checkpoint_sub_path, weight_folder, f"experiment_{experiment_number}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # 统一的优化器
    # 优化器 lr太高容易错过最小值，太低容易得到局部最小值权重衰减防止过拟合 （L2正则化）
    lr = 1e-2
    weight_decay = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"learning rate setting:{lr}, weight_decay:{weight_decay}")
    # 创建 ReduceLROnPlateau 学习率调度器实例
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.75, min_lr=0) # 5个epoch指标不下降的话，lr降为原来的四分之三

    # 重新开始训练前的检查最新epoch的逻辑
    start_epoch = 0
    print(f"here is checkpoint_path:{checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print("find last epoch")
        epoch_dirs = [d for d in os.listdir(checkpoint_path) if d.startswith("epoch") and os.path.isdir(os.path.join(checkpoint_path, d))]
        if epoch_dirs:
            epoch_numbers = sorted([int(d.replace("epoch", "")) for d in epoch_dirs], reverse=True)
            start_epoch = epoch_numbers[0]  # 取最大的编号
            latest_model_path = os.path.join(checkpoint_path, f"epoch{start_epoch}", "model.pth")
            model.load_state_dict(torch.load(latest_model_path))
    
    print(f"Starting training from epoch {start_epoch} for experiment {experiment_number}")

    # 训练和验证循环
    best_loss = float('inf')
    best_loss_epoch = 0
    best_auc = 0.0
    best_auc_epoch = 0
    best_acc = 0.0
    best_acc_epoch = 0
    train_losses = []
    val_losses = []
    experiment_checkpoint_path = f"{checkpoint_path}"
    best_model_path = f"{experiment_checkpoint_path}/best"

    # 检查是否存在先前的最佳指标，并加载它们
    best_loss_file = f"{best_model_path}/best_loss_metrics.txt"
    best_auc_file = f"{best_model_path}/best_auc_metrics.txt"
    best_acc_file = f"{best_model_path}/best_acc_metrics.txt"
    # 读取最佳loss指标
    if os.path.isfile(best_loss_file):
        with open(best_loss_file, 'r') as f:
            line = f.readline()
            best_acc = float(line.split()[2])
            best_acc_epoch = int(line.split()[5])
    # 读取最佳auc指标
    if os.path.isfile(best_auc_file):
        with open(best_auc_file, 'r') as f:
            line = f.readline()
            best_acc = float(line.split()[2])
            best_acc_epoch = int(line.split()[5])
    # 读取最佳acc指标
    if os.path.isfile(best_acc_file):
        with open(best_acc_file, 'r') as f:
            line = f.readline()
            best_acc = float(line.split()[2])
            best_acc_epoch = int(line.split()[5])
    
    # 定义将图像从任意大小调整为224x224的转换
    resize_transform = transforms.Resize((224, 224))
    try:
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Epoch Progress"):
            print(f"Epoch {epoch}/{num_epochs-1} Training")

            # 训练阶段
            model.train()
            train_loss = 0.0
            train_y_true = []
            train_y_pred = []
            train_y_score = []
            for i, data in enumerate(tqdm(train_loader)): # data是一个批次的数据 逐批次取完所有训练数据
                # 加载数据和标签
                combined_volume, tokenized_report, labels, filename = data

                # 如果使用ViT,需要改变H和W为224
                if vit==True:
                    b, s, c, h, w = combined_volume.shape
                    # 创建一个新的空tensor用于存储resize后的图像
                    resized_volume = torch.empty((b, s, c, 224, 224), device=combined_volume.device)
                    for l in range(b):
                        for k in range(s):
                            # 对每个图像进行resize
                            resized_volume[l, k] = resize_transform(combined_volume[l, k])
                    combined_volume = resized_volume

                # 输入数据移动到GPU
                combined_volume, labels = combined_volume.to(device), labels.to(device)
                # 将字典中的每个张量移动到GPU
                tokenized_report = {m: n.to(device) for m, n in tokenized_report.items()}
                # 前向传播
                if fc_type == 'img+rep': # 返回融合特征 (16,512+512)
                    outputs, _, _, student_features = model(combined_volume, tokenized_report) # 模型的输出均为logits
                elif fc_type == 'img_only':
                    outputs, student_features = model(xis=combined_volume)
                elif fc_type == 'rep_only':
                    outputs, _, _ = model(xrs_encoded_inputs=tokenized_report)
                # 引入模拟报告生成
                elif fc_type == 'rep_supervise':
                    # outputs shape: (16, 3) 模拟报告特征 shape: (16, 512) 
                    outputs, simulated_report_feature, rep_features_dict, batch_topk_rep_filenames, student_features = model(combined_volume, filename) 

                # 一定有的分类loss 注意：交叉熵的输入不需要加softmax，会自动添加，也就是输入应该就是logits
                classification_loss = classification_criterion(outputs, labels)
                
                # 如果加入蒸馏模块，计算蒸馏损失
                if is_student:
                    with torch.no_grad(): 
                        # teacher_outputs shape: 16,2
                        teacher_outputs, _, _, teacher_features = teacher_model(combined_volume, tokenized_report)  # teacher_features shape: 16,1024
                    if use_mad:
                        # 使用MAD损失
                        distillation_loss = mad_loss_unweighted(teacher_features, student_features, teacher_outputs) # student_features shape: 16,1024
                    else:
                        # 将学生模型的输出logits通过log_softmax转换，以适配KL散度损失
                        student_outputs = torch.log_softmax(outputs, dim=1) # student_outputs shape: 16,3
                        # 教师模型的输出logits要用过softmax转换
                        teacher_outputs_softmax = F.softmax(teacher_outputs, dim=1)  # teacher_outputs shape: 16,2
                        distillation_loss = distillation_criterion(student_outputs, teacher_outputs_softmax)
                else:
                    distillation_loss = torch.tensor(0, device=device)  # 如果不是学生模型，则蒸馏损失为0

                # 如果加入模拟报告特征生成模块，计算报告对比损失
                if fc_type == 'rep_supervise':
                    actual_report_features = [rep_features_dict[f"{name}_TXT.pt"] for name in filename]
                    actual_report_features = torch.stack(actual_report_features).to(device) # shape (16,512)
                    report_loss = report_criterion(simulated_report_feature, actual_report_features)
                else:
                    report_loss = torch.tensor(0, device=device)  # 如果没有报告监督，报告损失为0

                # 将所有损失加权求和
                loss = loss_weights['classification'] * classification_loss + loss_weights['distillation'] * distillation_loss + loss_weights['report'] * report_loss

                train_loss += loss.item()

                y_score = F.softmax(outputs, dim=1) #概率
                _, y_pred = torch.max(outputs, 1) #预测类别
                y_true = labels #真实标签

                # 累计每个batch的结果
                train_y_true.append(y_true.cpu().numpy())
                train_y_pred.append(y_pred.cpu().numpy())
                train_y_score.append(y_score.cpu().detach().numpy())

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 把嵌套列表变为一维数组
            train_y_true = np.concatenate(train_y_true)
            train_y_pred = np.concatenate(train_y_pred)
            train_y_score = np.concatenate(train_y_score)
            # 计算metrics
            train_acc, train_auc, train_f1, train_precision, train_recall = calculate_metrics(train_y_true, train_y_pred, train_y_score)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_y_true = []
            val_y_pred = []
            val_y_score = []

            # 在训练循环中，在开始验证之前，可以调用这个函数来检查是否有重复元素
            overlap = check_dataset_overlap(train_loader, val_loader)
            if overlap:
                print(f"Warning: There is an overlap between training and validation datasets: {overlap}")
            else:
                print("No overlap between training and validation datasets.")

            print()
            print(f"Epoch {epoch}/{num_epochs-1} Validation")
            with torch.no_grad():
                for data in tqdm(val_loader):
                    # 加载数据和标签
                    combined_volume, tokenized_report, labels, filename = data
                    
                    # 如果使用ViT,需要改变H和W为224
                    if vit==True:
                        b, s, c, h, w = combined_volume.shape
                        # 创建一个新的空tensor用于存储resize后的图像
                        resized_volume = torch.empty((b, s, c, 224, 224), device=combined_volume.device)
                        for l in range(b):
                            for k in range(s):
                                # 对每个图像进行resize
                                resized_volume[l, k] = resize_transform(combined_volume[l, k])
                        combined_volume = resized_volume

                    # 输入数据移动到GPU
                    combined_volume, labels = combined_volume.to(device), labels.to(device)
                    # 将字典中的每个张量移动到GPU
                    tokenized_report = {m: n.to(device) for m, n in tokenized_report.items()}

                    # 前向传播
                    if fc_type == 'img+rep':
                        outputs, _, _, student_features = model(combined_volume, tokenized_report)
                    elif fc_type == 'img_only':
                        outputs, student_features = model(xis=combined_volume)
                    elif fc_type == 'rep_only':
                        outputs, _, _ = model(xrs_encoded_inputs=tokenized_report)
                    # 引入模拟报告生成
                    elif fc_type == 'rep_supervise':
                        # outputs shape: (16, 3) 模拟报告特征 shape: (16, 512) 
                        outputs, simulated_report_feature, rep_features_dict, batch_topk_rep_filenames, student_features = model(combined_volume, filename) 

                    classification_loss = classification_criterion(outputs, labels)

                    # outputs是logits
                    y_score = F.softmax(outputs, dim=1) # 概率
                    _, y_pred = torch.max(outputs, 1) # 预测类别
                    y_true = labels
                    
                    # 累计每个batch的结果
                    val_y_true.append(y_true.cpu().numpy())
                    val_y_pred.append(y_pred.cpu().numpy())
                    val_y_score.append(y_score.cpu().detach().numpy())

                    # 将所有损失加权求和
                    loss =  classification_loss
                    val_loss += loss.item()
                # 把嵌套列表变为一维数组
                val_y_true = np.concatenate(val_y_true)
                val_y_pred = np.concatenate(val_y_pred)
                val_y_score = np.concatenate(val_y_score)
                # 计算metrics
                val_acc, val_auc, val_f1, val_precision, val_recall = calculate_metrics(val_y_true, val_y_pred, val_y_score)
            
            # 计算并存储该epoch的平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 使用验证损失来更新学习率
            # scheduler.step(val_loss/len(val_loader))

            # 输出当前epoch的训练和验证损失
            print(f"Epoch: {epoch}/{num_epochs}\n"
                    f"    Training Loss: {train_loss/len(train_loader):.4f}, Training Acc: {train_acc:.4f},Training AUC: {train_auc:.4f},Training F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}\n"
                    f"    Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_acc:.4f},Validation AUC: {val_auc:.4f},Validation F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            
            # 我如果重复训练的话，希望能覆盖掉之前的结果
            # 比较并更新最佳指标 下面是保存
            # 根据fc_type的不同，选择保存的方式，fold0下有img_only,txt_only和img+txt三个文件夹

            # 为每次实验创建独特的文件夹
            experiment_checkpoint_path = f"{checkpoint_path}"
            best_model_path = f"{experiment_checkpoint_path}/best"
            os.makedirs(best_model_path, exist_ok=True)

            epoch_model_path = f"{experiment_checkpoint_path}/epoch{epoch}"
            os.makedirs(epoch_model_path, exist_ok=True)
            
            fig_dir = f"{experiment_checkpoint_path}/fig"
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = f"{fig_dir}/loss_plot_experiment_{experiment_number}.png"

            # 下面是loss最低，AUC最高，ACC最高的检查点保存
                # 检查并更新最佳损失
            if val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch
                torch.save(model.state_dict(), f"{best_model_path}/best_model_loss.pth")
                with open(f"{best_model_path}/best_loss_metrics.txt", 'w') as f:
                    f.write(f"Best loss: {best_loss/len(val_loader):.4f} at Epoch: {best_loss_epoch}")

            # 检查并更新最佳AUC
            if val_auc > best_auc:
                best_auc = val_auc
                best_auc_epoch = epoch
                torch.save(model.state_dict(), f"{best_model_path}/best_model_auc.pth")
                with open(f"{best_model_path}/best_auc_metrics.txt", 'w') as f:
                    f.write(f"Best AUC: {best_auc} at Epoch: {best_auc_epoch}")

            # 检查并更新最佳准确率
            if val_acc > best_acc:
                best_acc = val_acc
                best_acc_epoch = epoch
                torch.save(model.state_dict(), f"{best_model_path}/best_model_acc.pth")
                with open(f"{best_model_path}/best_acc_metrics.txt", 'w') as f:
                    f.write(f"Best ACC: {best_acc} at Epoch: {best_acc_epoch}")

            # 保存当前epoch的模型和指标
            torch.save(model.state_dict(), f"{epoch_model_path}/model.pth")
            with open(f"{epoch_model_path}/metrics.txt", 'w') as f:
                f.write(f"loss: {val_loss/len(val_loader):.4f}, ACC: {val_acc}, AUC: {val_auc}, F1: {val_f1}, Precision: {val_precision}, Recall:{val_recall}")
        
        # 训练完成后绘制损失图
        plot_losses(train_losses, val_losses, fc_type, fig_path)
        print("Training completed")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 保存当前epoch的损失曲线
        plot_losses(train_losses, val_losses, fc_type, fig_path)
        print("Current training progress and losses plot saved.")

def check_dataset_overlap(train_loader, val_loader):
    # 获取训练集和验证集中所有的文件名
    train_filenames = [data[3] for data in train_loader.dataset]
    val_filenames = [data[3] for data in val_loader.dataset]
    
    # 转换为集合形式以去除重复项，并求交集
    overlap = set(train_filenames).intersection(val_filenames)
    return overlap