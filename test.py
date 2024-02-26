from utils.metrics import calculate_metrics, save_confusion_matrices_to_excel
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from models.multimodal_model import Multimodal_Net  
import os 
from sklearn.metrics import confusion_matrix
import pandas as pd
from data.dataset import BrainTumorDataset
import torch.nn.functional as F
from models.report_supervision_model import ReportSupervisionModel
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

def read_report_content(report_path):
    # 读取报告的内容
    try:
        with open(report_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file {report_path}: {e}"

def label_match_check(original_label, similar_label):
    # 检查标签是否一致
    return original_label.split('_')[0] == similar_label.split('_')[0]

def test(root_path, test_path, best_model_path, batch_size, loss_weights, retrieval_type, device='cuda:0',  aggregation_type='ATTENTION', fc_type='img+rep', is_student=False, num_runs=5, vit=False):
    report_dir = os.path.join(root_path,'TXT')
    # 保存特征的路径
    img_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/IMG'  # 替换为实际的影像特征文件夹路径
    rep_feature_dir = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/features_database/TXT'   # 替换为实际的文本特征文件夹路径
    # 初始化DataFrame
    results_df = pd.DataFrame(columns=["检查流水号", "真实标签", "预测结果", "预测概率", "正确与否"])
    # 读取Excel文件以获取测试集的check_number
    df = pd.read_excel(test_path)
    test_check_numbers_excel = df['check_number'].tolist()
    actual_labels_excel = df['label'].tolist()
    print("长度：", len(test_check_numbers_excel))
    # 创建数据集
    dataset = BrainTumorDataset(img_dir=root_path, report_dir=root_path)  # 使用和训练相同的数据集类
    dataset.save_preprocessed_data()  # 确保数据已预处理
    _, val_set, test_set, _, _, _, _ = dataset.split_dataset(test_path)  # 获取测试集

    # 打印test_set中前几个元素的信息
    for i, item in enumerate(test_set[:5]):
        combined_volume, tokenized_report, label, filename = item
        print(f"Sample {i}:")
        print(f"Filename: {filename}")
        print(f"Label: {label.item()}")
        print()
        
    all_results = []
    confusion_matrices = []
    # 损失权重
    weight_str = "_".join(f"{key}{value}" for key, value in loss_weights.items())
    for run in range(num_runs):
        print(f"Running test {run+1}/{num_runs}")
        # 创建测试数据加载器
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        if is_student==True and fc_type=='rep_supervise':
            model = ReportSupervisionModel(img_feature_dir, rep_feature_dir, aggregation_type, device, retrieval_type).to(device)  # 初始化报告监督模型并移动到GPU
            fc_type = 'student+rep_supervise'  # 如果是学生＋报告监督模型，更改fc_type，便于保存
        elif is_student==True and fc_type!='rep_supervise':
            fc_type = 'student'  # 如果是单独学生模型，更改fc_type，便于保存
            model = Multimodal_Net(fc_type = 'img_only').to(device)  # 学生模型是img_only的
        elif is_student==False and fc_type=='rep_supervise':
            model = ReportSupervisionModel(img_feature_dir, rep_feature_dir, aggregation_type, device, retrieval_type).to(device)  # 初始化报告监督模型并移动到GPU
            fc_type = 'rep_supervise'  # 如果是单独报告监督模型，更改fc_type，便于保存
        else:
            if vit==True:
                model = Multimodal_Net(fc_type=fc_type,
                                        input_W=224,  # width of slice
                                        input_H=224,  # height of slice
                                        input_D=24,  # slice number
                                    ).to(device)
            else:
                model = Multimodal_Net(fc_type=fc_type).to(device)
        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        # 测试并计算指标
        model.eval()
        test_y_true = []
        test_y_pred = []
        test_y_score = []

        # 在测试函数的开始定义一个空列表来存储结果
        retrieved_reports_info = []
        resize_transform = transforms.Resize((224, 224))
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                combined_volume, tokenized_report, labels, filenames = data
                # 如果使用ViT,需要改变H和W为224
                if vit==True:
                    b, s, c, h, w = combined_volume.shape
                    # 创建一个新的空tensor用于存储resize后的图像
                    resized_volume = torch.empty((b, s, c, 224, 224), device=combined_volume.device)
                    for k in range(b):
                        for l in range(s):
                            # 对每个图像进行resize
                            resized_volume[k, l] = resize_transform(combined_volume[k, l])
                    combined_volume = resized_volume

                combined_volume, labels = combined_volume.to(device), labels.to(device)
                tokenized_report = {m: n.to(device) for m, n in tokenized_report.items()}

                # 预测
                if fc_type == 'img+rep':
                    outputs, _, _, student_features= model(combined_volume, tokenized_report)
                elif fc_type == 'img_only' or fc_type == 'student':
                    outputs, student_features = model(xis=combined_volume)
                elif fc_type == 'rep_only':
                    outputs, _, _ = model(xrs_encoded_inputs=tokenized_report)
                elif fc_type == 'rep_supervise' or fc_type == 'student+rep_supervise':
                    outputs, simulated_report_feature, rep_features_dict, batch_topk_rep_filenames, student_features = model(combined_volume, filenames) 

                # outputs是logits
                y_score = F.softmax(outputs, dim=1) #概率
                _, y_pred = torch.max(outputs, 1)
                y_true = labels
                print("labels:", labels.shape)
                test_y_true.append(y_true.cpu().numpy())
                test_y_pred.append(y_pred.cpu().numpy())
                test_y_score.append(y_score.cpu().detach().numpy())

                for j in range(labels.size(0)):  # Iterate over each sample in the batch
                    index = i * batch_size + j
                    # print(f"i:{i},j:{j}")
                    check_number = test_check_numbers_excel[index]
                    actual_label = actual_labels_excel[index]
                    predicted_label = y_pred[j].item()
                    prob = y_score[j].cpu().numpy()
                    correct = (predicted_label == actual_label)
                    # Add a new row to the DataFrame using the loc indexer
                    results_df.loc[len(results_df.index)] = {
                        "检查流水号": check_number,
                        "真实标签": actual_label,
                        "预测结果": predicted_label,
                        "预测概率": prob,
                        "正确与否": correct
                    }

                if fc_type == 'rep_supervise' or fc_type == 'student+rep_supervise':
                    # 收集信息用于后续保存到Excel表格 batch_topk_rep_filenames嵌套列表(16,10)
                    for file_name, topk_filenames in zip(filenames, batch_topk_rep_filenames):
                        original_report_filename = file_name + '.txt'
                        original_report_path = os.path.join(report_dir, original_report_filename)
                        original_report_content = read_report_content(original_report_path)
                        original_label = file_name.split('_')[0]

                        report_info = {
                            'Image_Filename': file_name,
                            'Original_Report': original_report_content
                        }
                                        
                                        
                        # 检查是否有有效的相似报告文件名列表
                        if topk_filenames is not None and all(filenames is not None for filenames in topk_filenames):
                        # 将相似报告的文件名分散到单独的列中
                            for idx, similar_filename in enumerate(topk_filenames):
                                report_filename = similar_filename.replace('_TXT.pt', '.txt') # 假设报告文件名的扩展名实际上是 '.txt'
                                report_path = os.path.join(report_dir, report_filename) # 构建报告文件的完整路径
                                report_content = read_report_content(report_path) # 读取报告的内容
                                similar_label = report_filename.split('_')[0]
                                label_match = label_match_check(original_label, similar_label)
                                
                                report_info[f'Similar_Report_{idx+1}'] = similar_filename
                                report_info[f'Similar_Report_{idx+1}_content'] = report_content # 将内容添加到字典的新键中
                                report_info[f'Label_Match_Similar_Report_{idx+1}'] = label_match
                        else:
                            # 处理None占位符的情况，你可以决定如何填充这些字段
                            for idx in range(10):
                                report_info[f'Similar_Report_{idx+1}'] = None
                                report_info[f'Similar_Report_{idx+1}_content'] = None
                                report_info[f'Label_Match_Similar_Report_{idx+1}'] = None
                
                        retrieved_reports_info.append(report_info)

        # # 创建验证集数据加载器
        # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # # 评估验证集
        # val_y_true = []
        # val_y_pred = []
        # val_y_score = []

        # with torch.no_grad():
        #     for data in tqdm(val_loader, desc="Validating"):
        #         combined_volume, tokenized_report, labels, filenames = data
        #         combined_volume, labels = combined_volume.to(device), labels.to(device)
        #         tokenized_report = {k: v.to(device) for k, v in tokenized_report.items()}

        #         # 前向传播
        #         if fc_type == 'img+rep':
        #             outputs, _, _, student_features= model(combined_volume, tokenized_report)
        #         elif fc_type == 'img_only' or fc_type == 'student':
        #             outputs, student_features = model(xis=combined_volume)
        #         elif fc_type == 'rep_only':
        #             outputs, _, _ = model(xrs_encoded_inputs=tokenized_report)
        #         elif fc_type == 'rep_supervise' or fc_type == 'student+rep_supervise':
        #             outputs, simulated_report_feature, rep_features_dict, batch_topk_rep_filenames, student_features = model(combined_volume, filenames) 

        #         # 计算验证结果
        #         y_score = F.softmax(outputs, dim=1)  # 概率
        #         _, y_pred = torch.max(outputs, 1)  # 预测类别
        #         y_true = labels

        #         val_y_true.append(y_true.cpu().numpy())
        #         val_y_pred.append(y_pred.cpu().numpy())
        #         val_y_score.append(y_score.cpu().detach().numpy())

        # # 计算验证集指标
        # val_y_true = np.concatenate(val_y_true)
        # val_y_pred = np.concatenate(val_y_pred)
        # val_y_score = np.concatenate(val_y_score)
        # val_acc, val_auc, val_f1, val_precision, val_recall = calculate_metrics(val_y_true, val_y_pred, val_y_score)

        # # 输出验证集的指标
        # print(f"Validation Set - Acc: {val_acc}, AUC: {val_auc}, F1: {val_f1}, Precision: {val_precision}, Recall: {val_recall}")

        test_y_true = np.concatenate(test_y_true)
        test_y_pred = np.concatenate(test_y_pred)
        test_y_score = np.concatenate(test_y_score)
        test_acc, test_auc, test_f1, test_precision, test_recall = calculate_metrics(test_y_true, test_y_pred, test_y_score)
        all_results.append((test_acc, test_auc, test_f1, test_precision, test_recall))
        # 输出当前运行的指标
        print(f"Run {len(all_results)} - Test Acc: {test_acc}, Test AUC: {test_auc}, Test F1: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}")

        # 生成混淆矩阵
        cm = confusion_matrix(test_y_true, test_y_pred)
        confusion_matrices.append(cm)

    # 计算平均指标
    avg_metrics = [sum(x) / len(all_results) for x in zip(*all_results)]

    # 创建并保存测试结果和混淆矩阵
    result_dir = os.path.join('./test_result', fc_type)
    os.makedirs(result_dir, exist_ok=True)
    result_file_name = f"{fc_type}_test_results.txt"
    result_file_path = os.path.join(result_dir, result_file_name)
    cm_file_name = f"{fc_type}_confusion_matrix.xlsx"
    cm_file_path = os.path.join(result_dir, cm_file_name)
    
    # 保存每次运行的混淆矩阵
    save_confusion_matrices_to_excel(confusion_matrices, cm_file_path, spacing_rows=3)

    # 保存所有运行的结果和平均结果到一个文件
    with open(result_file_path, 'w') as f:
        for i, result in enumerate(all_results):
            f.write(f"Run {i+1} - Test Acc: {result[0]}, Test AUC: {result[1]}, Test F1: {result[2]}, Test Precision: {result[3]}, Test Recall: {result[4]}\n")
        f.write("\n")
        f.write(f"Average - Test Acc: {avg_metrics[0]}, Average Test AUC: {avg_metrics[1]}, Average Test F1: {avg_metrics[2]}, Average Test Precision: {avg_metrics[3]}, Average Test Recall: {avg_metrics[4]}\n")
    
    results_df.to_excel(os.path.join(result_dir, f'{fc_type}_detailed_test_results.xlsx'))

    if fc_type == 'rep_supervise' or fc_type == 'student+rep_supervise':
        # 将包含报告内容的信息转换成DataFrame
        retrieved_reports_df = pd.DataFrame.from_records(retrieved_reports_info)
        # 保存包含报告内容的DataFrame到Excel文件
        # retrieved_reports_df.to_excel(os.path.join(result_dir, 'retrieved_reports_with_content.xlsx'), index=False)
        
        # Extract the check number from the 'Image_Filename' column (assuming the check number is after the first underscore)
        retrieved_reports_df['检查流水号'] = retrieved_reports_df['Image_Filename'].str.split('_').str[1]
        # Ensure the '检查流水号' column is of the same data type in both DataFrames
        retrieved_reports_df['检查流水号'] = retrieved_reports_df['检查流水号'].astype(str)
        results_df['检查流水号'] = results_df['检查流水号'].astype(str)
        # Merge the DataFrames on '检查流水号'
        combined_df = pd.merge(retrieved_reports_df, results_df, on='检查流水号', how='left')
        # Save the combined DataFrame to an Excel file
        combined_df.to_excel(os.path.join(result_dir, 'combined_reports_with_predictions.xlsx'), index=False)
