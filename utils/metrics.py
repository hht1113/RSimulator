from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def calculate_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    # auc是类别1为正类
    auc = roc_auc_score(y_true, y_score[:, 1], multi_class='ovr', average='macro') if y_score is not None else None
    f1 = f1_score(y_true, y_pred, average='macro') # 两个类别分别做正类的均值
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    return acc, auc, f1, precision, recall

#热力图输出混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_losses(train_losses, val_losses, fc_type, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{fc_type}:Training and Validation Loss vs. Epochs')
    plt.legend()
    plt.savefig(save_path)  # 保存图像到文件
    plt.show()

def save_confusion_matrices_to_excel(confusion_matrices, cm_file_path, spacing_rows=3):
    wb = Workbook()
    ws = wb.active
    # 当前写入的行位置
    current_row = 1

    for i, cm in enumerate(confusion_matrices):
        # 将混淆矩阵转换为DataFrame
        cm_df = pd.DataFrame(cm, index=[f"Actual {j}" for j in range(len(cm))], 
                             columns=[f"Predicted {k}" for k in range(len(cm[0]))])
        # 将DataFrame写入Excel工作表
        for r in dataframe_to_rows(cm_df, index=True, header=True):
            ws.append(r)
        # 更新当前行位置，添加间隔
        current_row += len(cm_df) + spacing_rows + 1  # 加1是因为标题行
        # 如果不是最后一个混淆矩阵，添加间隔行
        if i < len(confusion_matrices) - 1:
            for _ in range(spacing_rows):
                ws.append([])
    # 保存工作簿
    wb.save(cm_file_path)