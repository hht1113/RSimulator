import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom
import nibabel as nib
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from itertools import islice

# 指定模型文件和配置文件的路径
rep_model_path = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/roberta_config/'

class BrainTumorDataset(Dataset):
    def __init__(self, img_dir=None, report_dir=None, transform=None): 
        super().__init__()
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.classes = ['wild_type', 'mutation_type'] # 0,1
        self.class_to_idx = {cls_name:i for i, cls_name in enumerate(self.classes)}
        self.img_labels = self._find_img_labels() # 一个列表，元素是字典，保存三个模态的影像和报告的地址以及标签
        self.tokenizer = BertTokenizer.from_pretrained(rep_model_path)
        # self.tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")
    
    def _find_img_labels(self):
        img_labels = []
        for cls_name, label in self.class_to_idx.items(): #('wild_type',0)
            cls_img_dirs = {modality: os.path.join(self.img_dir, modality) for modality in ['T1', 'T2', 'T1C', 'FLAIR', 'DWI']}
            cls_report_dir = os.path.join(self.report_dir, 'TXT')
            # t1_files = [f for f in os.listdir(cls_img_dirs['T1']) if f.startswith(str(label)) and f.endswith('_T1.nii.gz')]
            t1_files = sorted([f for f in os.listdir(cls_img_dirs['T1']) if f.startswith(str(label)) and f.endswith('_T1.nii.gz')])

            # 打印前几个t1_files以进行验证
            # print(f"First few t1_files for class {cls_name}: {t1_files[:5]}")

            for t1_file in t1_files:
                check_number = t1_file.split('_')[1]
                # 生成对应T2、T1C、FLAIR、DWI模态的文件名
                t2_file = f"{label}_{check_number}_t2_Warped.nii.gz"
                t1c_file = f"{label}_{check_number}_t1c_Warped.nii.gz"
                flair_file = f"{label}_{check_number}_flair_Warped.nii.gz"
                dwi_file = f"{label}_{check_number}_dwi_InverseWarped.nii.gz"

                # 创建每个模态的完整路径
                img_paths = [
                    os.path.join(cls_img_dirs['T1'], t1_file),
                    os.path.join(cls_img_dirs['T2'], t2_file),
                    os.path.join(cls_img_dirs['T1C'], t1c_file),
                    os.path.join(cls_img_dirs['FLAIR'], flair_file),
                    os.path.join(cls_img_dirs['DWI'], dwi_file)
                ]

                # 移除影像文件名中的后缀并加上.txt来创建报告文件的路径
                report_file = f"{label}_{check_number}.txt"
                report_path = os.path.join(cls_report_dir, report_file)
                
                img_labels.append({
                    'img_paths': img_paths,
                    'report_path': report_path,
                    'label': label
                })
        return img_labels
    
    def save_preprocessed_data(self):
        os.makedirs('preprocessed_data_24_128_128', exist_ok=True)  # 确保目录存在
        idx_check_number_mapping = {}  # 创建一个空字典来存储idx和check_number的映射
        
        for idx in tqdm(range(len(self.img_labels)), desc="Processing data"):
            img_label = self.img_labels[idx]
            # 提取检查流水号
            check_number = os.path.basename(img_label['img_paths'][0]).split('_')[1]
            
            # 添加到映射中
            idx_check_number_mapping[idx] = check_number

            # 构建保存路径
            combined_volume_path = f'preprocessed_data_24_128_128/combined_volume_tensor_{check_number}.pt'
            tokenized_report_path = f'preprocessed_data_24_128_128/tokenized_report_{check_number}.pt'
            label_path = f'preprocessed_data_24_128_128/label_tensor_{check_number}.pt'
            mapping_path = 'preprocessed_data_24_128_128/idx_check_number_mapping.pt'
            # 如果数据已存在，则跳过
            if os.path.exists(combined_volume_path) and os.path.exists(tokenized_report_path) and os.path.exists(label_path):
                continue
            
            # 处理数据
            combined_volume_tensor, tokenized_report, label_tensor, filename = self.__getitem__(idx)

            # 保存张量到磁盘
            torch.save(combined_volume_tensor, combined_volume_path)
            torch.save(tokenized_report, tokenized_report_path)
            torch.save(label_tensor, label_path)
            # 保存idx和check_number的映射到磁盘
            torch.save(idx_check_number_mapping, mapping_path)

    def __len__(self):
        return len(self.img_labels)
    
    def split_dataset(self, test_path, preprocessed_data_24_128_128_dir='preprocessed_data_24_128_128'): # preprocessed_data_24_128_128是最新的预处理数据
        # 读取Excel文件以获取测试集的check_number
        df = pd.read_excel(test_path)
        test_check_numbers = df['check_number'].tolist()
        # 确保测试集的check_number是字符串形式
        test_check_numbers = [str(check_number) for check_number in test_check_numbers]

        # 读取映射
        mapping_path = os.path.join(preprocessed_data_24_128_128_dir, 'idx_check_number_mapping.pt')
        if os.path.exists(mapping_path):
            idx_check_number_mapping = torch.load(mapping_path)
        else:
            raise FileNotFoundError("The index to check number mapping file does not exist.")

        # 创建反向映射：检查流水号到索引
        # check_number_to_idx_mapping = {check_number: idx for idx, check_number in idx_check_number_mapping.items()}
        # 创建反向映射：检查流水号到索引，确保键是字符串
        check_number_to_idx_mapping = {str(check_number): idx for idx, check_number in idx_check_number_mapping.items()}

        # 从映射中找到不在测试集中的索引，这些将用于训练和验证
        train_val_idxs = [idx for idx, check_number in idx_check_number_mapping.items() if check_number not in test_check_numbers]
        # 将剩余索引分为训练和验证集
        train_idxs, val_idxs = train_test_split(train_val_idxs, test_size=0.1/0.9, random_state=47) # 整体10%的数据用作验证
        # 使用__getitem__方法来获取相应的数据集
        train_set = [self[idx] for idx in train_idxs]  # 注意这里使用的是self[idx]而不是self.__getitem__(idx)
        val_set = [self[idx] for idx in val_idxs]
        test_set = [self[check_number_to_idx_mapping[check_number]] for check_number in test_check_numbers]

        # 计算每个集合中每个类别的样本数
        train_labels = [self[idx][2].item() for idx in train_idxs]  # 假设self[idx][2]是类别标签
        val_labels = [self[idx][2].item() for idx in val_idxs]
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)

        # 计算总样本数
        total_train_samples = len(train_set)
        total_val_samples = len(val_set)
        return train_set, val_set, test_set, total_train_samples, total_val_samples, train_label_counts, val_label_counts
    
    def __getitem__(self, idx):
        img_label = self.img_labels[idx]
        filename = os.path.basename(img_label['img_paths'][0]).split('_')[0] + '_' + os.path.basename(img_label['img_paths'][0]).split('_')[1]
        check_number = os.path.basename(img_label['img_paths'][0]).split('_')[1]
        
        # 根据检查流水号构建文件路径
        combined_volume_path = f'/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/preprocessed_data_24_128_128/combined_volume_tensor_{check_number}.pt'
        tokenized_report_path = f'/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/preprocessed_data_24_128_128/tokenized_report_{check_number}.pt'
        label_path = f'/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/preprocessed_data_24_128_128/label_tensor_{check_number}.pt'

        if os.path.exists(combined_volume_path) and os.path.exists(tokenized_report_path) and os.path.exists(label_path):
            # 如果预处理数据已存在，直接加载
            combined_volume_tensor = torch.load(combined_volume_path)
            tokenized_report = torch.load(tokenized_report_path)
            label_tensor = torch.load(label_path)
        else:
            img_paths = self.img_labels[idx]['img_paths'] # 得到列表，包含五个模态影像
            report_path = self.img_labels[idx]['report_path']
            label = self.img_labels[idx]['label']
            
            modalities = []
            for img_path in img_paths:
                nii_data = nib.load(img_path).get_fdata()
                nii_data = adjust_slices(nii_data) #H,W,slice 固定到256,256,24

                # 对每个切片应用 normalize 函数 注释掉与直接归一化比较效果
                # for slice_idx in range(nii_data.shape[2]):
                #     nii_data[:, :, slice_idx] = normalize(nii_data[:, :, slice_idx])

                modalities.append(nii_data)
            # Stack along the channel dimension and apply transforms
            # C,H,W,slice==3,256,256,24
            combined_volume = np.stack(modalities, axis=0)
            #把slice换到最前，slice,C,H,W == 24,3,256,256
            combined_volume = np.transpose(combined_volume, (3, 0, 1, 2))

            # combined_volume = self.transform(combined_volume) # ndarray转为张量 transform无法接受四维的输入
            # 将 combined_volume 转换为张量
            combined_volume_tensor = torch.from_numpy(combined_volume).float()
            # 已经在normalize归一化过了
            combined_volume_tensor /= 255.0

            # 读取并tokenize报告
            with open(report_path, 'r') as file:
                report = file.read()
            # 字典，包含input_ids,attention_mask等
            tokenized_report = self.tokenizer(report, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            # 移除tokenizer自动添加的批次维度，以防dataloader再添加维度造成维度错误 
            tokenized_report = {key: value.squeeze(0) for key, value in tokenized_report.items()}
            label_tensor = torch.tensor(label)


        return combined_volume_tensor, tokenized_report, label_tensor, filename# dataset_getitem()返回的要是张量
    
def adjust_slices(volume, desired_slices=24):
    current_slices = volume.shape[2] #H W slice
    new_volume = resample_volume(volume, desired_slices)

    # 下面操作更精细
    # if current_slices == desired_slices:
    #     # No adjustment needed
    #     new_volume = resample_volume(volume, desired_slices)
    #     return new_volume
    # elif 14 < current_slices < desired_slices:
    #     # Fewer slices than desired, replicate slices from both ends
    #     slices_to_add = desired_slices - current_slices
    #     front = volume[:, :, :1].repeat(slices_to_add // 2 + slices_to_add % 2, axis=2)
    #     back = volume[:, :, -1:].repeat(slices_to_add // 2, axis=2)
    #     new_volume = np.concatenate((front, volume, back), axis=2)
    #     new_volume = resample_volume(new_volume, desired_slices)
    # elif 24 < current_slices <= 34:
    #     # More slices than desired but within the threshold, drop slices from both ends
    #     slices_to_drop = current_slices - desired_slices
    #     front = slices_to_drop // 2
    #     back = current_slices - slices_to_drop + front
    #     new_volume = volume[:, :, front:back]
    #     new_volume = resample_volume(new_volume, desired_slices)
    # else:
    #     # Significantly more slices than desired, resample
    #     # Here we use linear interpolation along the slice axis
    #     new_volume = resample_volume(volume, desired_slices)
    
    return new_volume

def resample_volume(volume, desired_slices, target_shape=(128, 128)):
    # 用插值方法调整体积数据的切片数量和体素大小
    current_slices = volume.shape[2]
    slice_factor = desired_slices / current_slices
    xy_factor = [target_shape[0] / volume.shape[0], target_shape[1] / volume.shape[1]]
    zoom_factors = xy_factor + [slice_factor]
    adjusted_volume = zoom(volume, zoom_factors, order=3)  # 使用三次样条插值 这是对体积volume的插值，三个维度插值到要求
    return adjusted_volume

# 归一化函数
def normalize(slice):
    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        mean = np.mean(image_nonzero)
        std = np.std(image_nonzero)
        slice = (slice - mean) / std
        # 将黑色背景区域设置为特定值（例如：-9）
        slice[slice == 0] = -9
    return slice

# def new_adjust_slices(volume, desired_slices=22):
