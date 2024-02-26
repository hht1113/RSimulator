import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, BertModel
from models.resnet3D import *
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models._utils import _ovewrite_named_param

from models.vit_model import *
from models.vit_model import vit_base_patch16_224_in21k as create_model
import os

def _modify_3d_resnet(model, channels):
    # 获取原始模型的第一层3D卷积
    old_conv1 = model.conv1

    # 创建新的3D卷积层，输入通道数为5，其余参数与原始卷积层相同
    new_conv1 = nn.Conv3d(channels, old_conv1.out_channels,
                          kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                          padding=old_conv1.padding, bias=False)

    # 初始化新卷积层的权重
    with torch.no_grad():
        new_conv1.weight[:, :3] = old_conv1.weight.clone()  # 复制前三个通道的权重
        # 对剩余的两个通道进行初始化（这里选择使用前三个通道的权重的平均值）
        new_conv1.weight[:, 3:] = torch.mean(old_conv1.weight, dim=1, keepdim=True).clone()

    # 将模型的第一层卷积替换为新的卷积层
    model.conv1 = new_conv1

    return model

def _modify_vit_input_layer(vit_model, channels=5):
    # 获取原始模型的patch embedding层
    old_patch_embed = vit_model.patch_embed

    # 创建新的卷积层，输入通道数为5，其余参数与原始卷积层相同
    new_proj = nn.Conv2d(channels, old_patch_embed.proj.out_channels,
                         kernel_size=old_patch_embed.proj.kernel_size, 
                         stride=old_patch_embed.proj.stride,
                         padding=old_patch_embed.proj.padding, bias=False)

    # 初始化新卷积层的权重
    with torch.no_grad():
        # 复制前三个通道的权重
        new_proj.weight[:, :3, :, :] = old_patch_embed.proj.weight.clone()
        # 对剩余的两个通道进行初始化（这里选择使用前三个通道的权重的平均值）
        new_proj.weight[:, 3:, :, :] = torch.mean(old_patch_embed.proj.weight, dim=1, keepdim=True).expand(-1, channels-3, -1, -1).clone()

    # 将patch embedding层的卷积层替换为新的卷积层
    vit_model.patch_embed.proj = new_proj

    return vit_model

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self,
                 image_model,
                 aggregation_type,
                 H,
                 W,
                 D,
                 channels,
                 mm_dim,
                 num_class,
                 num_heads,
                 bias,
                 dropout_rate,
                 mask_columns):
        super(ImageEncoder, self).__init__()

        # init Resnet
        self.aggregation = aggregation_type
        self.H = H
        self.W = W
        self.slice_num = D
        self.channels = channels
        self.mm_dim = mm_dim
        self.num_class = num_class
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout_rate
        self.mask_columns = mask_columns
        
        if aggregation_type == 'ViT':
            # 假设我们使用预训练的ViT模型
            self.vit_model = create_model(num_classes=0, has_logits=False).to('cuda:0') # class=0是为了得到特征
            # 加载预训练权重
            weights_path = '/data1/houhaotian/PyCharmCode/Vision-Transformer-pytorch-main/weights/vit_base_patch16_224_in21k.pth'
            # 检查权重文件路径是否存在
            assert os.path.exists(weights_path), f"Weights file not found: {weights_path}"
            
            weights_dict = torch.load(weights_path, map_location='cuda:0')
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if self.vit_model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(self.vit_model.load_state_dict(weights_dict, strict=False))

            # 在加载预训练权重之后修改模型输入层以适应5个通道的输入
            self.vit_model = _modify_vit_input_layer(self.vit_model, channels=5)

            # for name, para in self.vit_model.named_parameters():
            #     # 除head, pre_logits外，其他权重全部冻结
            #     if "head" not in name and "pre_logits" not in name:
            #         para.requires_grad_(False)
            #     else:
            #         print("training {}".format(name))

        #导入resnet模型 fc_input=512
        resnet_model, fc_input = self._get_res_basemodel(image_model, aggregation_type, H, W, D, channels)
        if aggregation_type == '3D':
            self.resnet_model_1 = nn.Sequential(*list(resnet_model.children())[:-1])  # drop FC
        else:
            self.resnet_model_1 = nn.Sequential(*list(resnet_model.children())[:-1])  # drop FC
            self.resnet_model_2 = nn.Sequential(*list(resnet_model.children())[:-2])  # drop FC and avgpool

        self.fc_input = fc_input

        # COS_SIM & ATTENTION
        self.Proj_REP_cs = nn.Linear(512, self.mm_dim, bias=self.bias)
        self.Proj_SLICE_cs = nn.Linear(self.fc_input, self.mm_dim, bias=self.bias)

    def _get_img_dim(self):
        return self.fc_input

    def _get_res_basemodel(self, image_model, aggregation_type, H, W, D, channels):
        # backbone
        if aggregation_type == '3D':
            model = resnet18(sample_input_W=W, sample_input_H=H, sample_input_D=D,
                            channels=3,  # 初始时假设通道数为3
                            shortcut_type='A', no_cuda=False, num_seg_classes=1)
            # 修改模型以接受5个通道的输入
            model = _modify_3d_resnet(model, channels)
        else:
            self.resnet_dict = {
                "resnet18": models.resnet18(weights=ResNet18_Weights.DEFAULT),
                "resnet34": models.resnet34(weights=ResNet34_Weights.DEFAULT),
                "resnet50": models.resnet50(weights=ResNet50_Weights.DEFAULT),
                "resnet101": models.resnet101(weights=ResNet101_Weights.DEFAULT)
            }
            model = self.resnet_dict[image_model] # 从torchvision.models导入
 
            # Modify the first convolutional layer
            old_conv1 = model.conv1
            new_conv1 = nn.Conv2d(channels, old_conv1.out_channels,
                                kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                padding=old_conv1.padding, bias=False)
            
            # Copy the weights from the old conv1 to the new, assuming that the first 3 channels
            # of the new conv1 should be initialized the same way as the old conv1
            with torch.no_grad():
                new_conv1.weight[:, :3] = old_conv1.weight.clone()  # Copy old weights
                # For the additional channels, use the mean of the weights of the first 3 channels
                new_conv1.weight[:, 3:] = torch.mean(old_conv1.weight, dim=1, keepdim=True).clone()

            # Overwrite the first convolutional layer
            model.conv1 = new_conv1

        print(f"Image feature extractor: {image_model}, aggregation type: {aggregation_type}")
        fc_input = 512 # 让三通道训练的resnet18接受五通道的输入
        return model, fc_input 

    def transfer_mask(self, mask_list):
        mask_arr = np.full((8, 8), fill_value=1)
        for m in mask_list:
            mask_arr[:, m] = 0 # m列设置为0，其余列保持为1
        return mask_arr

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, #每个头的隐藏单元数
                      -1)  # reshape ('batch_size', 'slice', 'num_heads', 'num_hiddens' / 'num_heads')
        X = X.permute(0, 2, 1, 3)
        return X

    def COS_SIM(self, xpool, xdt):
        # xdt (batch_size, 768)
        # xpool (batch_size,slice,512)
        xdmm = self.Proj_REP_cs(xdt)  # (batch_size, mmdim)
        xpmm = self.Proj_SLICE_cs(xpool)  # (batch_size,slice,mmdim)

        # cosine 每行被转换为单位向量
        xdmm_norm = F.normalize(xdmm, dim=-1).unsqueeze(dim=1)  # (batch,1,512)
        xpmm_norm = F.normalize(xpmm, dim=-1)  # (batch,20,512)
        # cosine其实就是两个向量单位向量的点积
        xdmm_norm = xdmm_norm.expand(-1, 20, -1)  # 扩展 xdmm_norm 的尺寸为 (batch, 20, 512)
        # print('-------------', xdmm_norm.shape) 余弦相似度需要两个向量维度相同
        similarity = F.cosine_similarity(xdmm_norm, xpmm_norm, dim=-1)  # (batch,slice)
        similarity_norm = F.softmax(similarity, dim=-1)
        # 增加到 (batch,1,slice)
        similarity_norm = similarity_norm.unsqueeze(dim=1)
        # 批次矩阵乘法，先不考虑batch
        similarity_weighted = torch.matmul(similarity_norm, xpool)  # (batch,1,512)

        v_weighted = similarity_weighted.squeeze()  # (batch,512)
        slice_atts = similarity_norm.squeeze()  # (batch,20) 对每个切片的余弦相似性

        return v_weighted, slice_atts

    def ATTENTION(self, xpool, xdt):
        # xdt (batch_size, 512)
        # xpool (batch_size,slice,512)
        xdmm = self.Proj_REP_cs(xdt)  # (batch_size, mmdim)
        xpmm = self.Proj_SLICE_cs(xpool)  # (batch_size,slice,mmdim)

        # attention
        xdmm_norm = F.normalize(xdmm, dim=-1).unsqueeze(dim=1)  # (batch,1,512)
        xpmm_norm = F.normalize(xpmm, dim=-1)  # (batch,24,512)
        # print("报告形状和影像形状：", xdmm_norm.shape, xpmm_norm.shape)
        try:
            cos_sim = torch.matmul(xdmm_norm, xpmm_norm.transpose(1, 2))  # (batch,1,slice)
        except IndexError:
            xpmm_norm = torch.unsqueeze(xpmm_norm,0)
            cos_sim = torch.matmul(xdmm_norm, xpmm_norm.transpose(1, 2))
        cos_sim_norm = F.softmax(cos_sim, dim=-1)  # (batch,1,24)
        # print(cos_sim_norm)
        v_weighted = torch.matmul(cos_sim_norm, xpool) # (batch,1,512)
        # v_weighted = torch.matmul(cos_sim_norm, xpmm)

        v_weighted = v_weighted.squeeze(1)  # (batch,512) 防止batch为1时，把batch维度去掉
        slice_atts = cos_sim_norm.squeeze()  # (batch,20)

        return v_weighted, slice_atts

    def forward(self, xis, xr_slice=None): # xis shape:(16,24,3,256,256)
        # Encoding
        # 3D resnet18

        if self.aggregation == 'ViT':
            if self.vit_model is None:
                raise ValueError("ViT model is not initialized.")
            # print(xis.shape)
            batch_size = xis.shape[0]
            xis = xis.reshape(-1, self.channels, self.H, self.W)  # batch和slice先乘一起，变成大批次

            # print("xis:",xis.shape)
            features = self.vit_model(xis)  # 提取特征 [batch*slice,C]
            # print("features_before:", features.shape)
            features = features.reshape(batch_size, self.slice_num, -1)  # 将特征恢复到原始的切片维度 [batch,slice,C]
            # print("features_after:", features.shape)
            v = features.mean(dim=1)  # 在切片维度上进行平均 [batch,C] C=768
            # print("v:",v.shape)

            # # print("xis:",xis.shape)
            # features = self.vit_model(xis)
            # # print("features_before:", features.shape)
            # v = features.reshape(batch_size, self.slice_num, -1)  # [batch,slice,2]
            # # print("v:",v.shape)
            # v = v.mean(dim=1)  # 在切片维度上进行平均 [batch,2]
            # # print("v:",v.shape)
            return v

        if self.aggregation == '3D':
            # 3D ResNet的输入shape: (batchsize,Channel,depth,H,W) 所以交换了一下
            xis = xis.transpose(1, 2)
            h = self.resnet_model_1(xis) # 3D ResNet的输出shape: (batch_size, num_features, Depth,h,w) 和输入维度相比有变化
            hi = nn.AdaptiveAvgPool3d((1, 1, 1))(h) # shape:(batch_size,num_features,1,1,1)
            hi = nn.Flatten()(hi)
            # print('ok')
            return hi
        
        # 2D & 2.5D
        # first squeeze before encoding
        # xis = xis.transpose(1, 2)
        batch_size = xis.shape[0]
        xis = xis.reshape(batch_size * self.slice_num, self.channels, self.H, self.W)  # (batch*slice, 3, 256,256)
        hi = self.resnet_model_2(xis)  # hi (batch*slice, 512, 8,8)
        # then expand after encoding
        hi = hi.reshape(batch_size, self.slice_num, 512, 4, 4)  # (batch,slice,512, 8,8)
        h_ = nn.AdaptiveAvgPool2d((1, 1))(hi)  # hi (batch,slice,512, 1, 1)
        # h_squeeze = h_.squeeze()  # (batch,slice,512)
        h_squeeze = h_.squeeze(-1).squeeze(-1) # 移除后两个维度，防止batch为1被squeeze
        # Aggregation
        v, slice_scores, region_atts = None, None, None
        if xr_slice is not None:
            if self.aggregation == 'AVG':
                v = torch.mean(h_squeeze, dim=1)  # avg on slice
            elif self.aggregation == 'COS_SIM':
                v, slice_scores = self.COS_SIM(h_squeeze, xr_slice)
            elif self.aggregation == 'ATTENTION':
                v, slice_scores = self.ATTENTION(h_squeeze, xr_slice)
            return v, slice_scores, region_atts
        else: #单影像模态的输入
            v = torch.mean(h_squeeze, dim=1)  # avg on slice
            return v



