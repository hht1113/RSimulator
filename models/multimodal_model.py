import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.text_model import RepEncoder
from models.image_model import ImageEncoder
import torch

class Multimodal_Net(nn.Module):
    def __init__(self,
                 image_model_name='resnet18',
                 report_model_name='chinese-roberta-wwm-ext',  # 'bert-base-chinese', 'chinese-roberta-wwm-ext'
                 img_aggregation_type='ViT',  # 'AVG', '3D', 'ViT', 'COS_SIM', 'ATTENTION'
                 input_W=128,  # width of slice
                 input_H=128,  # height of slice
                 input_D=24,  # slice number
                 multimodal_dim=512,
                 mhsa_heads=8,
                 dropout_rate=0.1,
                 mask_columns=2,
                 bias=False, #为什么是False
                 channels=5, #为什么不是1
                 nb_class=2,
                 freeze_layers=[0, 1, 2, 3, 4, 5], # 去掉冻结层，改成全参量微调 我把冻结的代码给注释掉了
                 fc_type='img+rep',  # 'img_only', 'rep_only', 'img+rep'
                 concate_type='direct',  # 'direct', 'proj'
                 student=False
                 ):
        super(Multimodal_Net, self).__init__()

        self.img_aggregation_type = img_aggregation_type
        # init image encoder
        self.Img_model = ImageEncoder(image_model=image_model_name,
                                      aggregation_type=img_aggregation_type,
                                      H=input_H,
                                      W=input_W,
                                      D=input_D,
                                      channels=channels,
                                      mm_dim=multimodal_dim,
                                      num_class=nb_class,
                                      num_heads=mhsa_heads,
                                      bias=bias,
                                      dropout_rate=dropout_rate,
                                      mask_columns=mask_columns,
                                      )

        # init report encoder
        self.Rep_model = RepEncoder(rep_model=report_model_name, freeze_layers=freeze_layers)

        # init fusion and prediction model
        self.Predict_model = Classifier(img_outputdim=self.Img_model._get_img_dim(), #指定为512
                                        rep_output_dim=512,
                                        multimodal_dim=multimodal_dim,
                                        bias=bias,
                                        num_class=nb_class,
                                        fc_type=fc_type,
                                        concat_type=concate_type,
                                        aggregation_type=img_aggregation_type,
                                        )
        
        self.fc_type = fc_type

    def forward(self, xis=None, xrs_encoded_inputs=None):  # 输入时xrs和xds输入相同
        '''
        xis: input image (batchsize, slice, C, H, W)
        xrs_encoded_inputs: report after tokenizing
        '''
        if self.fc_type == 'img+rep':
            # Encoding
            # REP CLS-embedding
            xre, _, _ = self.Rep_model(xrs_encoded_inputs)

            # IMG 用xde交互得到xie
            xie, slice_scores, region_atts = self.Img_model(xis, xr_slice=xre)

            # Interaction
            z, fusion_feature = self.Predict_model(xie, xre)  # xre是整个报告的特征 xie是加权后的影像特征
            # z = self.Predict_model(xie, xde)
            return z, slice_scores, region_atts, fusion_feature
        
        elif self.fc_type == 'img_only':
            xie = self.Img_model(xis)
            z, fusion_feature = self.Predict_model(zie=xie)
            return z, fusion_feature # z就是logits
            
        elif self.fc_type == 'rep_only':
            xre, _, _ = self.Rep_model(xrs_encoded_inputs)
            z = self.Predict_model(zre=xre)
            return z, None, None
        
# Fusion and Prediction
class Classifier(nn.Module):
    def __init__(self, img_outputdim, rep_output_dim, multimodal_dim, bias, num_class, fc_type='img+rep',
                 concat_type='direct',aggregation_type='ViT'):
        super(Classifier, self).__init__()

        self.img_dim = img_outputdim
        self.rep_dim = rep_output_dim
        self.mm_dim = multimodal_dim
        self.bias = bias
        self.num_class = num_class
        self.fc_type = fc_type
        self.concat_type = concat_type
        self.aggregation_type = aggregation_type

        # PROJECTION MATRICES
        self.proj_img = nn.Linear(self.img_dim, self.mm_dim, bias=self.bias)
        self.proj_rep = nn.Linear(self.rep_dim, self.mm_dim, bias=self.bias)

        # FCs
        self.FC_vit = nn.Sequential( 
            nn.Linear(768, self.num_class),
        )

        # FC for img_only baselines
        self.FC_img = nn.Sequential( 
            nn.Linear(self.img_dim, self.num_class),
        )
        # FC for rep_only baselines
        self.FC_rep = nn.Sequential(
            nn.Linear(self.rep_dim, self.num_class),
        )
        # FC for multi-modal baselines
        # direct 直接torch.cat
        self.FC_mm = nn.Sequential(
            nn.Linear(self.img_dim + self.rep_dim, self.num_class),
        )
        # proj 投射后再相连
        self.FC_mm_proj = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.num_class),
        )
        # no_use
        self.MLP_mm_proj = nn.Sequential(
            nn.Linear(self.mm_dim + self.mm_dim, self.mm_dim),
            nn.BatchNorm1d(num_features=self.mm_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.mm_dim, self.num_class),
        )

    def forward(self, zie=None, zre=None):
        z = None

        if self.fc_type == 'img_only':
            if self.aggregation_type=='ViT':
                z = self.FC_vit(zie)
                # z = zie
                fusion_feature = z
            else:
                z = self.FC_img(zie)
                fusion_feature = z

        elif self.fc_type == 'rep_only':
            z = self.FC_rep(zre)
        elif self.fc_type == 'img+rep' and zre!=None:
            if self.concat_type == 'direct':
                #print('zie,zre:', zie.shape, zre.shape)
                try:
                    fusion_feature = torch.cat([zie, zre], dim=-1)
                except Exception:
                    zie = torch.unsqueeze(zie,0)
                    fusion_feature = torch.cat([zie, zre], dim=-1)
                z = self.FC_mm(fusion_feature)
            elif self.concat_type == 'proj':
                zim = self.proj_img(zie) # 512
                fusion_feature = torch.cat([zim, zre], dim=-1)
                z = self.FC_mm_proj(fusion_feature)
                # z = self.MLP_mm_proj(z_)
            else:
                print('wrong value of concat_type')
        else:
            print('wrong value of fc_type')

        return z, fusion_feature # shape: (batchsize,class)