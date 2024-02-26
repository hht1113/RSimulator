import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel

# 指定模型文件和配置文件的路径
rep_model_path = '/data1/houhaotian/PyCharmCode/SimulatorNet_Idh/roberta_config/'

# Report Encoder
class RepEncoder(nn.Module):
    def __init__(self, rep_model, freeze_layers):
        super(RepEncoder, self).__init__()
        # init roberta
        self.roberta_model = self._get_rep_basemodel(rep_model, freeze_layers)
        self.linear = nn.Linear(768, 512)
    
    def _get_rep_basemodel(self, rep_model_name, freeze_layers):
        try:
            print("report feature extractor:", rep_model_name)
            if rep_model_name == 'bert-base-chinese':
                model = AutoModel.from_pretrained(rep_model_name)
            elif rep_model_name == 'chinese-roberta-wwm-ext':
                # 加载预训练模型
                model = BertModel.from_pretrained(rep_model_path)
                #model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

            print("--------Report pre-train model load successfully --------")
        except Exception as e:
            raise RuntimeError("Invalid model name or path") from e
        
        # if freeze_layers is not None: # 冻结最底下几层[0-5]
        #     for layer_idx in freeze_layers:
        #         for param in list(model.encoder.layer[layer_idx].parameters()):
        #             param.requires_grad = False

        return model

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, encoded_inputs):
        # encoded_inputs = encoded_inputs.to(device_)
        outputs = self.roberta_model(**encoded_inputs)
        mp_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
        cls_embeddings = outputs[1] # batch_size*768 16*768
        # 通过一个全连接层降维到指定的维度
        reduced_embeddings = self.linear(cls_embeddings)  # (batch_size, output_dim)
        token_embeddings = outputs[0]

        return reduced_embeddings, mp_embeddings, token_embeddings