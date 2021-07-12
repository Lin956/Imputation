from Encoder import Encoder
from Decoder import Decoder
import torch
import torch.nn as nn

"""
c:提取序列信息
x_c,x_F 预测缺失值
y_pred 预测值

本模型未考虑空间信息

encoder:input[C,T],m[C,T], output序列信息x[C,T]
decoder:input:   x 原本序列, 未缺失:[C,T]
        c 是序列信息:[C,T]
        output:P 是更新了缺失值的序列:[C, T], c 更新序列信息[C,T]
        
整个模型流程：
encoder预训练，提取序列信息c:[C, T]
decoder：第一阶段输入c,ib更新缺失数据x_b，transformer提取序列信息，更新缺失数据
         第二阶段ib更新缺失数据x_f, transformer提取序列信息c_F:[C, T]
预测层：输入c-F,两层卷积预测
"""


# x_miss:[C, T], y：[C,out_T_dim]目标预测序列  更新缺失数据时，T=out_T_dim
# output y_pred: [C,T]
class Model(nn.Module):
    def __init__(self, C, T, out_T_dim, embed_size, heads, en_num_layers, dec_num_layers, map_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(C, T, en_num_layers, embed_size, heads, map_dim)
        self.decoder = Decoder(C, T, out_T_dim, en_num_layers, dec_num_layers, embed_size, heads, map_dim)
        self.conv1 = nn.Conv1d(1, embed_size, 1)
        self.conv2 = nn.Conv1d(embed_size, 1, 1)

    def forward(self, x_miss, y, m, delta):
        # encoder阶段
        c = self.encoder(x_miss, m)  # [C,T], [C,T]
        # print(c.shape)
        # decoder阶段
        x_B, x_F, x_pred, c_f = self.decoder(x_miss, y, c, m, delta)  # [C,T], [C,T], [C,T], [C,out_T_dim]
        # print(c_f.shape)

        # 预测阶段, 用c_F预测
        c_f = c_f.unsqueeze(2)  # [C, out_T_dim, 1]
        c_f = c_f.permute(0, 2, 1)  # [C, 1, out_T_dim]
        c_f = self.conv1(c_f)  # [C, embed_size, out_T_dim]
        y_pred = self.conv2(c_f)  # [C, 1, out_T_dim]
        y_pred = y_pred.permute(0, 2, 1)
        y_pred.squeeze(2)

        return x_B, x_F, x_pred, y_pred
