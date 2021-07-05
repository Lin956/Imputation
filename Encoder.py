import torch
import torch.nn as nn
from modules import ETransformer
from Layers import T2DCR


# input(x,m): [C, T, 2], C is the number of sensor.
# output: x[C, T],m[C,T]                                                                       ] 为缺失数据指示，为1或者0, 0为此处数据缺失
class Encoder(nn.Module):
    def __init__(self, C, T, en_num_layers, embed_size, heads, map_dim):
        super(Encoder, self).__init__()
        self._2dcr = T2DCR(C, T)
        self.etransformer = ETransformer(en_num_layers, embed_size, heads, map_dim)

    def forward(self, input):
        C, T, _ = input.shape

        """处理input"""
        m = input[:, :, 1:2].view(C, T)
        # print(m)
        x = m * input[:, :, 0:1].view(C, T) + (1 - m) * self._2dcr(input[:, :, 0:1].view(C, T))  # [C, T]

        # Transformer提取序列信息
        information = self.etransformer(x)  # [C, T, embed_size)
        return information, m


"""
x = torch.randn(3, 20, 2)
model = Encoder(3, 20, 1, 512, 4, 2048)
Y, _ = model(x)
print(Y.shape)
"""



