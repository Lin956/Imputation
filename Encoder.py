import torch
import torch.nn as nn
from modules import ETransformer
from Layers import T2DCR


# input(x,m): x[C, T], C is the number of sensor.m:[C, T]
# output: x[C, T],m[C,T] 为缺失数据指示，为1或者0, 0为此处数据缺失
class Encoder(nn.Module):
    def __init__(self, C, T, en_num_layers, embed_size, heads, map_dim):
        super(Encoder, self).__init__()
        self._2dcr = T2DCR(C, T)
        self.etransformer = ETransformer(en_num_layers, embed_size, heads, map_dim)

    def forward(self, input, m):

        """处理input"""
        x = m * input + (1 - m) * self._2dcr(input)  # [C, T]

        # Transformer提取序列信息
        information = self.etransformer(x)  # [C, T]
        return information


"""
x = torch.randn(2, 3)
m = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.float32)
model = Encoder(2, 3, 1, 512, 4, 2048)
Y = model(x, m)
print(Y.shape)
"""




