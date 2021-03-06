import torch
import torch.nn as nn
from Layers import ImputationBlock
from modules import DTransformer,ETransformer


# input:c[C,T]
# output: c_F[C,out_T_dim]
class Decoder(nn.Module):
    def __init__(self, C, T, out_T_dim, en_num_layers, dec_num_layers, embed_size, heads, map_dim):
        super(Decoder, self).__init__()
        self.ib = ImputationBlock(C, T)
        self.dtransformer1 = DTransformer(T, out_T_dim, en_num_layers, dec_num_layers, embed_size, heads, map_dim)
        self.dtransformer2 = DTransformer(T, out_T_dim, en_num_layers, dec_num_layers, embed_size, heads, map_dim)

    def forward(self, x_miss, y, c, m, delta):
        """

        x_miss :[C,T]更新缺失数据时，T=out_T_dim
        y 最终预测目标序列:[C,out_T_dim]
        c 是序列信息:[C,T]
        P 是更新了缺失值的序列:[C, T]
        """
        x_B, p_B = self.ib(x_miss, c, m, delta)
        c_B = self.dtransformer1(p_B, x_miss)  # [C,T]
        # print(c_B.shape)

        x_F, p_F = self.ib(x_B, c_B, m, delta)
        c_F = self.dtransformer2(p_F, y)  # [C, out_T_dim]

        x_pred = (x_B + x_F) / 2
        return x_B, x_F, x_pred, c_F
