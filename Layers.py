import torch
import torch.nn as nn
from entmax.activations import entmax15
import torch.nn.functional as F


# shape of input: [C, T, embed_size]
# shape of output: [C, T, embed_size]
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)

    def forward(self, x):
        C, T, E = x.shape

        x = x.view(C, T, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # scaled dot-product
        attn = entmax15(torch.matmul(queries, torch.transpose(keys, 2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [C, T, heads, heads]
        # print(attn.shape)
        out = torch.matmul(attn, values)  # [C, T, heads, per_dim]
        # print(out.shape)

        out = out.view(C, T, self.heads*self.per_dim)
        return out


# x_de: [C, T, embed_size]
# y_de: [C, out_T_dim, embed_size]
# out: [C, out_T_dim, embed_size]
class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads, T, out_T_dim):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.conv1 = nn.Conv2d(T, out_T_dim, 1)

    def forward(self, x_en, y_de):
        C, T, E = x_en.shape
        C_y, T_y, E_y = y_de.shape

        x = x_en.view(C, T, self.heads, self.per_dim)
        x = self.conv1(x)  # [C, out_T_dim, self.per_dim]
        y = y_de.view(C, T_y, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(y)
        keys = self.keys(x)
        values = self.values(x)

        # scaled dot-product
        attn = entmax15(torch.matmul(queries, torch.transpose(keys, 2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [C, T, heads, heads]
        out = torch.matmul(attn, values)  # [C, T, heads, per_dim]

        out = out.view(C, T_y, self.heads*self.per_dim)
        return out


"""
model = CrossAttention(512, 8, 20, 1)
x = torch.randn(1, 20, 512)
y = torch.randn(1, 1, 512)
out = model(x, y)
print(out.shape)
"""


# input:[C, out_T_dim, embed_size]
# output:[C, out_T_dim, embed_size]
"""
x_T是encoder输入序列时间维度
"""
class MaskMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MaskMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)

    def forward(self, y_de):
        C, T, E = y_de.shape

        y_de = torch.tril(y_de, diagonal=-1)
        x = y_de.view(C, T, self.heads, self.per_dim)
        # print(y_de)

        # compute queries, keys and values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # scaled dot-product
        attn = entmax15(torch.matmul(queries, torch.transpose(keys, 2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [C, out_T_dim, heads, heads]
        # print(attn.shape)
        out = torch.matmul(attn, values)  # [C,out_T_dim, heads, per_dim]
        # print(out.shape)

        out = out.view(C, T, self.heads * self.per_dim)
        return out


"""
model = MaskMultiHeadAttention(4, 2)
x = torch.randn(2, 3, 4)
out = model(x)

print(out.shape)
"""


# input: [C, T]
# output: [C, T]
class T2DCR(nn.Module):
    def __init__(self, C, T):
        super(T2DCR, self).__init__()
        self.fc1 = nn.Linear(T, T, bias=True)
        self.fc2 = nn.Linear(C, C, bias=True)

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))  # [C, T]
        x_2 = F.relu(self.fc2(x.transpose(0, 1)))  # [T, C]
        out = x_1 + x_2.transpose(0, 1)
        return out


"""
x = torch.randn(5, 20)
model = T2DCR(5, 20)
out = model(x)
print(out.shape)
"""


# input:[C,T]
# output:[C, T]
class ZLinear(nn.Module):
    def __init__(self, C, T):
        super(ZLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(C, C))
        self.bias = nn.Parameter(torch.randn(T))

    def forward(self, x):
        diag_ele = torch.diag_embed(torch.diagonal(self.weight, dim1=0, dim2=1))  # 对角元素
        W = self.weight - diag_ele  # weight对角元素为0
        # print(W)

        out = torch.matmul(W, x) + self.bias
        return out


"""
x = nn.Parameter(torch.randn(4, 3))
model = ZLinear(4, 3)
y =model(x)
print(y)
print(y.shape)


x = nn.Parameter(torch.randn(1, 3, 3))
y = torch.diagonal(x, dim1=1, dim2=2)
y_e = torch.diag_embed(y)
out = x - y_e
print(y_e)
print(x)
print(out)
"""


# input:[C,T]
# output:[C,T]
class ImputationBlock(nn.Module):
    def __init__(self, C, T):
        super(ImputationBlock, self).__init__()
        self.fc1 = nn.Linear(T, T, bias=True)
        self.fc2 = ZLinear(C, T)
        self.fc3 = nn.Linear(T, T, bias=True)
        self.fc4 = nn.Linear(T, T, bias=True)

    def forward(self, x, c, m, delta):
        """
        m 缺失标记:[C,T]
        x 原本序列， 有缺失:[C,T]
        c 是序列信息:[C,T]
        P 是更新了缺失值的序列:[C, T]
        """
        # print(c.shape)
        z_t = self.fc1(c)
        z_t_hat = m * c + (1 - m) * z_t

        out = self.fc2(z_t_hat)  # [C, T]
        namida = torch.sigmoid(F.relu(self.fc3(delta)) + F.relu(self.fc4(m)))

        # estimate x_hat
        x_hat = (1 - namida) * z_t + namida * out

        p = m * x + (1 - m) * x_hat
        return x_hat, p


"""
x = torch.randn(2, 3)
input = torch.randn(2,3)
model = ImputationBlock(2, 3, torch.tensor([[1.2, 0.2, 0.1], [0.9, 0.1, 0.2]], dtype=torch.float32))
y = model(x, input, torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32))
print(y.shape)
"""

