import torch
import numpy as np
import pandas as pd
from Model import Model
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma
    return x


if __name__ == '__main__':
    df = pd.read_csv("electricity.txt", header=None)
    df = df.values

    # Z-score
    df = Z_ScoreNormalization(df, df.mean(), df.std())

    """
    没有缺失值, 321个用户，每个用户26304条数据，每15min记录一次数据
    """
    datasets = torch.tensor(df, dtype=torch.float32).transpose(0, 1)
    # Z-score
    datasets = Z_ScoreNormalization(datasets, df.mean(), df.std())
    # print(datasets.shape)
    matrix = torch.tensor(np.ones_like(datasets))

    train_data = datasets[0:1, :]
    m_i = matrix[0:1, :]

    delta = torch.full((1, 26304), 0.25)

    # hyparameters:
    num_layers = 3
    T = 20
    out_T_dim = 1
    embed_size = 512
    heads = 8
    map_dim = 2048
    C = 1
    rou = 0.1
    lr = 0.0001
    epochs = 5

    model = Model(C, T, out_T_dim, embed_size, heads, num_layers, num_layers, map_dim).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    losses = []
    for epoch in range(epochs):
        for i in range(26304-21):
            x_miss, target = Variable(train_data[:, i:i+20]).cuda(), Variable(train_data[:, i+20:i+21]).cuda()
            m = Variable(m_i[:, i:i+20]).cuda()
            delt = Variable(delta[:, i:i+20]).cuda()

            x_B, x_F, x_pred, y_pred = model(x_miss, target, m, delt)

            # compute loss
            loss_bw = torch.sum(m * torch.norm(x_miss - x_B, dim=1))
            loss_fw = torch.sum(m * torch.norm(x_miss - x_F, dim=1))
            loss_consis = torch.sum(m * torch.norm(x_B - x_F), dim=1)
            loss_apporox = loss_bw + loss_fw + loss_consis

            loss_foreca = torch.mean(criterion(y_pred, target), dim=-1)

            loss_total = rou * loss_apporox + (1 - rou) * loss_foreca
            losses.append(loss_total)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            print("Epoch{}:LOSS:{}".format(epoch, loss_total.item()))

plt.plot(losses)
plt.show()



