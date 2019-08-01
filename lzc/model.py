import torch
import torch.nn as nn
import torch.nn.functional as F
import lzc.tcn as tcn


class MyModel(nn.Module):
    def __init__(self, in_chs, out_chs, k_size, layer_sz, classes):
        super(MyModel, self).__init__()
        self.cnn = nn.Sequential()
        [self.cnn.add_module('conv' + str(i), nn.Conv1d(in_chs[i], out_chs[i], k_size[i])) for i in range(layer_sz)]
        self.out_ch = out_chs
        self.fc = nn.Linear(out_chs[-1], classes)

    def forward(self, input, xxx):
        x = self.cnn(input)
        x = F.relu(x)
        x = x.sum(dim=-1).view([-1, self.out_ch[-1]])
        x = self.fc(x)
        return x


class TCNModel(nn.Module):
    def __init__(self, input_sz, num_channels, k_size, classes):
        super(TCNModel, self).__init__()
        self.cnn = tcn.TemporalConvNet(input_sz, num_channels, kernel_size=k_size)
        self.fc = nn.Linear(num_channels[-1], classes)
        self.out_ch = num_channels

    def forward(self, input, xxx):
        x = self.cnn(input)
        # print(x.size())
        x = F.relu(x)
        x = x.sum(dim=-1).view([-1, self.out_ch[-1]])
        x = self.fc(x)
        return x
