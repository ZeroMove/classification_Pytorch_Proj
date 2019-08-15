# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet']


class LeNet(nn.Module):
    def __init__(self, num_classes=10, inputsize_w=32, inputsize_h=32):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        # pooling：14  14-5+1=10
        self.conv_2 = nn.Conv2d(6, 16, 5)
        # pooling：5
        self.fc_1 = nn.Linear(int(16 * ((((inputsize_w - 4) / 2) - 4) / 2) * ((((inputsize_h - 4) / 2) - 4) / 2)), 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)
        # (((inputsize-4）/2)-4)/2

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


def lenet(pretrained, num_classes, inputsize_w, inputsize_h):
    # lenet have no pretained models, or i can't find
    return LeNet(num_classes=num_classes, inputsize_w=inputsize_w, inputsize_h=inputsize_h)
