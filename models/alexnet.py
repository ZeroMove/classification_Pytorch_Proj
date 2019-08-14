# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified

import torch.nn as nn
import torchvision.models as models

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes, inputsize_w=32, inputsize_h=32):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        self.output = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.output(x)
        return x


def alexnet(pretrained, num_classes, inputsize_w, inputsize_h):
    my_alexnet = AlexNet(num_classes=num_classes, inputsize_w=inputsize_w, inputsize_h=inputsize_h)

    if pretrained:
        # 是不是有更简单的加载方式 可以直接使用alexnet，改变最后一层即可，就行resnet
        alexnet = models.alexnet(pretrained=True)
        # 读取参数
        pretrained_dict = alexnet.state_dict()
        model_dict = my_alexnet.state_dict()

        # 将pretrained_dict里不属于model_dict的键剔除掉
        # todo: k,v
        # for k, v in pretrained_dict.items():
        #     print(k)
        # print('*****************************')
        # for k, v in model_dict.items():
        #     print(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        my_alexnet.load_state_dict(model_dict)

    return my_alexnet
