# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified
# 说明： 该init文件是模型调用该可
#       包文件中模块的中继站,import的
#       时候也需要一同导入

from .lenet import *
from .alexnet import *
from .resnet_152 import *


# from .vgg import *
# from .resnet import *
# from .preresnet import *
# from .senet import *
# from .resnext import *
# from .densenet import *
# from .shake_shake import *
# from .sknet import *
# from .genet import *
# from .cbam_resnext import *


def get_model(config):
    # todo
    return globals()[config.architecture](config.if_pretrain, config.num_classes, config.input_size_w,
                                          config.input_size_h)
    # return config.architecture
