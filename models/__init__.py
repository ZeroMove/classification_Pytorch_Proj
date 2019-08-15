# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified


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
