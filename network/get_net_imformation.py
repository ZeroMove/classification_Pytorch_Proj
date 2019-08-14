# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified

import numpy as np
import torch.nn as nn
import torch


# count model's total parameters 返回参数的个数
# todo； 这个算参方式准确吗？
def count_parameters(model):
    model.parameters()
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in list(model.parameters()))


# estimate GPU-used pace
# 默认数据类型是float32, 所以type_size=4
# 该函数暂时不打算用了，只能调用到init()函数中定义的层，复杂一点的网络可能也无法计算；
def gpu_used_estimate(model, input_tensor, type_size=4):
    for batch_index, (inputs, targets) in enumerate(input_tensor):
        if batch_index == 0:
            input = inputs
            break
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    # input_ = input.type(torch.FloatTensor)
    input_.requires_grad_(requires_grad=False)
    # input_ = input_.cpu()
    input_ = input_.to('cuda')

    mods = list(model.modules())
    out_sizes = []

    for i in range(2, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            # 激活函数Relu()有一个默认参数inplace，默认设置为False，
            # 当设置为True时，我们在通过relu()计算时的得到的新值不会占用新的空间而是直接覆盖原来的值，
            # 这也就是为什么当inplace参数设置为True时可以节省一部分内存的缘故。
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))

    return total_nums * type_size * 2 / 1000 / 1000 + para * type_size / 1000 / 1000
