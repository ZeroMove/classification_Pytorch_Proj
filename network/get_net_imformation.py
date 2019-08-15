# -*-coding:utf-8-*-
# coder: Jiawen Zhu
# date: 2019.7.13
# state: modified

import numpy as np
import torch.nn as nn


# count model's total parameters
def count_parameters(model):
    model.parameters()
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in list(model.parameters()))


# estimate GPU-used pace
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
