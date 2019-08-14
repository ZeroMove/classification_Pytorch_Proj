# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.6.9
# state: half modified

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class EasyDataset():
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            # print(words[0], int(words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        # plt.imshow(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class cls_257_Dataset():
    def __init__(self, mode, csv, transform=None, target_transform=None, loader=default_loader):
        fh = open(csv, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()  # 删除尾部空格
            words = line.split(',')
            if words[0] != 'Name':
                if mode == 'train':
                    imgs.append(
                        ('./datasets/cls_257/TrainSet/' + words[0][1:-1].replace('\\', '/'), int(int(words[1]) - 1)))
                else:
                    imgs.append(
                        ('./datasets/cls_257/TestSet/' + words[0][1:-1].replace('\\', '/'), int(int(words[1]) - 1)))
                # print(words[0], int(int(words[1])-1))
            # print(words[0], int(words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        # plt.imshow(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# TODO： 想把数据增强独立成一个文件
def data_augmentation(config, is_train=True, w=32, h=32):
    aug = []
    # 不论训练还是测试，先resize
    aug.append(transforms.Resize((w, h)))
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            # padding=4 todo: make it changeable
            # 我的理解是先填充，再任意裁剪到原来大小
            aug.append(transforms.RandomCrop(config.input_size_w, padding=config.aug_padding))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == 'cifar10':
            aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        else:
            aug.append(transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    # if is_train and config.augmentation.cutout:
    #     # cutout
    #     aug.append(Cutout(n_holes=config.augmentation.holes,
    #                       length=config.augmentation.length))
    return aug


def get_data_loader(transform_train, transform_test, config):
    # 先写好数据集逐个获取的接口，再使用多进程按要求获取batch的数据
    # assert config.dataset == 'cifar10' or config.dataset == 'cifar100'
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == "easy":
        trainset = EasyDataset(txt=config.data_path + 'easy_train.txt', transform=transform_train)
        testset = EasyDataset(txt=config.data_path + 'easy_test.txt', transform=transform_test)
    elif config.dataset == "cls_257":
        trainset = cls_257_Dataset(mode='train', csv=config.data_path + 'TrainSet/TrainSetLabels.csv',
                                   transform=transform_train)
        testset = cls_257_Dataset(mode='val', csv=config.data_path + 'TestSet/TestSetLabels.csv',
                                  transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, download=True, transform=transform_test)

    # num_workers:使用多进程加载的进程数，0代表不使用多进程
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers)

    return train_loader, test_loader
