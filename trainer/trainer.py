# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.6.9
# state: modified

import torch.backends.cudnn as cudnn
import os
import time
import shutil
from network.get_net_imformation import *
from models import *
from datasets.data_preprocess import *
from tensorboardX import SummaryWriter
from proj_logs.create_logger import *


class Trainer():
    def __init__(self, output_path=None, config=None):
        # output_path
        self.output_path = output_path
        # read config parameters
        self.config = config
        # start a training log
        self.logger = Logger(log_file_name=output_path + '/log.txt',
                             log_level=logging.DEBUG, logger_name="").get_log()
        # start a SummaryWriter
        self.writer = SummaryWriter(log_dir=self.output_path + '/event')
        # define network
        self.net = get_model(self.config)
        # set device
        self.device = 'cuda' if self.config.use_gpu else 'cpu'
        # best model point to save
        self.best_prec = 0

    def start_train_and_val(self):
        # logging config para, net, total parameters todo: how to cal
        self.logger.info(self.config)
        self.logger.info(self.net)
        self.logger.info(" == total parameters: " + str(count_parameters(self.net)))

        if self.device == 'cuda':
            # todo: default is multiple-GPU
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        # load model
        self.net.to(self.device)
        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), self.config.lr_scheduler.base_lr,
                                    momentum=self.config.optimize.momentum,
                                    weight_decay=self.config.optimize.weight_decay,
                                    nesterov=self.config.optimize.nesterov)
        # resume from a checkpoint
        last_epoch = -1
        ckpt_file_name = self.output_path + '/' + self.config.ckpt_name + '.pth.tar'
        # todo: check if can read
        if self.config.use_checkpoint and os.path.exists(ckpt_file_name):
            self.best_prec, last_epoch = self.load_checkpoint(ckpt_file_name, self.net, optimizer=optimizer)
        # load training data, do data augmentation and get data loader
        transform_train = transforms.Compose(
            data_augmentation(self.config, w=self.config.input_size_w, h=self.config.input_size_h))
        transform_test = transforms.Compose(
            data_augmentation(self.config, is_train=False, w=self.config.input_size_w, h=self.config.input_size_h))
        train_loader, test_loader = get_data_loader(transform_train, transform_test, self.config)
        # self.logger.info(" == estimate training GPU-used space: " + str(gpu_used_estimate(self.net, train_loader)) + 'M')
        self.logger.info("            =======  Training  =======\n")
        # start train and val
        for epoch in range(last_epoch + 1, self.config.epochs):
            lr = self.adjust_learning_rate(optimizer, epoch, self.config)
            self.writer.add_scalar('learning_rate', lr, epoch)
            self.train(train_loader, self.net, criterion, optimizer, epoch, self.device)
            if epoch == 0 or (epoch + 1) % self.config.eval_freq == 0 or epoch == self.config.epochs - 1:
                self.validate(test_loader, self.net, criterion, optimizer, epoch, self.device, self.best_prec)
        self.logger.info("======== Training Finished.   best_test_acc: {:.3f}% ========".format(self.best_prec))

    # -----------------------------------------------------------
    def train(self, train_loader, net, criterion, optimizer, epoch, device):
        start = time.time()
        # todo
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        self.logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, self.config.epochs))
        for batch_index, (inputs, targets) in enumerate(train_loader):
            # move tensor to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # zero the gradient buffers
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weight
            optimizer.step()
            # count the loss and acc
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if (batch_index + 1) % self.config.train_show_freq == 0:
                self.logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                    batch_index + 1, len(train_loader),
                    train_loss / (batch_index + 1), 100.0 * correct / total, self.get_current_lr(optimizer)))
        end = time.time()
        self.logger.info("   == cost time: {:.4f}s".format(end - start))
        # todo
        train_loss = train_loss / (batch_index + 1)
        train_acc = correct / total
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)
        return train_loss, train_acc

    def validate(self, test_loader, net, criterion, optimizer, epoch, device, best_prec):
        best_prec = best_prec
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        self.logger.info(" === Validate ===".format(epoch + 1, self.config.epochs))

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                # print(outputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        self.logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
            test_loss / (batch_index + 1), 100.0 * correct / total))
        test_loss = test_loss / (batch_index + 1)
        test_acc = correct / total
        self.writer.add_scalar('test_loss', test_loss, epoch)
        self.writer.add_scalar('test_acc', test_acc, epoch)
        # Save checkpoint.
        acc = 100. * correct / total
        state = {
            'state_dict': net.state_dict(),
            'best_prec': best_prec,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        is_best = acc > best_prec
        self.save_checkpoint(state, is_best, self.output_path + '/' + self.config.ckpt_name)
        if is_best:
            self.best_prec = acc

    def load_checkpoint(self, checkpoint_path=None, model=None, optimizer=None):
        if os.path.isfile(checkpoint_path):
            # logging.info("=== loading checkpoint '{}' ===".format(path))

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            if optimizer != None:
                best_prec = checkpoint['best_prec']
                last_epoch = checkpoint['last_epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                # logging.info("=== done. also loaded optimizer from checkpoint '{}' (epoch {}) ===".format(
                #     path, last_epoch + 1))
                return best_prec, last_epoch

    def save_checkpoint(self, state, is_best, filename):
        torch.save(state, filename + '.pth.tar')
        if is_best:
            shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')

    def get_current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def adjust_learning_rate(self, optimizer, epoch, config):
        lr = self.get_current_lr(optimizer)
        if config.lr_scheduler.type == 'STEP':
            if epoch in config.lr_scheduler.lr_epochs:
                lr *= config.lr_scheduler.lr_mults
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
