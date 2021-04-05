import torch
import torch.nn as nn
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import os
from loss import loss_func, dice_bce_loss

class MyFrame():
    def __init__(self, net, lr, name, evalmode = False):
        self.model = net
        self.cuda_net = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        if evalmode:
            for i in self.model.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        self.isTrain = True
        self.num_classes = 1
        self.tensorborad_dir = "log/tensorboard_log/"
        self.model_dir = "weights/"
        # self.lr = 0.007
        self.lr = lr
        self.lr_power = 0.9
        self.momentum = 0.9
        self.wd = 0.0001  # weight decay
        self.accum_steps = 1
        self.iterSize = 10
        self.net_name = name

        self.which_epoch = 0

        # self.device =
        if self.isTrain:
            # self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda() # maybe edit
            # Change the crossentropyloss to BCEloss
            # self.criterionSeg = torch.nn.BCELoss().cuda()
            # self.criterionSeg = loss_func().cuda()
            self.criterionSeg = dice_bce_loss().cuda()
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.wd)
            params_w = list(self.model.decoder.dupsample.conv_w.parameters())
            params_p = list(self.model.decoder.dupsample.conv_p.parameters())
            self.optimizer_w = torch.optim.SGD(params_w + params_p, lr=self.lr, momentum=self.momentum)
            self.old_lr = self.lr
            self.averageloss = []

            self.writer = SummaryWriter(self.tensorborad_dir)
            self.counter = 0

        self.model.cuda()

        self.normweightgrad = 0.

        # if not self.isTrain and self.loaded_model != ' ':
        #     self.load_pretrained_network(self.model, self.opt.loaded_model, strict=True)
        #     print('test model load sucess!')

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        return self.img

    def forward(self, pre_compute_flag=0, isTrain=True):
        # self.img = V(self.img.cuda(), volatile=volatile)
        # if self.mask is not None:
        #     self.mask = V(self.mask.cuda(), volatile=volatile)
        accum_steps = self.accum_steps

        if pre_compute_flag == 1:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.wd)

            print("pre compute OK")

        self.img.requires_grad = not isTrain

        if self.mask is not None:
            self.seggt = self.mask.cuda()
        else:
            self.seggt = None

        self.segpred = self.cuda_net.forward(self.img)
        if self.isTrain:
            self.loss = self.criterionSeg(self.segpred, self.seggt) / accum_steps
            self.averageloss += [self.loss.data * accum_steps]

        loss_pred, loss_vgg = 0., 0.
        for i in range(len(results)):
            loss_pred += (i + 1) * self.loss_pred(labels, results[i][0])
            for j in range(len(gt_vgg)):
                loss_vgg += (i + 1) * self.loss_topo(gt_vgg[j], results[i][1][j])
        coeff = 0.5 * len(results) * (len(results) + 1)
        curr_loss_pred = loss_pred / coeff
        curr_loss_vgg = loss_vgg / coeff

        train_loss = curr_loss_pred + 0.1 * curr_loss_vgg
        train_loss.backward()

        # self.seggt = torch.squeeze(self.seggt, dim=1)
        if isTrain:
            self.loss.backward()
        return self.averageloss
        
    def optimize(self,precompute_flag=0):
        self.forward()
        self.optimizer.zero_grad()
        loss_list=self.forward(pre_compute_flag=precompute_flag)
        self.optimizer.step()
        return sum(loss_list)/len(loss_list)

    def pre_compute_W(self, i):
        self.model.zero_grad()
        self.seggt = self.mask  # N 1 H W
        N, channel, H, W = self.seggt.size()
        C = self.num_classes
        scale = self.model.decoder.dupsample.scale
        # N, C, H, W
        # self.seggt = torch.squeeze(self.seggt, dim=1)
        #
        # self.seggt[self.seggt == 0] = 0
        # self.seggt_onehot = torch.zeros(N, C, H, W).scatter_(1, self.seggt, self.seggt)
        self.seggt_onehot = self.seggt
        # N, H, W, C
        self.seggt_onehot = self.seggt_onehot.permute(0, 2, 3, 1)
        # N, H, W/sacle, C*scale
        self.seggt_onehot = self.seggt_onehot.contiguous().view((N, H,
                                                                 int(W / scale), C * scale))
        # N, W/sacle, H, C*scale
        self.seggt_onehot = self.seggt_onehot.permute(0, 2, 1, 3)
        # N, W/scale, H/scale, C*scale*scale
        self.seggt_onehot = self.seggt_onehot.contiguous().view((N, int(W / scale),
                                                                 int(H / scale), C * scale * scale))
        # N, C*scale*scale, H/scale, W/scale
        self.seggt_onehot = self.seggt_onehot.permute(0, 3, 2, 1)

        self.seggt_onehot = self.seggt_onehot.cuda()

        self.seggt_onehot_reconstructed = self.model.decoder.dupsample.conv_w(
            self.model.decoder.dupsample.conv_p(self.seggt_onehot))
        self.reconstruct_loss = torch.mean(torch.pow(self.seggt_onehot -
                                                     self.seggt_onehot_reconstructed, 2))
        self.reconstruct_loss.backward()
        self.optimizer_w.step()
        if i % 200 == 0: # output per 200 iters
            print('pre_compute_loss: %f' % (self.reconstruct_loss))

    def save(self, path):
        torch.save(self.cuda_net.state_dict(), path)
        
    def load(self, path):
        dict=torch.load(path)
        self.cuda_net.load_state_dict(dict)
    
    def update_lr_poly(self, step, total_step, mylog, th):
        # poly learning rate update
        if step <= th:
            new_lr = max(self.lr * (step / th) ** self.lr_power, 1e-6)
        else:
            new_lr = max(self.lr * (1 - step / total_step) ** self.lr_power, 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def update_tensorboard(self, step):
        if self.isTrain:
            # self.writer.add_scalar(self.net_name + '/Accuracy/', data[0], step)
            # self.writer.add_scalar(self.net_name + '/Accuracy_Class/', data[1], step)
            # self.writer.add_scalar(self.net_name + '/Mean_IoU/', data[2], step)
            # self.writer.add_scalar(self.net_name + '/FWAV_Accuracy/', data[3], step)

            self.trainingavgloss = sum(self.averageloss)/len(self.averageloss)
            self.writer.add_scalars(self.net_name + '/loss', {"train": self.trainingavgloss}, step)

            self.writer.add_scalar("learning rate",self.old_lr,step)

            # file_name = os.path.join(self.save_dir, 'MIoU.txt')
            # with open(file_name, 'wt') as opt_file:
            #     opt_file.write('%f\n' % (data[2]))
            # self.writer.add_scalars('losses/'+self.opt.name, {"train": self.trainingavgloss,
            #                                                  "val": np.mean(self.averageloss)}, step)
            self.averageloss = []
    def close_tensorboard(self):
        self.writer.close()


    def name(self):
        return 'DUNet'

