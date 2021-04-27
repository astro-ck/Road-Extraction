import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import argparse
import json
from time import time
from datetime import datetime

from dlinknet import DLinkNet34, MyFrame, dice_bce_loss
from datainput import ImageFolder

BATCHSIZE_PER_CARD = 2


def train(strDataPath, strModelPath, strOutPath):
    print("......Training......")
    tic = time()
    timestamp = datetime.fromtimestamp(tic).strftime('%Y%m%d-%H:%M')
    NAME = 'road_' + timestamp
    print(NAME)
    SHAPE = (1024, 1024)
    sat_dir = strDataPath + '/train/sat/'
    lab_dir = strDataPath + '/train/lab/'
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(sat_dir))
    trainlist = map(lambda x: x[:-8], imagelist)

    solver = MyFrame(DLinkNet34, dice_bce_loss, 5e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(trainlist, sat_dir, lab_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)
    if os.path.isdir(strOutPath + '/logs/'):
        pass
    else:
        os.makedirs(strOutPath + '/logs/')

    mylog = open(strOutPath + '/logs/' + NAME + '.log', 'w')
    no_optim = 0
    total_epoch = 300
    train_epoch_best_loss = 100.
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('SHAPE:', SHAPE, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save(strModelPath + '/' + NAME + '.th')
        if no_optim > 6:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load(strModelPath + '/' + NAME + '.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        # img1 = np.concatenate([img[None],img[None]])
        # img1 = img1.transpose(0,3,1,2)
        # img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        # maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        # return maska[0,:,:]+maska[1,:,:]
        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def test(strDataPath, strModelPath, strOutPath):
    print("......Testing......")
    testdir = strDataPath
    test = os.listdir(testdir)
    solver = TTAFrame(DLinkNet34)
    print("WARNING: please give the path to the model you want to test")
    print("e.g., /road/model/road_20200421-11:28.th")
    solver.load(strModelPath)
    tic = time()
    target_grey = strOutPath + '/results/'
    if os.path.isdir(target_grey):
        pass
    else:
        os.makedirs(target_grey)

        for i, name in enumerate(test): # !!!!!!
        # if i % 10 == 0:
            print(i/10, '    ', '%.2f' % (time()-tic))
        mask = solver.test_one_img_from_path(testdir + name)
        mask_binary = mask.copy()
        mask_binary[mask_binary > 4] = 255
        mask_binary[mask_binary <= 4] = 0

        # # generate gray predicted result
        # mask[mask < 0] = 0
        # mask = (mask / 8) * 255

        # cv2.imwrite(target_grey+name[:-7]+"pred.png", mask.astype(np.uint8))
        cv2.imwrite(target_grey+name[:-7]+"mask.png", mask_binary.astype(np.uint8))


if __name__=="__main__":
    dic = {"task":"test",
    "data":"data/test/",
    "model":"model/massa_dlinknet.th",
    "out":"result/"}

    if 'data' in dic:
        strDataPath = dic['data']
    else:
        strDataPath = '/road/data/'

    if 'model' in dic:
        strModelPath = dic['model']
    else:
        strModelPath = '/road/model/'

    if 'out' in dic:
        strOutPath = dic['out']
    else:
        strOutPath = '/road/out/'

    if dic['task'] == 'train':
        train(strDataPath, strModelPath, strOutPath)
    if dic['task'] == 'test':
        test(strDataPath, strModelPath, strOutPath)

    # print(strDataPath)
    # print(strModelPath)
    # print(strOutPath)