import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

from networks.dinknet import DUNet


BATCHSIZE_PER_CARD = 2

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img(self, img, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_1(img)
        elif batchsize >= 4:
            return self.test_one_img_2(img)
        elif batchsize >= 2:
            return self.test_one_img_4(img)

    def test_one_img_8(self, img):
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

    def test_one_img_4(self, img):
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
    
    def test_one_img_2(self, img):
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
    
    def test_one_img_1(self, img):
        
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


if __name__=="__main__":
    testdir = "~/data/deepglobe/test/sat/"
    test = os.listdir(testdir)
    solver = TTAFrame(DUNet)
    # solver = TTAFrame(Unet)
    # solver.load('weights/dlinknet.th')
    solver.load('~/pyprojects/RoadNet/boost_train/weights/deepglobe_roadnet_2.th')
    tic = time()
    target = '~/data/deepglobe/out/boosting/tiny_boost2/'
    if os.path.isdir(target):
        pass
    else:
        os.makedirs(target)

    window_size=256
    step_length = 256

    for i, name in enumerate(test):
        if i%10 == 0:
            print(i/10, '    ','%.2f'%(time()-tic))
        img=cv2.imread(testdir + name)
        image_size=img.shape[0]
        mask=np.zeros((1024,1024))

        for start_row in range(0, image_size, step_length):
            for start_col in range(0, image_size, step_length):
                tiny_sat = img[start_row:start_row + window_size, start_col:start_col + window_size]
                tiny_mask = solver.test_one_img(tiny_sat)
                mask[start_row:start_row + window_size, start_col:start_col + window_size]=tiny_mask

        # mask[mask > 4] = 255
        # mask[mask <= 4] = 0

        # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

        # generate gray predicted result
        maximum = mask.max()
        minimum = mask.min()
        # mask = (mask - minimum) / (maximum - minimum)
        mask = ((mask - minimum) / 8) * 255
        # print("max={},min={},mask={}".format(maximum,minimum,mask))

        # img = cv2.imread(testdir+name)
        # # img = Image.open(testdir+name)
        # mask = dense_crf(np.array(img).astype(np.uint8), mask)

        # mask[mask>0.5] = 255
        # mask[mask<=0.5] = 0

        cv2.imwrite(target+name[:-7]+'mask.png', mask.astype(np.uint8))
