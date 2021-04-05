import torch

import torch.utils.data as data

import os
from time import time
from networks.dinknet import DUNet
from framework import MyFrame
from data import ImageFolder

SHAPE = (256, 256)
ROOT = 'E:/shao_xing/tiny_dataset/new0228/tiny_sat_lab/'
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
NAME = 'ratio_16'
BATCHSIZE_PER_CARD = 2

solver = MyFrame(DUNet, lr=0.00005)
# solver = MyFrame(Unet, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

mylog = open('log/' + NAME + '_finetune.log', 'a')
tic = time()
no_optim = 0
total_epoch = 100
train_epoch_best_loss = 100.

solver.load('weights/ratio_16.th')
print('* load existing model *')

epoch_iter = 0

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

    print("train best loss is {}".format(train_epoch_best_loss))
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '_finetune.th')
        print("save new model")
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '_finetune.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
        # solver.update_lr_poly(epoch,total_epoch, mylog)
    # if epoch%10==0:
    #     solver.update_lr_poly(epoch, total_epoch, mylog)
    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()