import torch

from time import time
from networks.dinknet import DUNet
from framework import MyFrame
from data import ImageFolder
from datagathering.adaboost_dataset import read_image_name

SHAPE = (256, 256)

def train_operation(train_paras):
    sat_dir = train_paras["image_dir"]
    lab_dir = train_paras["gt_dir"]
    train_id = train_paras["train_id"]
    image_list_dir = train_paras["image_list_dir"]
    logfile_dir = train_paras["logfile_dir"]
    model_dir = train_paras["model_dir"]
    model_name = train_paras["model_name"]
    learning_rate = train_paras["learning_rate"]

    imagelist = read_image_name(image_list_dir + "train_image_file_" + str(train_id) + ".txt")

    trainlist = list(map(lambda x: x[:-8], imagelist))
    # trainlist = trainlist[:1000]
    BATCHSIZE_PER_CARD = 2
    solver = MyFrame(DUNet, learning_rate, model_name)
    # solver = MyFrame(Unet, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(trainlist, sat_dir, lab_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    mylog = open(logfile_dir + model_name + '.log', 'w')
    print("**************" + model_name + "******************", file=mylog)
    print("**************" + model_name + "******************")
    print("current train id:{}".format(train_id),file=mylog)
    print("current train id:{}".format(train_id))
    print("batch size:{}".format(batchsize),file=mylog)
    print("total images: {}".format(len(trainlist)))
    print("total images: {}".format(len(trainlist)), file=mylog)

    tic = time()
    no_optim = 0
    total_epoch = train_paras["total_epoch"]
    train_epoch_best_loss = 100.

    # solver.load('weights/dlinknet_new_lr_decoder.th')
    # print('* load existing model *')

    epoch_iter = 0
    print("learning rate is {}".format(learning_rate), file=mylog)
    print("Precompute weight for 5 epoches", file=mylog)
    print("Precompute weight for 5 epoches")
    save_tensorboard_iter = 5
    pre_compute_flag = 1
    # pretrain W
    for epoch in range(1, 6):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        if epoch < 5:
            no_optim = 0
            t = 0
            for img, mask in data_loader_iter:
                t += 1
                solver.set_input(img, mask)
                solver.pre_compute_W(t)
        print('********', file=mylog)
        print('pre-train W::', epoch, '    time:', int(time() - tic), file=mylog)
        print('********')
        print('pre-train W:', epoch, '    time:', int(time() - tic))

    print("pretrain is OVER")
    print("pretrain is OVER", file=mylog)
    step_update = False
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            imgs = solver.set_input(img, mask)
            train_loss = solver.optimize(pre_compute_flag)
            pre_compute_flag = 0
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
        if epoch % save_tensorboard_iter == 1:
            solver.update_tensorboard(epoch)
        # imgs=imgs.to(torch.device("cpu"))
        # solver.writer.add_graph(solver.model,imgs)
        print("train best loss is {}".format(train_epoch_best_loss))
        print("train best loss is {}".format(train_epoch_best_loss), file=mylog)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save(model_dir + model_name + '.th')
        if no_optim > 6:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        elif no_optim > 3:
            step_update = True
            solver.update_lr(5.0, factor=True, mylog=mylog)
            print("update lr by ratio 0.5")
        elif no_optim > 2:
            if solver.old_lr < 5e-7:
                break
            solver.load(model_dir + model_name + '.th')
            # solver.update_lr(5.0, factor=True, mylog=mylog)
            if step_update:
                solver.update_lr(5.0, factor=True, mylog=mylog)
            else:
                solver.update_lr_poly(epoch, total_epoch, mylog, total_epoch / 40)
        if not step_update:
            solver.update_lr_poly(epoch, total_epoch, mylog, total_epoch / 40)
        mylog.flush()

    solver.close_tensorboard()
    print('*********************Finish!***********************', file=mylog)
    print('Finish!')
    mylog.close()

if __name__=="__main__":
    image_list_dir = "E:/shao_xing/tiny_dataset/boost_train/"
    train_id = 2
    logfile_dir = "log/"
    model_dir = 'weights/'
    model_name = 'dlinknet_dupsample_test' + str(train_id)
    sat_dir = 'E:/shao_xing/tiny_dataset/D1/original/sat/'
    lab_dir = 'E:/shao_xing/tiny_dataset/D1/original/lab/'
    train_paras={"learning_rate":0.003,
                 "total_epoch":200,
                 "train_id":train_id,
                 "image_dir":sat_dir,
                 "gt_dir":lab_dir,
                 "image_list_dir":image_list_dir,
                 "logfile_dir":logfile_dir,
                 "model_dir":model_dir,
                 "model_name":model_name}
    train_operation(train_paras)
