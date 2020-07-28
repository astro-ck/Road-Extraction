import numpy as np
import os
from test import TTAFrame
from networks.dinknet import DUNet
import cv2
import random

def read_image_prob(path):
    image_dict = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.strip().split(",")
            image_dict.setdefault((temp[0]), float(temp[1]))
    return image_dict


def read_image_name(path):
    image_name_list = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip()
            image_name_list.append(temp)
    return image_name_list


def generate_boosting_data(train_paras):
    """
    generate training data for next training
    """
    image_dir = train_paras["image_dir"]
    gt_dir = train_paras["gt_dir"]
    train_id = train_paras["train_id"]
    image_name_file_save_dir = train_paras["image_name_file_save_dir"]
    image_prob_file_save_dir = train_paras["image_prob_file_save_dir"]
    model_dir = train_paras["model_dir"]
    model_name = train_paras["model_name"]

    # set image probability for choosing
    image_list = read_image_name(image_name_file_save_dir + "train_image_file_"+ str(train_id) + ".txt")
    dataset_size = len(image_list)
    if train_id < 2:
        image_prob = dict.fromkeys(image_list, 1 / dataset_size)
    else:
        image_prob = read_image_prob(image_prob_file_save_dir + "image_prob_" + str(train_id) + ".txt")

    iou_dict = dict.fromkeys(image_list, 0)
    error_list = []
    total_error = 0
    IOU_THRESHOLD = 0.7  # set for M dataset, for zhejiang = 0.7

    solver = TTAFrame(DUNet)
    solver.load(model_dir + model_name + ".th")

    for image_name in image_list:
        mask = solver.test_one_img_from_path(image_dir + image_name)

        # mask[mask <= 4] = 0
        # mask[mask > 4] = 255
        gray_mask = (mask / 8 * 255).astype(np.uint8)
        th, mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_OTSU)
        if th < 30:
            mask[gray_mask < 128] = 0
            mask[gray_mask >= 128] = 255

        gt_name = image_name[:-7] + 'lab.png'
        gt = cv2.imread(gt_dir + gt_name, 0)
        dilated_kernel = np.ones((3, 3))
        dilated_gt = cv2.dilate(gt, dilated_kernel)
        # cv2.imwrite("test.png",dilated_gt)
        gt = gt / 255
        mask = mask / 255
        dilated_gt = dilated_gt / 255
        # intersection = np.sum(mask * gt)
        intersection = np.sum(mask * dilated_gt)
        sum_mask = np.sum(mask)
        sum_gt = np.sum(gt)
        if sum_gt < 1000:
            iou = 1
        else:
            # iou = (intersection) / (sum_mask + sum_gt - intersection + 0.00001)
            iou = (intersection) / (sum_mask + sum_gt - np.sum(mask * gt) + 0.00001)
        if iou > 1:
            iou = 1
        # print(image_name+": "+str(iou))
        iou_dict[image_name] = iou
        # print(iou)
        err = 1 - iou
        total_error += err * image_prob[image_name]

    print("************************")
    print("total error rate:" + str(total_error))
    if total_error > 0.5:
        print("bad classifier")
    error_list.append(total_error)
    prob_update_weight = total_error / (1 - total_error)

    # update weight
    for image_name in image_list:
        if iou_dict[image_name] >= IOU_THRESHOLD:
            image_prob[image_name] = image_prob[image_name] * prob_update_weight

    # normalize probability
    sum_prob = 0
    for key in image_prob.keys():
        sum_prob += image_prob[key]

    for key in image_prob.keys():
        image_prob[key] = image_prob[key] / sum_prob

    # randomly choosing tiny_dataset; different image numbers between different classifiers
    for j in range(10):
        picked_image_list = []  # 定义在这里是想在产生随机数不好的情况下进行循环再产生，然后重新挑选
        # rand 0~1 divided by dataset_size
        rand_prob = np.random.rand(len(image_prob))
        sum_rand_prob = np.sum(rand_prob)
        rand_prob = rand_prob / sum_rand_prob
        i = 0
        print("Random number is:" + str(rand_prob[i]))

        for image_name in image_prob.keys():
            if image_prob[image_name] > rand_prob[i]:
                picked_image_list.append(image_name)
            i+=1
            # else:
            # print("not choose" + image_name)
        if len(image_prob)*4//5 > len(picked_image_list) > len(image_prob)//5:
            print("picked total images:{}".format(len(picked_image_list)))
            break
        else:
            print("picked {} images and re-pick".format(len(picked_image_list)))
            continue
    picked_image_array=np.array(picked_image_list)

    # new_image_name_list = []
    # total_choose_number=dataset_size
    # for name in image_list:
    #     for num in range(int(total_choose_number*image_prob[name]+0.5)):
    #         new_image_name_list.append(name)
    # random.shuffle(new_image_name_list)
    # if len(new_image_name_list)>0:
    #     picked_image_list_inx=np.random.randint(0,len(new_image_name_list),[dataset_size])
    #     picked_image_array=np.array(new_image_name_list)[picked_image_list_inx]
    #     print("picked total images:{}".format(len(picked_image_list_inx)))
    # else:
    #     print("no train image left for next training")
    #     print("stop early !!!")
    #     picked_image_array=np.array([])


    with open(image_name_file_save_dir + "train_image_file_" + str(train_id + 1) + ".txt", "w") as f:
        for name in picked_image_array:
            f.write(name + "\n")
    print("total img for classifier #" + str(train_id + 1) + ": " + str(picked_image_array.shape[0]))

    # weight = np.log(1 / total_error) # zk edited the wrong one
    weight = np.log(1 / prob_update_weight)  # wcc edited the right one
    print("classifier #" + str(train_id) + " 's weight:" + str(weight))

    with open(image_prob_file_save_dir + "image_prob_" + str(train_id + 1) + ".txt", "w") as f:
        f.write("classifier #" + str(train_id) + " 's weight:" + str(weight) + "\n")
        for image_name in image_prob.keys():
            f.write(image_name + "," + str(image_prob[image_name]) + "\n")


if __name__ == "__main__":
    model_path = '../weights/multi_tiny_weights/dlinknet_dupsample_5.th'
    image_dir = "E:/shao_xing/tiny_dataset/D1/original/sat/"
    gt_dir = "E:/shao_xing/tiny_dataset/D1/original/lab/"
    image_prob_file_save_dir = "E:/shao_xing/tiny_dataset/"
    image_name_file_save_dir = "E:/shao_xing/tiny_dataset/"
    train_id = 1
    model_dir = 'E:/shao_xing/projects/dlinknet_dupsampling/weights/'
    model_name = 'dlinknet_dupsample_' + str(train_id)

    train_paras = {"model_dir": model_dir,
                   "model_name": model_name,
                   "image_dir": image_dir,
                   "gt_dir": gt_dir,
                   "train_id": train_id,
                   "image_prob_file_save_dir": image_prob_file_save_dir,
                   "image_name_file_save_dir": image_name_file_save_dir}
    generate_boosting_data(train_paras)
    print("Merci! Thomas et Sophie~")
