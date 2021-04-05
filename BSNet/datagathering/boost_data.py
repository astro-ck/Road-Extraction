import cv2
import os
import numpy as np

def cal_iou(mask, osm):
    i = np.sum(mask)
    j = np.sum(osm)
    intersection = np.sum(mask*osm)
    smooth = 0.0000001
    iou = (2*intersection+smooth)/(i+j+smooth)
    return iou

previous_sat_dir = 'E:/shao_xing/tiny_dataset/D1/original/tiny_sat_lab/'
previous_gt_dir = 'E:/shao_xing/tiny_dataset/D1/original/tiny_sat_lab/'
previous_mask_dir = 'E:/shao_xing/tiny_dataset/segment1_dupsample/'

sat_dir = 'E:/shao_xing/tiny_dataset/D2_dupsample/'
gt_dir = sat_dir

th = 0.2

for sat_name in os.listdir(previous_sat_dir): # 读上一个网络训练集sat或lab里的文件，因为mask是在全部D1上infer的 ！！！
    print("processing " + sat_name)
    gt_name = sat_name[:-7] + "lab.png"
    mask_name = sat_name[:-7] + "mask.png"

    sat_img = cv2.imread(previous_sat_dir + sat_name)
    gt = cv2.imread(previous_gt_dir + gt_name, 0)
    gt_img = gt / 255
    mask = cv2.imread(previous_mask_dir + mask_name, 0)
    mask_img = mask / 255

    iou = cal_iou(mask_img, gt_img)
    weights_retrain = 1 - iou
    if weights_retrain > th:
        new_sat = sat_img
        new_gt = gt_img * 255

        cv2.imwrite(sat_dir + sat_name, new_sat)
        cv2.imwrite(gt_dir + sat_name[:-7] + "osm.png", new_gt)

