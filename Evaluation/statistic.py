import cv2
import numpy as np
import os


def dilate_image(file_name):# 通过graph生成缓冲区，最后进行分割结果合并用的
    osm = cv2.imread(file_name, 0)
    kernels = np.ones((15, 15))
    dilated_img = cv2.dilate(osm, kernels)
    return dilated_img


def merge_result(tiny_mask_file, large_mask_file, dilated_osm):
    tiny_pred_mask = cv2.imread(tiny_mask_file, 0)
    large_pred_mask = cv2.imread(large_mask_file, 0)
    result = large_pred_mask
    result[dilated_osm > 128] = tiny_pred_mask[dilated_osm > 128] + large_pred_mask[dilated_osm > 128]
    result[result > 255] = 255
    return result


if __name__ == "__main__":
    large_mask_dir = "E:/shao_xing/out40/seg_mask_vancouver/"
    tiny_mask_dir = "E:/shao_xing/out40/tiny_1024mask/"
    dilate_osm_dir = "E:/shao_xing/out40/corners/graph_to_mask/" # convert centerline graph to mask here 1024×1024
    file_name_list = os.listdir(tiny_mask_dir) # xxxxx_1_1_infer.png
    for file_name in file_name_list:
        dilated = dilate_image(dilate_osm_dir + file_name[:-8] + "osm.png")
        tiny_mask_file = file_name
        large_mask_file = file_name[:-8] + "mask.png"
        result = merge_result(tiny_mask_dir + tiny_mask_file, large_mask_dir + large_mask_file, dilated)

        save_name = "E:/shao_xing/out40/result_buffer/" + file_name[:-8] + "merge.png"
        print(save_name)
        cv2.imwrite(save_name, result.astype(np.uint8))
