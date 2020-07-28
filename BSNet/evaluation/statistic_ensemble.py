import cv2
import numpy as np
import os


def merge_result(large_mask, iou_1, tiny_mask_1, graph):
    pro_large_mask = large_mask / 255
    pro_tiny_mask_1 = tiny_mask_1 / 255
    # pro_tiny_mask_2 = tiny_mask_2 / 255
    # pro_tiny_mask_3 = tiny_mask_3 / 255
    # pro_tiny_mask_4 = tiny_mask_4 / 255
    # pro_tiny_mask_5 = tiny_mask_5 / 255
    kernels = np.ones((15, 15))
    dilated_graph = cv2.dilate(graph, kernels)
    gauss_graph = cv2.GaussianBlur(dilated_graph, (31, 31), 0)
    pro_graph = gauss_graph / 255

    pro_all = pro_large_mask
    pro_tiny = (iou_1 * pro_tiny_mask_1) / (iou_1)
    pro_all[pro_graph > 0.5] += pro_tiny[pro_graph > 0.5]
    pro_all[pro_tiny > 0.5] += pro_tiny[pro_tiny > 0.5]
    pro_all[pro_all > 1] = 1

    merged = (pro_all * 255).astype(np.uint8)
    th, result = cv2.threshold(merged, 0, 255, cv2.THRESH_OTSU)
    return result


if __name__ == "__main__":
    large_mask_dir = "E:/shao_xing/out/seg_mask_d_grey/"
    tiny_mask_dir_1 = "E:/shao_xing/out/tiny_1024grey_dupsample1/"
    tiny_mask_dir_2 = "E:/shao_xing/out/tiny_1024grey_dupsample2/"
    tiny_mask_dir_3 = "E:/shao_xing/out/tiny_1024grey_dupsample3/"
    tiny_mask_dir_4 = "E:/shao_xing/out/tiny_1024grey_dupsample4/"
    tiny_mask_dir_5 = "E:/shao_xing/out/tiny_1024grey_dupsample5/"
    graph_dir = "E:/shao_xing/out/corners/graph_to_mask_new0228/"

    file_name_list = os.listdir(large_mask_dir)  # xxxxx_1_1_mask.png
    for file_name in file_name_list:
        large_mask_file = file_name
        tiny_mask_file_1 = file_name[:-8] + 'mask.png'
        # tiny_mask_file_2 = file_name[:-8] + 'mask.png'
        # tiny_mask_file_3 = file_name[:-8] + 'mask.png'
        # tiny_mask_file_4 = file_name[:-8] + 'mask.png'
        # tiny_mask_file_5 = file_name[:-8] + 'mask.png'
        graph_file = file_name[:-8] + "osm.png"

        large_mask = cv2.imread(large_mask_dir + large_mask_file, cv2.IMREAD_GRAYSCALE)
        tiny_mask_1 = cv2.imread(tiny_mask_dir_1 + tiny_mask_file_1, cv2.IMREAD_GRAYSCALE)
        # tiny_mask_2 = cv2.imread(tiny_mask_dir_2 + tiny_mask_file_2, cv2.IMREAD_GRAYSCALE)
        # tiny_mask_3 = cv2.imread(tiny_mask_dir_3 + tiny_mask_file_3, cv2.IMREAD_GRAYSCALE)
        # tiny_mask_4 = cv2.imread(tiny_mask_dir_4 + tiny_mask_file_4, cv2.IMREAD_GRAYSCALE)
        # tiny_mask_5 = cv2.imread(tiny_mask_dir_5 + tiny_mask_file_5, cv2.IMREAD_GRAYSCALE)
        graph = cv2.imread(graph_dir + graph_file, 0)

        iou_1 = 0.75479
        iou_2 = 0.87956
        iou_3 = 0.81797
        iou_4 = 0.68272
        iou_5 = 0.72961
        result = merge_result(large_mask, iou_1, tiny_mask_1, graph)

        save_name = "E:/shao_xing/out/result_boost1/" + file_name[:-8] + "merge.png"
        cv2.imwrite(save_name, result.astype(np.uint8))
