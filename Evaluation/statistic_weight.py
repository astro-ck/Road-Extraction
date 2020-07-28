import cv2
import numpy as np
import os


def merge_result(large_mask, tiny_mask, graph):
    pro_large_mask = large_mask / 255
    pro_tiny_mask = tiny_mask / 255
    gauss_graph = cv2.GaussianBlur(graph, (31, 31), 0)
    pro_graph = gauss_graph / 255

    pro_all = 0.8 * pro_large_mask + 0.2 * pro_tiny_mask
    #参数是自己设计的？我测试了0.6和0.4，那样小分割带来的噪声有点多 然后这里改成0.8和0.2之后，就把下一行
    # 改成只加小分割了 因为大分割整体占比已经够了 相当于还是在大分割基础上加小分割 用中心线约束 中心线在这里只是加强其中一部分的连接性 用proall是不是更能说的通
    # 可以试试啊 反正现在还不用调代码 注意输出路径 ok，那我输出到我的文件夹
    pro_all += pro_tiny_mask * pro_graph # 用pro all会不会更能讲得通？
    pro_all[pro_all > 1] = 1
    merged = (pro_all * 255).astype(np.uint8)
    th, result = cv2.threshold(merged, 0, 255, cv2.THRESH_OTSU)
    return result


if __name__ == "__main__":
    large_mask_dir = "E:/shao_xing/out/seg_mask_d_grey/"
    tiny_mask_dir = "E:/shao_xing/out/ck0328/tiny_gray_res34_r16_dice_dilate_234/"
    graph_dir = "E:/shao_xing/out/corners/graph_to_mask_new0228/"

    file_name_list = os.listdir(tiny_mask_dir)  # xxxxx_1_1_infer.png
    for file_name in file_name_list:
        large_mask_file = file_name[:-8] + "mask.png"
        tiny_mask_file = file_name
        graph_file = file_name[:-8] + "osm.png"

        large_mask = cv2.imread(large_mask_dir + large_mask_file, cv2.IMREAD_GRAYSCALE)
        tiny_mask = cv2.imread(tiny_mask_dir + tiny_mask_file, cv2.IMREAD_GRAYSCALE)
        graph = cv2.imread(graph_dir + graph_file, 0)

        result = merge_result(large_mask, tiny_mask, graph)

        save_name = "E:/shao_xing/out/ck0328/result_res34_r16_dice_dilate_234/" + file_name[:-8] + "merge.png"
        cv2.imwrite(save_name, result.astype(np.uint8))
