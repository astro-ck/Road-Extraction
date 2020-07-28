import numpy as np
import cv2
import sys
sys.path.append('/networks')
from networks.dinknet import DUNet
import time
from test import TTAFrame
import torch
from torch.autograd import Variable as V

# test 1024 image
def test_image(solver,img):
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

    maska = solver.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
    maskb = solver.net.forward(img6).squeeze().cpu().data.numpy()

    mask1 = maska + maskb[:,:,::-1]
    mask2 = mask1[:2] + mask1[2:,::-1]
    mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
    return mask3

# read nodes in the graph only (no edge)
def read_graph(graph_path):
    node_list = []
    with open(graph_path, "r") as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            if len(temp) == 2:
                node_list.append([int(temp[0]), int(temp[1])])
            else:
                print("a white line, jump out")
                break

    return node_list

def is_linked(x,y,large_mask):
    if np.sum(large_mask[x-5:x+5,y-5:y+5])>128:
        return True
    else:
        return False


def get_seg_region(large_image, x, y):
    window_size=256
    window_radius=window_size//2
    if x-window_radius<0 or x+window_radius>=large_image.shape[0] or y-window_radius<0 or y+window_radius>=large_image.shape[1]:
        return np.zeros((1,1))
    else:
        return large_image[x-window_radius:x+window_radius,y-window_radius:y+window_radius]


solver = TTAFrame(DUNet)
solver.load('~/pyprojects/RoadNet/boost_train/weights/massa_roadnet_2.th')
region_name_list = [["c",-2,-2,2,2],["g",-2,-2,2,2],["k",-2,-2,2,2],["o",-2,-2,2,2]]
for region_info in region_name_list:
    print("test region: "+region_info[0])
    graph_name = '~/data/massa/out/graph_infer/roadtracer-M/'+ region_info[0] + ".out.graph" 
    graph_nodes = read_graph(graph_name)
    for x in range(region_info[1], region_info[3]):
        for y in range(region_info[2], region_info[4]):
            city_name = region_info[0]+"_%s_%s" % (x, y)
            
            large_mask = cv2.imread("~/data/massa/out/result_seg_dlink/" + city_name + "_mask.png", 0)
            large_sat = cv2.imread("~/data/Massachusetts/test/sat/" + city_name + "_sat.png")
            tic = time.time()

            # graph_nodes is for whole city, here the image size 1024×1024 is what we need, so we make zero point from the left-top of 0_0 image to the left-top of each image
            nodes = np.array(graph_nodes)
            nodes[:, 1] += -y * 1024
            nodes[:, 0] += -x * 1024

            empty_mask = np.zeros(large_mask.shape)
            empty_sat = np.zeros(large_sat.shape)

            for node in nodes:
                if not is_linked(node[0], node[1], large_mask):
                    seg_region = get_seg_region(large_sat, node[0], node[1])
                    if seg_region.shape[0] > 10:
                        tiny_mask = test_image(solver, seg_region)
                        # tiny_mask[tiny_mask <= 4.0] = 0
                        # tiny_mask[tiny_mask > 4.0] = 255
                        maximum = tiny_mask.max()
                        minimum = tiny_mask.min()
                        tiny_mask = (tiny_mask - minimum) / 8
                        # empty_mask[node[0] - 128:node[0] + 128, node[1] - 128:node[1] + 128] += tiny_mask
                        empty_mask[node[0] - 128:node[0] + 128, node[1] - 128:node[1] + 128] = np.max((tiny_mask,empty_mask[node[0] - 128:node[0] + 128, node[1] - 128:node[1] + 128]),axis=0,keepdims=True)[0]

            # empty_mask[empty_mask > 255] = 255

            empty_mask[empty_mask > 1] = 1
            empty_mask *= 255

            print("done")
            save_name = "~/data/massa/out/pick_test/test2/%s_infer.png" % (city_name)
            print(save_name)
            cv2.imwrite(save_name, empty_mask)