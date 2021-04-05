import sys

sys.path.append("./discoverlib")
from discoverlib import graph
import cv2
import numpy as np
import skimage
from skimage import morphology, measure
import os
import scipy
from scipy import ndimage

def read_graph(graph_path):
    node_list = []
    edge_list = []
    with open(graph_path, "r") as f:
        vertex_section = True  # read node firstly
        for line in f.readlines():
            parts = line.strip().split(" ")
            if vertex_section:
                if len(parts) == 2:
                    node_list.append([int(float(parts[0])), int(float(parts[1]))])
                else:
                    print("Blank Line")
                    vertex_section = False
            elif len(parts) == 2:  # then read edge
                edge_list.append([int(parts[0]), int(parts[1])])
    return node_list, edge_list


def is_connected(node, mask):
    mask = mask / 255
    empty = np.zeros((1024, 1024))
    empty = cv2.line(empty, (node[0], node[1]), (node[2], node[3]), color=1, thickness=1)
    edge_length = np.sum(empty)
    pred_length = np.sum(mask[empty > 0])
    if pred_length < edge_length and pred_length > 0:
        return False, empty
    else:
        return True, empty


def get_sub_region(mask, x, y):
    window_size = 40
    window_radius = window_size // 2
    if x - window_radius < 0 or x + window_radius >= mask.shape[0] or y - window_radius < 0 or y + window_radius >= \
            mask.shape[1]:
        return np.zeros((1, 1))
    else:
        return mask[x - window_radius:x + window_radius, y - window_radius:y + window_radius]


def thin_image(mask_file):
    im = cv2.imread(mask_file, 0)
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


def fusion_massa_cities(patch_nodes, boost_mask_grey):

    th, boost_mask = cv2.threshold((boost_mask_grey*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    thresh = th / 255.0 + 1.0

    blank_mask = np.zeros((1024, 1024))
    for node in patch_nodes:
        if (node >= [0, 0, 0, 0]).all() and (node < [1024, 1024, 1024, 1024]).all():
            blank_mask = cv2.line(blank_mask, (node[0], node[1]), (node[2], node[3]), color=1, thickness=8)
    graph_mask = blank_mask

    seg_graph_prob = np.zeros((1024, 1024))
    seg_graph_prob[graph_mask > 0] = graph_mask[graph_mask > 0] + boost_mask_grey[graph_mask > 0]

    if thresh > 1:
        boost_mask_grey[seg_graph_prob > thresh] = (thresh / 2.0) * graph_mask[seg_graph_prob > thresh]
        for i in range(seg_graph_prob.shape[0]):
            for j in range(seg_graph_prob.shape[1]):
                if seg_graph_prob[i, j] > 1 and seg_graph_prob[i, j] <= thresh:
                    boost_mask_grey[i, j] += (thresh - 1) / 2.0
        boost_mask_grey[boost_mask_grey > (thresh / 2.0)] = thresh / 2.0
        result_mask_grey = boost_mask_grey * 255
        th1, result_mask = cv2.threshold(result_mask_grey.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    else:
        result_mask = boost_mask

    # post-processing
    fusion_mask = result_mask
    labels = measure.label(result_mask, connectivity=2)
    props = measure.regionprops(labels)
    for i in range(len(props)):
        if props[i].area < 100:
            corrdinates = props[i].coords
            fusion_mask[corrdinates[:, 0], corrdinates[:, 1]] = 0

    return fusion_mask


def fusion_zj(patch_nodes, boost_mask_grey):

    th, boost_mask = cv2.threshold((boost_mask_grey*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    thresh = th / 255.0 + 1.0

    blank_mask = np.zeros((1024, 1024))
    for node in patch_nodes:
        if (node >= [0, 0, 0, 0]).all() and (node < [1024, 1024, 1024, 1024]).all():

            isLinked, edge_mask = is_connected(node, boost_mask)
            if not isLinked:
                kernels = np.ones((11, 11))
                dilated_edge_mask = cv2.dilate(edge_mask, kernels)  # buffer

                area = np.sum(boost_mask[dilated_edge_mask > 0])
                length = np.sum(edge_mask * boost_mask)
                if area > 0 and length > 0:
                    adap_width = area / (length + 10)
                    if adap_width > 40:
                        adap_width = 40
                    if adap_width < 1:
                        adap_width = 1
                    blank_mask = cv2.line(blank_mask, (node[0], node[1]), (node[2], node[3]), color=1, thickness=int(adap_width))
            # else:
            #     blank_mask = cv2.line(blank_mask, (node[0], node[1]), (node[2], node[3]), color=1, thickness=8)

            # blank_mask = cv2.line(blank_mask, (node[0], node[1]), (node[2], node[3]), color=1, thickness=8)
    graph_mask = blank_mask

    seg_graph_prob = np.zeros((1024, 1024))
    seg_graph_prob[graph_mask > 0] = graph_mask[graph_mask > 0] + boost_mask_grey[graph_mask > 0]

    if thresh > 1:
        boost_mask_grey[seg_graph_prob > thresh] = (thresh / 2.0) * graph_mask[seg_graph_prob > thresh]
        for i in range(seg_graph_prob.shape[0]):
            for j in range(seg_graph_prob.shape[1]):
                if seg_graph_prob[i, j] > 1 and seg_graph_prob[i, j] <= thresh:
                    boost_mask_grey[i, j] += (thresh - 1) / 2.2
        boost_mask_grey[boost_mask_grey > (thresh / 2.0)] = thresh / 2.0
        result_mask_grey = boost_mask_grey * 255
        th1, result_mask = cv2.threshold(result_mask_grey.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    else:
        result_mask = boost_mask

    # post-processing
    fusion_mask = result_mask
    labels = measure.label(result_mask, connectivity=2)
    props = measure.regionprops(labels)
    for i in range(len(props)):
        if props[i].area < 200:
            corrdinates = props[i].coords
            fusion_mask[corrdinates[:, 0], corrdinates[:, 1]] = 0

    # result_mask_fill = ndimage.binary_fill_holes(result_mask, structure=np.ones((5, 5)).astype(result_mask.dtype)*255)
    # fusion_mask = np.zeros((1024, 1024)).astype(result_mask.dtype)
    # fusion_mask[result_mask_fill == True] = 255

    return fusion_mask


if __name__ == "__main__":
    # boost_mask_grey_dir = "E:/TGRS/data/massa/out/result_boost3/withDlink_grey/"
    # graph_dir = "E:/TGRS/data/massa/out/graph_infer/roadtracer-M/"

    boost_mask_grey_dir = "E:/TGRS/data/zj/out/result_boost2/withDlink_grey/"
    graph_dir = "E:/TGRS/data/zj/out/graph_infer/roadtracer-M/"

    # boost_mask_grey_dir = "E:/TGRS/data/cities/out/result_boost2/withDlink_grey/"
    # graph_dir = "E:/TGRS/data/cities/out/graph_infer/roadtracer-M/"

    save_dir = "E:/TGRS/data/zj/out/result_fusion1/"
    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir)

    # region_list = ['c', 'g', 'k', 'o']
    region_list = ['c2', 'd']
    # region_list = ["amsterdam", "chicago", "denver", "la", "montreal", "paris", "pittsburgh", "saltlakecity", "san diego", "tokyo", "toronto", "vancouver"]

    for region_name in region_list:
        graph_infer = graph.read_graph(graph_dir + region_name + '.out.graph')
        edge_nodes = []
        for i, edge in enumerate(graph_infer.edges):
            if i % 2 == 0:
                edge_nodes.append([edge.src.point.x, edge.src.point.y, edge.dst.point.x, edge.dst.point.y])
        edge_nodes = np.array(edge_nodes)

        for x in range(-4, 4):
            for y in range(-4, 4):
                boost_mask_grey_file = region_name + '_' + str(x) + '_' + str(y)+'_boost.png'
                boost_mask_grey = cv2.imread(boost_mask_grey_dir+boost_mask_grey_file, 0)/255

                offset = [-x * 1024, -y * 1024, -x * 1024, -y * 1024]
                patch_nodes = edge_nodes + offset

                # result = fusion_massa_cities(patch_nodes, boost_mask_grey)
                result = fusion_zj(patch_nodes, boost_mask_grey)

                save_file = region_name + '_' + str(x) + '_' + str(y) + "_fusion.png"
                cv2.imwrite(save_dir + save_file, result.astype(np.uint8))
