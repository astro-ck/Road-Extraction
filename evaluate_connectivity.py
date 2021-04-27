import sys
sys.path.append("./discoverlib")
from discoverlib import geom, graph
import numpy as np
import cv2
import skimage
from skimage import morphology


"""
evaluate connectivity based on ground truth graph
We split the total graph into segments with length of around 20 pixels
Then, statistic the number of fully connected segments in segmentation masks.
The connectivity ratio is the percentage of fully connected segments. 
"""

log_name = 'mask' # evaluation log file
mask_dir = '~/data/out/mask/' # segmentation masks for evaluating

total_gt_number = 0
total_connected_number = 0
total_not_connected_number = 0
total_pred_number = 0

total_connected_length = 0
total_gt_length = 0
total_pred_length = 0
mylog = open('~/data/out/eval_log/' + log_name + '_connect.log', 'w')

region_name_list = [["amsterdam",-4,-4,4,4], ["chicago",-4,-4,4,4], ["denver",-4,-4,4,4]]
for region_info in region_name_list:
    print("test region: "+region_info[0])
    graph_name = '~/data/graph_gt/'+ region_info[0] + ".graph" # ground truth graph

    gt_graph = graph.read_graph(graph_name)
    edge_nodes=[]
    for i,edge in enumerate(gt_graph.edges):
        if i % 2 ==0:
            edge_nodes.append([edge.src.point.x,edge.src.point.y,edge.dst.point.x,edge.dst.point.y])
    base_gt_mask=np.zeros((1024, 1024))
    edge_nodes=np.array(edge_nodes)

    for i in range(region_info[1], region_info[3]):
        for j in range(region_info[2], region_info[4]):
            mask_file = region_info[0] + '_' + str(i) + '_' + str(j) + '_fusion.png'
            # print(mask_dir + mask_file)
            mask = cv2.imread(mask_dir + mask_file, 0)/255

            patch_gt_number=0
            patch_connected_number=0
            patch_not_connected_number=0

            patch_connected_length = 0
            patch_gt_length = 0

            offset=[-i*1024, -j*1024, -i*1024, -j*1024]
            patch_nodes=edge_nodes+offset
            for seg_edge in patch_nodes:
                if (seg_edge>=[0,0,0,0]).all() and (seg_edge<[1024,1024,1024,1024]).all():
                    base_gt_mask = np.zeros((1024, 1024))
                    patch_gt_number+=1 # number of segments on the ground-truth graph
                    base_gt_mask=cv2.line(base_gt_mask,(seg_edge[0],seg_edge[1]),(seg_edge[2],seg_edge[3]),color=1, thickness=1)
                    pred_seg_length=np.sum(mask[base_gt_mask>0])
                    gt_length=np.sum(base_gt_mask>0)
                    patch_gt_length += gt_length
                    if pred_seg_length < gt_length:
                        patch_not_connected_number+=1
                    else:
                        patch_connected_number+=1
                        patch_connected_length += gt_length
                else:
                    pass

            im = (mask*255) > 128
            selem = skimage.morphology.disk(2)
            im = skimage.morphology.binary_dilation(im, selem)
            im = skimage.morphology.thin(im)
            thin_mask = im.astype(np.uint8) * 255

            patch_pred_length = np.sum(thin_mask > 0)

            patch_pred_number = patch_pred_length / 20.0 # number of segments on the prediction graph

            ratio = 2*patch_connected_length/(patch_gt_length+patch_pred_length+0.00001)

            print('test image {}_{} connected:not:total {}/{}/{}, ratio: {}'.format(i,j,patch_connected_number,
                                                                                        patch_not_connected_number,
                                                                                        patch_gt_number,
                                                                                        round(ratio, 4)))
            print('test image {}_{} connected:not:total {}/{}/{}, ratio: {}'.format(i, j, patch_connected_number,
                                                                                    patch_not_connected_number,
                                                                                    patch_gt_number,
                                                                                    round(ratio, 4)), file=mylog)
            total_gt_number += patch_gt_number
            total_connected_number += patch_connected_number
            total_not_connected_number += patch_not_connected_number
            total_pred_number += patch_pred_number

            total_connected_length += patch_connected_length
            total_gt_length += patch_gt_length
            total_pred_length += patch_pred_length

# total_ratio = 2*total_connected_number/(total_gt_number+total_pred_number)
total_ratio = 2*total_connected_length/(total_gt_length+total_pred_length)

print('********************************')
print("total connected:not:total {}/{}/{}, ratio: {}".format(total_connected_number,
                                                             total_not_connected_number,
                                                             total_gt_number,
                                                             round(total_ratio, 4)))
print("total_gt_length:{}".format(total_gt_length))
print("average gt length:{}".format(total_gt_length/total_gt_number))
print('********************************', file=mylog)
print("total connected:not:total {}/{}/{}, ratio: {}".format(total_connected_number,
                                                             total_not_connected_number,
                                                             total_gt_number,
                                                             round(total_ratio, 4)),
      file=mylog)
print("total_gt_length:{}".format(total_gt_length),file=mylog)
print("average gt length:{}".format(total_gt_length/total_gt_number),file=mylog)

mylog.close()

