import sys
sys.path.append("./discoverlib")
from evaluation.discoverlib import geom, graph
import numpy as np
import cv2

"""
evaluate connectivity based on ground truth graph
We split the total graph into segments with length of around 20 pixels
Then, statistic the number of fully connected segments in segmentation masks.
The connectivity ratio is the percentage of fully connected segments. 
"""

log_name = "withoutDlink" #!!!!!!!!
mask_dir = '/home/ck/pyprojects/data/massa/result_boost/withoutDlink/'#!!!!!!!

total_gt_number=0
total_connected_number=0
total_not_connected_number=0

total_gt_length=0
mylog = open('/home/ck/pyprojects/data/massa/result_boost/eval_log/' + log_name + '_connect.log', 'w')
region_name_list = ['c', 'g', 'k', 'o']
for region in region_name_list:
    print("test region: "+region)
    graph_name = '/media/ck/新加卷/TGRS/data/Massachusetts/graph/'+region+".graph" #!!!!!!!gt_graph
    gt_graph = graph.read_graph(graph_name)
    edge_nodes=[]
    for i,edge in enumerate(gt_graph.edges):
        if i % 2 ==0:
            edge_nodes.append([edge.src.point.x,edge.src.point.y,edge.dst.point.x,edge.dst.point.y])
    base_gt_mask=np.zeros((1024, 1024))
    edge_nodes=np.array(edge_nodes)

    for i in range(-2, 2):
        for j in range(-2, 2):
            mask_file = region + '_' + str(i) + '_' + str(j) + '_boost.png'
            mask = cv2.imread(mask_dir + mask_file, 0)/255
            patch_gt_number=0
            patch_connected_number=0
            patch_not_connected_number=0
            offset=[-i*1024, -j*1024, -i*1024, -j*1024]
            patch_nodes=edge_nodes+offset
            for seg_edge in patch_nodes:
                if (seg_edge>=[0,0,0,0]).all() and (seg_edge<[1024,1024,1024,1024]).all():
                    base_gt_mask = np.zeros((1024, 1024))
                    patch_gt_number+=1
                    base_gt_mask=cv2.line(base_gt_mask,(seg_edge[0],seg_edge[1]),(seg_edge[2],seg_edge[3]),color=1, thickness=1)
                    pred_seg_length=np.sum(mask[base_gt_mask>0])
                    gt_length=np.sum(base_gt_mask>0)
                    total_gt_length+=gt_length
                    if pred_seg_length < gt_length:
                        patch_not_connected_number+=1
                    else:
                        patch_connected_number+=1
                else:
                    pass

            ratio=patch_connected_number/(patch_gt_number+0.0001)
            # print('test image ', str(i), '_', str(j), 'ratio:', round(ratio, 2), file=mylogs)
            print('test image {}_{} connected:not:total {}/{}/{}, ratio: {}'.format(i,j,patch_connected_number,
                                                                                        patch_not_connected_number,
                                                                                        patch_gt_number,
                                                                                        round(ratio, 2)))
            print('test image {}_{} connected:not:total {}/{}/{}, ratio: {}'.format(i, j, patch_connected_number,
                                                                                    patch_not_connected_number,
                                                                                    patch_gt_number,
                                                                                    round(ratio, 2)),file=mylog)
            total_gt_number+=patch_gt_number
            total_connected_number+=patch_connected_number
            total_not_connected_number+=patch_not_connected_number

print('********************************')
print("total connected:not:total {}/{}/{}, ratio: {}".format(total_connected_number,
                                                             total_not_connected_number,
                                                             total_gt_number,
                                                             round(total_connected_number/total_gt_number,2)))
print("total_gt_length:{}".format(total_gt_length))
print("average gt length:{}".format(total_gt_length/total_gt_number))
print('********************************', file=mylog)
print("total connected:not:total {}/{}/{}, ratio: {}".format(total_connected_number,
                                                             total_not_connected_number,
                                                             total_gt_number,
                                                             round(total_connected_number/total_gt_number,2)),
      file=mylog)
print("total_gt_length:{}".format(total_gt_length),file=mylog)
print("average gt length:{}".format(total_gt_length/total_gt_number),file=mylog)

mylog.close()
