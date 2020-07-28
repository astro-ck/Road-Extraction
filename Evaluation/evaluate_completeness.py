import numpy as np
import skimage
from skimage import morphology
import cv2

def thin_image(mask_dir, mask_file):
    im = cv2.imread(mask_dir + mask_file, 0)
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


mask_dir = 'E:/shao_xing/out/result_boost12345/'
gt_dir = 'E:/shao_xing/test/lab/'
region = 'd'

mylogs = open('E:/shao_xing/out/tiny_evalog/new_metric/' + region + '_boost12345.log','w')
ratio_list=[]
for i in range(-4, 4):
    for j in range(-4, 4):
        gt_file = region + '_' + str(i) + '_' + str(j) + '_osm.png'
        mask_file = gt_file[:-7] + 'merge.png'

        mask = cv2.imread(mask_dir + mask_file, 0)
        thin_gt = thin_image(gt_dir, gt_file)
        num_mask = np.sum(thin_gt[mask > 128]) / 255
        num_gt = np.sum(thin_gt) / 255
        ratio = num_mask / (num_gt+0.00001)
        if num_gt != 0:
            ratio_list.append(ratio)
        print('test image ', str(i), '_', str(j), 'ratio:', round(ratio, 2), file=mylogs)
        print('test image ', str(i), '_', str(j), 'ratio:', round(ratio, 2))

print('********************************', file=mylogs)
print('Average Ratio:', round((sum(ratio_list) / len(ratio_list)), 2), file=mylogs)
print('********************************')
print('Average Ratio:', round((sum(ratio_list) / len(ratio_list)), 2))
mylogs.close()
