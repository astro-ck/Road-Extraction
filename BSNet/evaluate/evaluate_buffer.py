import skimage
from skimage import morphology
import cv2
import numpy as np

def thin_image(mask_dir, mask_file):
    im = cv2.imread(mask_dir + mask_file, 0)
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


def dilated_buffer(mask_dir, mask_file):
    mask = cv2.imread(mask_dir + mask_file, 0)
    kernels = np.ones((5, 5))
    dilated_mask = cv2.dilate(mask, kernels)
    return dilated_mask


mask_dir = 'E:/shao_xing/out/result_ostu/'
gt_dir = 'E:/shao_xing/test/lab/'

mask_file = 'd_-2_2_merge.png'
gt_file = mask_file[:-9] + 'osm.png'

thin_mask = thin_image(mask_dir, mask_file)
thin_gt = thin_image(gt_dir, gt_file)
gt_buffer = dilated_buffer(gt_dir, gt_file)
cv2.imwrite("thin_image.png",thin_mask)
cv2.imwrite("gt_image.png",thin_gt)
num_mask = np.sum(thin_mask[gt_buffer > 128]) / 255
num_gt = np.sum(thin_gt) / 255
ratio = num_mask / num_gt
print(ratio)



