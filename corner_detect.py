import cv2
import numpy as np
import scipy.ndimage
import skimage.morphology

def good_feature_to_track(thin_mask, mask, out_name, save_path):
    """
     Apply the detector on the segmentation map to detect the road junctions as starting points for tracing.
    :param thin_mask: one-pixel width segmentation map
    :param mask: road segmentation map
    :param out_name: filename
    :param save_path: the directory of corner detection results
    :return:
    """
    # set a padding to avoid image edge corners
    padding_x = 128+5
    padding_y = 128

    corners = cv2.goodFeaturesToTrack(thin_mask, 100, 0.1, 500)
    corners = np.int0(corners)
    img = np.zeros((mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = mask
    img[:, :, 1] = mask
    img[:, :, 2] = mask
    corner_num = 0
    with open(save_path+out_name[:-4]+".txt", "w") as f:
        for i in corners:
            x, y = i.ravel()
            if x < padding_x or x > img.shape[0]-padding_x:
                continue
            if y < padding_y or y > img.shape[1]-padding_y:
                continue

            f.write("{},{}\n".format(x,y))
            cv2.circle(img, (x, y), 20, (0, 0, 255), -1)
            corner_num += 1
    print("total corners number:{}".format(corner_num))
    cv2.imwrite(save_path+out_name[:-4]+'_with_corners.png', img)


def thin_image(mask_dir, filename):
    """
    Skeletonize the road segmentation map to a one-pixel width
    :param mask_dir: the directory of road segmentation map
    :param filename: the filename of road segmentation map
    :return: one-pixel width segmentation map
    """
    im = scipy.ndimage.imread(mask_dir + filename)
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


if __name__ == "__main__":
    mask_dir = "/out/corner_detect/seg_mask/" # the directory of segmentation map
    txt_dir = "/out/corner_detect/corners/" # the directory of corner detection results
    region_list = ["amsterdam", "chicago", "denver", "la", "montreal", "paris", "pittsburgh", "saltlakecity", "san diego", "tokyo", "toronto", "vancouver"]
    for region in region_list:
        print("region "+region)
        mask_filename = region + '_seg.png'
        thin_filename = region + '_thin.png'

        thin_img = thin_image(mask_dir, mask_filename)
        mask = cv2.imread(mask_dir + mask_filename, 0)

        good_feature_to_track(thin_img, mask, mask_filename, txt_dir)