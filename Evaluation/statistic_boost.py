import cv2
import numpy as np
import os


def merge_boost_result(initial_mask, T, boost_mask, voting, withInitial = True):
    pro_boost_mask = [" "]
    all_weight_boost = 0
    all_voting = 0
    for i in range(1, T + 1):
        pro_boost_mask.append(boost_mask[i] / 255)
        all_weight_boost += voting[i] * pro_boost_mask[i]
        all_voting += voting[i]
    ave_weight_boost = all_weight_boost / all_voting
    result=ave_weight_boost
    pro_initial_mask = initial_mask / 255
    if withInitial:
        result = pro_initial_mask

    result[ave_weight_boost>0.8]=ave_weight_boost[ave_weight_boost>0.8]
    
    pro_boost = result * 255
    th, bina_boost = cv2.threshold(pro_boost.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    if th < 80:
        bina_boost[pro_boost < 128] = 0
        bina_boost[pro_boost >= 128] = 255
    # pro_boost=result*255
    # bina_boost=result
    # bina_boost[bina_boost>0.5]=255
    # bina_boost[bina_boost<=0.5]=0
    return pro_boost, bina_boost


if __name__ == "__main__":
    T = 4
    initial_mask_dir = "./data/deepglobe/out/result_seg_dlink_grey/"
    binary_mask_dir="./data/deepglobe/out/result_seg_dlink/"
    boost_mask_dir = [" "]
    for i in range(1, T+1):
        boost_mask_dir.append("./data/deepglobe/out/boosting/boost"+str(i)+"/")

    save_dir = "./data/deepglobe/out/result_boost4/"
    if os.path.isdir(save_dir + 'withDlink/'):
        pass
    else:
        os.makedirs(save_dir + 'withDlink/')
        os.makedirs(save_dir + 'witDlink_grey/')

    # region_name_list = [["c",-2,-2,2,2],["g",-2,-2,2,2],[s"k",-2,-2,2,2],["o",-2,-2,2,2]]
    # region_name_list = [["c2",-4,-4,4,4], ["d",-4,-4,4,4]]
    # region_name_list = [["amsterdam",-4,-4,4,4], ["chicago",-4,-4,4,4], ["denver",-4,-4,4,4], ["la",-4,-4,4,4], ["montreal",-4,-4,4,4],
    #                     ["paris",-4,-4,4,4], ["pittsburgh",-4,-4,4,4], ["saltlakecity",-4,-4,4,4], ["san diego",-4,-4,4,4],
    #                     ["tokyo",-4,-4,4,4], ["toronto",-4,-4,4,4], ["vancouver",-4,-4,4,4]]
    region_name_list=os.listdir(binary_mask_dir)
    for region_name in region_name_list:
        region_id=region_name[:-9]
        initial_mask_file = region_id + '_mask.png'
        boost_mask_file = [" "]
        for i in range(1, T+1):
            boost_mask_file.append(initial_mask_file)

        initial_mask = cv2.imread(initial_mask_dir + initial_mask_file, cv2.IMREAD_GRAYSCALE)
        boost_mask = [" "]
        for i in range(1, T+1):
            # print(boost_mask_dir[i] + boost_mask_file[i])
            boost_mask.append(cv2.imread(boost_mask_dir[i] + boost_mask_file[i], cv2.IMREAD_GRAYSCALE))
        # voting = [1, 0.8568, 0.4686] # boosting id starts from 1 !!!
        # voting = [1, 0.7419, 1.4687, 1.0127, 1.0500, 0.8032] # zj
        voting = [1, 0.8813,1.3198,0.8428,0.6102,1.0422]
        pro_boost, bina_boost = merge_boost_result(initial_mask, T, boost_mask, voting, withInitial=True)

        save_file = initial_mask_file[:-8] + 'boost.png'
        cv2.imwrite(save_dir + 'withDlink_grey/' + save_file, pro_boost.astype(np.uint8))
        cv2.imwrite(save_dir + 'withDlink/' + save_file, bina_boost.astype(np.uint8))
