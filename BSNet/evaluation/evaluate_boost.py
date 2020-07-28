import numpy as np
import os
import cv2
import collections
from PIL import Image

class result:

    def __init__(self, prediction, truevalue):
        # we read in a numpy ndarray and need to flatten it to some sort of a list (not exactly)
        self.Yhat = prediction.flatten('C')
        self.Y = truevalue.flatten('C')
        self.confusionlist = []
        self.countdict = dict
        self.resultmatrix = np.empty(shape=prediction.shape)
        self.recall = 0.00
        self.precision = 0.00
        self.f1 = 0.00
        self.accuracy = 0.00
        self.iou = 0.00
        self.summary = str

    def calculate_measures(self):

        #### based on confusion list
        # initialize empty list for storage of TP/FP/FN/TN
        confusion_list = []

        # loop over both flattened arrays using zip
        for i, j in zip(self.Yhat, self.Y):
            if i == 1 and j == 1:
                confusion_list.append('TP')
            elif i == 0 and j == 0:
                confusion_list.append('TN')
            elif i == 0 and j == 1:
                confusion_list.append('FN')
            elif i == 1 and j == 0:
                confusion_list.append('FP')
        self.confusionlist = confusion_list

        ## calculate draw matrix
        # get the TP/FN/FP/TN Matrix in shape of the input image
        self.resultmatrix = np.asarray(self.confusionlist).reshape(self.resultmatrix.shape)

        ## calculate Measures
        # get counts dictionary
        self.countdict = collections.Counter(self.confusionlist)

        # RECALL
        if (self.countdict['TP'] + self.countdict['TN']) > 0:
            self.recall = self.countdict['TP'] / (self.countdict['TP'] + self.countdict['FN'])

        # PRECISION
        if (self.countdict['TP'] + self.countdict['FP']) > 0:
            self.precision = self.countdict['TP'] / (self.countdict['TP'] + self.countdict['FP'])

        # F1 = harmonic mean precision and recall
        # control for 0 division
        if (self.precision + self.recall) > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        # ACCURACY
        self.accuracy = (self.countdict['TP'] + self.countdict['TN']) / len(confusion_list)

        # IoU
        if (self.countdict['TP'] + self.countdict['FP'] + self.countdict['FN']) > 0:
            self.iou = self.countdict['TP'] / (self.countdict['TP'] + self.countdict['FP'] + self.countdict['FN'])

    def printmeasures(self, mylogs):
        print('Recall: ', round(self.recall, 2), ',Precision: ', round(self.precision, 2),
              ',F1-Score: ', round(self.f1, 2), ',Accuracy: ', round(self.accuracy, 2),
              ',IoU: ', round(self.iou, 2), file=mylogs)
        print('Recall:', round(self.recall, 2), ',Precision:', round(self.precision, 2),
              ',F1-Score:', round(self.f1, 2), ',Accuracy:', round(self.accuracy, 2),
              ',IoU:', round(self.iou, 2))

def visualize_prediction(resultmatrix, image):
    # flag image as writeable
    image.flags.writeable = True

    for i in range(resultmatrix.shape[0]):
        for j in range(resultmatrix.shape[0]):
            measure = resultmatrix[i, j]
            # check for all 4 cases and draw coloured point accordingly
            if measure == 'TP':
                # green
                image[i, j, 0] = 0
                image[i, j, 1] = 255
                image[i, j, 2] = 0
            elif measure == 'FN':
                # blue
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 255
            elif measure == 'FP':
                # red
                image[i, j, 0] = 255
                image[i, j, 1] = 0
                image[i, j, 2] = 0
    return image

if __name__ == "__main__":
    img_root = 'E:/shao_xing/tiny_dataset/D1/original/sat/'
    lab_root = 'E:/shao_xing/tiny_dataset/D1/original/tiny_sat_lab/'
    mask_root = 'E:/shao_xing/tiny_dataset/segment2_dupsample/'

    region_name = 'd'
    # vis_root = 'visual/dlinknet/' + region_name + '/'
    # os.makedirs(vis_root)

    # initialize lists that store the performance measure values for all predicted images
    recall_list = []
    precision_list = []
    f1_list = []
    accuracy_list = []
    iou_list = []

    mylogs = open('E:/shao_xing/tiny_dataset/w_evalog/' + region_name + '_dupsample_2.log','w')

    mask_name_list=os.listdir(mask_root)

    for mask_name in mask_name_list:
        img_name = mask_name[:-8] + 'sat.png'
        lab_name = mask_name[:-8] + 'osm.png'
        img = img_root + img_name
        lab = lab_root + lab_name
        mask = mask_root + mask_name
        if os.path.exists(img):
            if os.path.exists(lab):
                if os.path.exists(mask):
                    image = cv2.imread(img)
                    prediction = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                    truevalue = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)
                    if np.sum(truevalue) < 10:
                        continue
                    image = np.ndarray.astype(image, dtype='uint8')
                    prediction = np.array(prediction, np.float32) / 255.0
                    truevalue = np.array(truevalue, np.float32) / 255.0
                    truevalue[truevalue >= 0.5] = 1
                    truevalue[truevalue <= 0.5] = 0
                    prediction[prediction >= 0.5] = 1
                    prediction[prediction <= 0.5] = 0

                    # initialize result class that calculates and stores all evaluation measures
                    res = result(prediction=prediction, truevalue=truevalue)
                    res.calculate_measures()
                    print('test image ' + img_name, file=mylogs)
                    print('test image ' + img_name)
                    res.printmeasures(mylogs)

                    # append to evaluation lists
                    recall_list.append(res.recall)
                    precision_list.append(res.precision)
                    f1_list.append(res.f1)
                    accuracy_list.append(res.accuracy)
                    iou_list.append(res.iou)

                    # vis = visualize_prediction(resultmatrix=res.resultmatrix, image=image)
                    # cv2.imwrite(vis_root + region_name + '_' + str(i) + '_' + str(j) + '_vis.png', vis)
                    # visual = Image.fromarray(vis, mode='RGB')
                    # visual.save(vis_root + region_name + '_' + str(i) + '_' + str(j) + '_vis.png', 'PNG')
                    # PIL和CV2的通道顺序不一样

    # print the results for the evaluation measures to the command line
    print('********************************', file=mylogs)
    print('Recall:', round((sum(recall_list) / len(recall_list)), 2), file=mylogs)
    print('Precision:', round((sum(precision_list) / len(precision_list)), 2), file=mylogs)
    print('F1-Score:', round((sum(f1_list) / len(f1_list)), 2), file=mylogs)
    print('Accuracy:',  round((sum(accuracy_list) / len(accuracy_list)), 2), file=mylogs)
    print('IoU:', round((sum(iou_list) / len(iou_list)), 2), file=mylogs)
    print('********************************')
    print('Recall:', round((sum(recall_list) / len(recall_list)), 2))
    print('Precision:', round((sum(precision_list) / len(precision_list)), 2))
    print('F1-Score:', round((sum(f1_list) / len(f1_list)), 2))
    print('Accuracy:', round((sum(accuracy_list) / len(accuracy_list)), 2))
    print('IoU:', round((sum(iou_list) / len(iou_list)), 2))

    mylogs.close()