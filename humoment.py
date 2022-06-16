import cv2
import argparse
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE
import seaborn as sns

def humoment(home, paths, isTrain):
    result = {}
    for label in paths:
        img = []
        moments = []
        huMoment = []
        avg_hu = [0,0,0,0,0,0,0]
        for name in paths[label]:
            # img += [0]
            # print(paths[label], label)
            try: 
                img += [cv2.threshold(np.sum(np.load(home + name), axis=2), 300, 1000, cv2.THRESH_BINARY)[1]]
            except:
                img += [cv2.threshold(np.load(home + name), 300, 1000, cv2.THRESH_BINARY)[1]]
            # Calculate Moments
            moments += [cv2.moments(img[-1])]
            h = cv2.HuMoments(moments[-1])
            for i in range(0,7):
                try:
                    h[i] = -1* math.copysign(1.0, h[i]) * math.log10(abs(h[i]))
                except:
                    pass
            #     if isTrain: avg_hu[i] += h[i]/float(len(paths[label]))
            # # Calculate Hu Moments
            # if not isTrain:
            if h[0] == h[1] == 0.0:
                print("error with", name)
            huMoment += [np.array(h)]

        # if isTrain:
        #     result[label] = np.array(avg_hu)
        # else:
        result[label] = huMoment
        
    return result

def parseData(path):
    data_parsed = {}
    with open(path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                data_parsed[row[1]] += [row[0]]
            except:
                if len(row) > 0:
                    data_parsed[row[1]] = [row[0]]
                    # print(row)
    return data_parsed

def main():

    #train
    data_gather = parseData("../data/new_set/train/train_labels.csv")
    data_gather = humoment("../data/new_set/train/", data_gather, True )

    #test
    data_gather_test = parseData("../data/new_set/test/test_labels.csv")
    data_gather_test = humoment("../data/new_set/test/", data_gather_test, False)

    labels = ["human", "item", "no_press"]
    
    # predict
    test_num = 0
    success = 0
    for label in data_gather_test:
        for hu in data_gather_test[label]:
            compare = [0,0,0]
            test_num += 1
            for d in data_gather["human"]:
                compare[0] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["human"]))
            for d in data_gather["item"]:
                compare[1] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["item"]))
            for d in data_gather["no_press"]:
                compare[2] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["no_press"]))
            if labels[compare.index(min(compare))] == label: success += 1
                
    print("test result:", str(success)+"/"+str(test_num), "Accuracy:", float(success)/float(test_num))

    # print("d1", cv2.matchShapes(data_gather["human"],data_gather["item"],cv2.CONTOURS_MATCH_I1,0))
    # print("d2", cv2.matchShapes(data_gather["human"],data_gather["item"],cv2.CONTOURS_MATCH_I2,0))
    # print("d2", cv2.matchShapes(data_gather["human"],data_gather["item"],cv2.CONTOURS_MATCH_I3,0))

    # print("d1", cv2.matchShapes(data_gather["human"],data_gather["no_press"],cv2.CONTOURS_MATCH_I1,0))
    # print("d2", cv2.matchShapes(data_gather["human"],data_gather["no_press"],cv2.CONTOURS_MATCH_I2,0))
    # print("d2", cv2.matchShapes(data_gather["human"],data_gather["no_press"],cv2.CONTOURS_MATCH_I3,0))

    # print("d1", cv2.matchShapes(data_gather["item"],data_gather["no_press"],cv2.CONTOURS_MATCH_I1,0))
    # print("d2", cv2.matchShapes(data_gather["item"],data_gather["no_press"],cv2.CONTOURS_MATCH_I2,0))
    # print("d2", cv2.matchShapes(data_gather["item"],data_gather["no_press"],cv2.CONTOURS_MATCH_I3,0))

if __name__ == '__main__':
    main()