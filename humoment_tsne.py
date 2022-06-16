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
import pandas as pd 

def humoment(home, paths, isTrain):
    result = {}
    huMoment = []
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
            huMoment += [h]

        # if isTrain:
        #     result[label] = np.array(avg_hu)
        # else:
        result[label] = [huMoment]
        
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
    # data_gather = parseData("../data/new_set/train/train_labels.csv")
    # data_gather = humoment("../data/new_set/train/", data_gather, True )

    label2num={"human":0, "item":1, "no_press":2}

    #test
    data_gather_test = parseData("../data/new_set/test/test_labels.csv")
    data_gather_test = humoment("../data/new_set/test/", data_gather_test, False)
    data_gather = []
    label = []
    for d in data_gather_test:
        for i in data_gather_test[d]:
            label += [label2num[d]]
            data_gather += i
    # print(np.array(data_gather).reshape(len(data_gather_test), 7))
    np.save("test.npy", label+np.array(data_gather).reshape(len(data_gather), 7))
    #test
    data_gather_test = parseData("../data/new_set/train/train_labels.csv")
    data_gather_test = humoment("../data/new_set/train/", data_gather_test, False)
    data_gather = []
    for d in data_gather_test:
        for i in data_gather_test[d]:
            data_gather += i
    # print(np.array(data_gather).reshape(30, 7))
    np.save("train_hu.npy", np.array(data_gather).reshape(len(data_gather), 7))

    print("data SIZE",len(data_gather_test), len(data_gather_test[0]), len(data_gather_test[0][0]))

    # labels = ["human", "item", "no_press"]
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(np.array(data_gather_test).reshape(30, 7))
    df = pd.DataFrame()
    df["y"] = data_gather_test
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df).set(title="Iris data T-SNE projection") 
    
    # # predict
    # test_num = 0
    # success = 0
    # for label in data_gather_test:
    #     for hu in data_gather_test[label]:
    #         compare = [0,0,0]
    #         test_num += 1
    #         for d in data_gather["human"]:
    #             compare[0] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["human"]))
    #         for d in data_gather["item"]:
    #             compare[1] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["item"]))
    #         for d in data_gather["no_press"]:
    #             compare[2] += cv2.matchShapes(d,hu,cv2.CONTOURS_MATCH_I2,0) / float(len(data_gather["no_press"]))
    #         if labels[compare.index(min(compare))] == label: success += 1
                
    # print("test result:", str(success)+"/"+str(test_num), "Accuracy:", float(success)/float(test_num))

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