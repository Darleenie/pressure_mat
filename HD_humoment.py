import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch_hd.hdlayers as hd
import torch_hd.utils as utils
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import cv2

def humoment_local(image):
    result = {}
    img = []
    moments = []
    huMoment = []
    avg_hu = [0,0,0,0,0,0,0]
    # img += [0]
    # print(paths[label], label)
    # try: 
    #     img += [cv2.threshold(np.sum(np.load(home + name), axis=2), 300, 1000, cv2.THRESH_BINARY)[1]]
    # except:
    img += [cv2.threshold(np.array(image), -1000, 1000, cv2.THRESH_BINARY)[1]]
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
    # huMoment += [np.array(h)]
        # if isTrain:
        #     result[label] = np.array(avg_hu)
        # else:
    # result[label] = huMoment
        
    return np.float32(np.array(h))

class PressMat(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, label2num={"human":0, "item":1, "no_press":2}):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # self.train = train
        self.label2num=label2num

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = np.float32(np.load(img_path))
        if len(image.shape) > 2:
            image = np.sum(image, axis=2)
        # image = read_image(img_path)
        image = humoment_local(image)
        label = torch.tensor(self.label2num[self.img_labels.iloc[idx, 1]])
        # print(label.double())
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#start
transform = transforms.Compose([
    transforms.ToTensor()
])
#feature extract

#training data 
training_data = PressMat(annotations_file="../data/new_set/train/train_labels.csv", transform=transform, img_dir="../data/new_set/train")
#test data
testing_data = PressMat(annotations_file="../data/new_set/test/test_labels.csv", transform=transform, img_dir="../data/new_set/test")

train_loader = DataLoader(training_data, batch_size = 512, shuffle = False)
test_loader = DataLoader(testing_data, batch_size = 512, shuffle = False)

#encoder setup
encoder = nn.Sequential(
        nn.Flatten(),
        hd.RandomProjectionEncoder(dim_in = 7, D = 10000, dist = 'bernoulli') #D, dim_in:7
    )
model = hd.HDClassifier(nclasses = 10, D = 10000)

#train
trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = 5)

#test
utils.test_hd(encoder, trained_model, test_loader)