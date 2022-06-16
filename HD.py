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
        hd.RandomProjectionEncoder(dim_in = 512, D = 10000, dist = 'bernoulli') #D, dim_in:7
    )
model = hd.HDClassifier(nclasses = 10, D = 10000)

#train
trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = 5)

#test
utils.test_hd(encoder, trained_model, test_loader)