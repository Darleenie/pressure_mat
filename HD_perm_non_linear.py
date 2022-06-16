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

#encoder setup
D = 10000 # dimensions, e.g., 5000
alpha = 1.0 # learning rate; I found 3.0, 1.0, 4.0, and 0.1 best for isolet, ucihar, face, and mnist
L = 32 # level hypervectors (determines quantization as well). L = 32 is enough for most cases
epoch = 20 # for most of them 20-25 is fine. mnist needs more, e.g., 40.
count = 1 # number of total simulations (to average out the randomness variation). default = 1

class PressMat(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, label2num={"walk":0, "jump":1, "lr_shift":2, "tiptoe":3}):
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
        image = np.load(img_path)
        # print("image shape", image.shape)
        time_image = []
        for ii in range(image.shape[0]):
            time_image += [[]]
            for jj in range(image.shape[1]):
                time_image[-1]+=[image[ii][jj][0]]
        # print("len time_image", len(time_image))
        for i in range(image.shape[2]-1):
            for ii in range(image.shape[0]):
                for jj in range(image.shape[1]):
                    time_image[ii]+=[image[ii][jj][i]]
        
        time_image = np.array(time_image)
        # time_image = encoding(np.array(time_image))
        # print("time_image shape", time_image.shape)
        
        # image = read_image(img_path)
        label = torch.tensor(self.label2num[self.img_labels.iloc[idx, 1]])
        # print(label.double())
        if self.transform:
            time_image = self.transform(time_image)
        if self.target_transform:
            label = self.target_transform(label)
        return np.float32(time_image), label


#start
transform = transforms.Compose([
    transforms.ToTensor()
])
#feature extract

#training data 
training_data = PressMat(annotations_file="../data/time_series/train/train_labels.csv", transform=transform, img_dir="../data/time_series/train")
#test data
testing_data = PressMat(annotations_file="../data/time_series/test/test_labels.csv", transform=transform, img_dir="../data/time_series/test")

train_loader = DataLoader(training_data, batch_size = 512, shuffle = False)
test_loader = DataLoader(testing_data, batch_size = 512, shuffle = False)


    
encoder = nn.Sequential(
        nn.Flatten(),
        # encoding(),
        hd.PermutationEncoder(dim_in = 512, D = 10000, dist = 'bernoulli', N=7) #D, dim_in:7
        # hd.RandomProjectionEncoder(dim_in = 3584, D = 10000, dist = 'bernoulli') #D, dim_in:7
    )
model = hd.HDClassifier(nclasses = 10, D = D)

#train
trained_model = utils.train_hd(encoder, model, train_loader, valloader = test_loader, nepochs = epoch)

#test
utils.test_hd(encoder, trained_model, test_loader)