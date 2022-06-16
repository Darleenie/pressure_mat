
import torch
import numpy as np
from tensorflow import keras
from copy import deepcopy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch_hd.hdlayers as hd
import torch_hd.utils as utils
from torch.utils.data import Dataset
import pandas as pd
import os
import glob

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
        image = np.float32(np.load(img_path))
        time_series = [[]*len(image)]
        for i in range(len(image[0][0])):
            for ii in range(len(image)):
                for jj in range(len(image[0])):
                    time_series[ii] += [image[ii][jj][i]]

        print(np.array(time_series).shape)
        # print(time_series_situp_temp[:7])#len = 7 (valid data)
        # print(label.double())
        if self.transform:
            time_series = self.transform(np.array(time_series))
        if self.target_transform:
            label = self.target_transform(label)
        return time_series, label

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#training data 
# x_train = PressMat(annotations_file="../data/time_series/train/train_labels.csv", transform=transform, img_dir="../data/time_series/train")
# #test data
# x_test = PressMat(annotations_file="../data/time_series/test/test_labels.csv", transform=transform, img_dir="../data/time_series/test")

def dataGen(path):
    for file in glob.glob(path+"train/*.npy")
        image = np.float32(np.load(img_path))
                time_series = [[]*len(image)]
                for i in range(len(image[0][0])):
                    for ii in range(len(image)):
                        for jj in range(len(image[0])):
                            time_series[ii] += [image[ii][jj][i]]
    for file in glob.glob(path+"train/*.npy")
        image = np.float32(np.load(img_path))
                time_series = [[]*len(image)]
                for i in range(len(image[0][0])):
                    for ii in range(len(image)):
                        for jj in range(len(image[0])):
                            time_series[ii] += [image[ii][jj][i]]
    
    return 
print("x_train type", type(x_train))
# for file
(x_train, y_train), (x_test, y_test) = dataGen("../data/time_series")

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

D = 10000 # dimensions, e.g., 5000
alpha = 1.0 # learning rate; I found 3.0, 1.0, 4.0, and 0.1 best for isolet, ucihar, face, and mnist
L = 100 # level hypervectors (determines quantization as well). L = 32 is enough for most cases
epoch = 20 # for most of them 20-25 is fine. mnist needs more, e.g., 40.
count = 1 # number of total simulations (to average out the randomness variation). default = 1

def encoding(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, N):
    enc_hv = []
    for i in range(len(X_data)):
		#if i % 100 == 0:
        #print(i)
        sum = np.array([0] * D)
        for j in range(len(X_data[i])):
            nGramHV = np.array([0] * D)
            for k in range(N):
                bin = min( int((X_data[i+k][j] - x_min)/bin_len), L-1)
                nGramHV = nGramHV + np.roll(lvl_hvs[bin], i)
            sum += nGramHV
        enc_hv.append(sum)
        # pass
    return enc_hv
	
def max_match(class_hvs, enc_hv, class_norms):
    max_score = -np.inf
    max_index = -1
    for i in range(len(class_hvs)):
        score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
        if score > max_score:
            max_score = score
            max_index = i
    return max_index


for iter in range(0, count):
	#split 20% of train data for validation
	permvar = np.arange(0, len(x_train))
	np.random.shuffle(permvar)
	x_train = [x_train[i] for i in permvar]
	y_train = [y_train[i] for i in permvar]
	cnt_vld = int(0.2 * len(x_train))
	x_validation = x_train[0:cnt_vld]
	y_validation = y_train[0:cnt_vld]
	x_train = x_train[cnt_vld:]
	y_train = y_train[cnt_vld:]

	#generate random id hypervectors with 50%/50% -1/1
	cnt_id = len(x_train[0])
	id_hvs = []
	for i in range(cnt_id):
		temp = [-1]*int(D/2) + [1]*int(D/2)
		np.random.shuffle(temp)
		id_hvs.append(np.asarray(temp))
	#id_hvs = map(np.int8, id_hvs)
		
	#generate level hypervectors; each level(k+1) has exactly 0.5D/L different bits (than level(k) (chosen randomly w/o replacement)
	lvl_hvs = []
	temp = [-1]*int(D/2) + [1]*int(D/2)
	np.random.shuffle(temp)
	lvl_hvs.append(temp)
	change_list = np.arange(D)
	np.random.shuffle(change_list)
	cnt_toChange = int(D/2 / (L-1))
	for i in range(1, L):
		temp = np.array(lvl_hvs[i-1])
		temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
		lvl_hvs.append(np.asarray(temp))
	#lvl_hvs = map(np.int8, lvl_hvs)

	#generate encoded hypervectors of training, validation, and test
	x_min = min( np.min(x_train), np.min(x_validation) )
	x_max = max( np.max(x_train), np.max(x_validation) )
	bin_len = (x_max - x_min)/float(L)
	train_enc_hvs = encoding(x_train, lvl_hvs, id_hvs, D, bin_len, x_min, 5)
	validation_enc_hvs = encoding(x_validation, lvl_hvs, id_hvs, D, bin_len, x_min, 5)
	test_enc_hvs = encoding(x_test, lvl_hvs, id_hvs, D, bin_len, x_min, 5)

	#initial training
	class_hvs = [[0.] * D] * (max(y_train) + 1)
	for i in range(len(train_enc_hvs)):
		class_hvs[y_train[i]] += train_enc_hvs[i]
	class_norms = [np.linalg.norm(hv) for hv in class_hvs]
		
	class_hvs_best = deepcopy(class_hvs)
	class_norms_best = deepcopy(class_norms)

	#retraining
	if epoch > 0:
		acc_max = -np.inf
		for i in range(epoch):
			for j in range(len(train_enc_hvs)):
				predict = max_match(class_hvs, train_enc_hvs[j], class_norms)
				if predict != y_train[j]:
					class_hvs[predict] -= np.multiply(alpha, train_enc_hvs[j]) #alpha*train_enc_hvs[j]
					class_hvs[y_train[j]] += np.multiply(alpha, train_enc_hvs[j]) #alpha*train_enc_hvs[j]
			class_norms = [np.linalg.norm(hv) for hv in class_hvs]
			correct = 0
			for j in range(len(validation_enc_hvs)):
				predict = max_match(class_hvs, validation_enc_hvs[j], class_norms)
				if predict == y_validation[j]:
					correct += 1
			acc = float(correct)/len(validation_enc_hvs)
			print(acc)
			if acc > acc_max:
				acc_max = acc
				class_hvs_best = deepcopy(class_hvs)
				class_norms_best = deepcopy(class_norms)


	#test:
	correct = 0
	#test_enc_hvs = encoding(X_test, lvl_hvs, id_hvs, D, bin_len, x_min)
	for i in range(len(test_enc_hvs)):
		predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms_best)
		if predict == y_test[i]:
			correct += 1
	print(float(correct)/len(test_enc_hvs))