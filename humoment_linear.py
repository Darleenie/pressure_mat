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
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
import math

def humoment_path(home, paths, isTrain):
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
        
    return h

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
        image = np.load(img_path)
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)#changed
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.fc1 = nn.Linear(32, 16)#changed

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     output = F.log_softmax(x, dim=1)

    #     return output


def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()
    h = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("type", type(data[0][0][0][0]), data[0][0][0][0])
        # print("SIZE:", data.shape)
        output = model(data.float())
        # print("SIZE", output.type())
        h += [humoment_local(output.detach().numpy())]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return h


def test(model, device, test_loader, trained_huset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # print(data)
            # print(type(data))
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            # print("output SIZE", output.size())
            # print("output",output)
            h = humoment_local(output)
            out_h = []
            for d in trained_huset:
                out_h += [(-1) * cv2.matchShapes(d,h,cv2.CONTOURS_MATCH_I2,0)]
            # print("h", h)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print("TYPEe output", output.type())
            # print("out_h", out_h)
            # print("output", output)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = torch.Tensor(out_h).argmin()  # get the index of the max log-probability
            # print(pred, pred.type())
            # test_num = 0
            # success = 0
            # labels = ["human", "item", "no_press"]
            # # print("targetin test", target)

            # compare = [0,0,0]
            # for d in trained_huset["human"]:
            #     compare[0] += cv2.matchShapes(d,h,cv2.CONTOURS_MATCH_I2,0) / float(len(trained_huset["human"]))
            # for d in trained_huset["item"]:
            #     compare[1] += cv2.matchShapes(d,h,cv2.CONTOURS_MATCH_I2,0) / float(len(trained_huset["item"]))
            # for d in trained_huset["no_press"]:
            #     compare[2] += cv2.matchShapes(d,h,cv2.CONTOURS_MATCH_I2,0) / float(len(trained_huset["no_press"]))
            # if labels[compare.index(min(compare))] == target: correct += 1
            # pred = compare.index(min(compare))
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target[pred])

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    parser = argparse.ArgumentParser()
    #parse
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    #train
    # data_gather = {}
    # data_gather = parseData("../data/new_set/train/train_labels.csv")
    # data_gather = humoment_path("../data/new_set/train/", data_gather, True )

    #test
    # data_gather_test = parseData("../data/new_set/test/test_labels.csv")
    # data_gather_test = humoment("../data/new_set/test/", data_gather_test, False)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #mean standard
        ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)

    #training data 
    training_data = PressMat(annotations_file="../data/new_set/train/train_labels.csv", transform=transform, img_dir="../data/new_set/train")
    #test data
    testing_data = PressMat(annotations_file="../data/new_set/test/test_labels.csv", transform=transform, img_dir="../data/new_set/test")

    # print(type(test_kwargs))
    train_loader = torch.utils.data.DataLoader(training_data,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testing_data, **test_kwargs)

    device = torch.device("cpu")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    data_gather = []
    for epoch in range(1, args.epochs + 1):
        data_gather += train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, data_gather)
        scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    # labels = ["human", "item", "no_press"]
    
    # print(data_gather.values())
    # predict
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
    #         print("label", label,"compare",compare)
    #         if labels[compare.index(min(compare))] == label: success += 1
    #         #linear layer classification (RELU)
                
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