import HDFunctions
import numpy as np
import pickle
import parse_example
import sys

if len(sys.argv) != 9:
    print('incorrect number of arguments')
    print('Usage: ')
    print('1st argument: Datadirectory')
    print('2nd argument: Dataset')
    print('3rd argument: Dimensionality')
    print('4th arguemnt: Ngram')
    print('5th arguemnt: Number of levels')
    print('6th arguemnt: Number of iterations')
    print('7th Argument: Compressed flag')
    print('8th Argument: Compression scale')
    exit()

directory = sys.argv[1]
dataset = sys.argv[2]
print('Datadirectory: ' + directory)
print('Dataset: ' + dataset)
D = int(sys.argv[3])
print('D: ' + str(D))
N = int(sys.argv[4])
print('N: ' + str(N))
nLevels = int(sys.argv[5])
print('nLevels: ' + str(nLevels))
n = int(sys.argv[6])
print('Number of training iterations: ' + str(n))
compressed = int(sys.argv[7])
s = int(sys.argv[8])
if compressed == 1:
    print('Model commpression enabled with s = ' + str(s))
else:
    print('No model compression enabled')

nTrainFeatures, nTrainClasses, trainData, trainLabels = parse_example.readChoirDat('../dataset/' + directory + '/' + dataset + '_train.choir_dat')
nTestFeatures, nTestClasses, testData, testLabels = parse_example.readChoirDat('../dataset/' + directory + '/' + dataset + '_test.choir_dat') 

print('Number of classes: ' + str(nTrainClasses))
print('Number of features: ' + str(nTrainFeatures))
print('Number of training data: ' + str(len(trainData)))
print('Number of testing data: ' + str(len(testData)))

#encodes the training data, testing data, and performs the initial training of the HD model
model = HDFunctions.buildHDModel(trainData, trainLabels, testData, testLabels, D, N, nLevels, directory, compressed, s)
#retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
#accuracy = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.testHVs, model.testLabels, n)
#prints the maximum accuracy achieved
#print('the maximum accuracy is: ' + str(max(accuracy)))

#model.collectData()
#model.printData()
model.dispClassHVDist(0)
