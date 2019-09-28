# CISC 6930 - Data Mining
#
# Homework 1
# Q3: 
# Fix Lambda = 1, 25, 150. For each of these values, plot a learning curve for the algorithm 
# using the dataset 1000-100.csv.
#       Note: a learning curve plots the performance (i.e., test set MSE) as a function of the 
#       size of the training set. To produce the curve, you need to draw random subsets (of 
#       increasing sizes) and record performance (MSE) on the corresponding test set when training 
#       on these subsets. In order to get smooth curves, you should repeat the process at least 10 
#       times and average the results. 
#
# Prof. Yijun Zhao
# 09/27/2019
# Alexey Sanko

import numpy as numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from operator import add

# Function Definitions
# Definition of get_W_Matrix function - it gets the weighted matrix for the training dataset
def get_W_Matrix(xMtx, yMtx, lamda):
    identityMatrix = numpy.identity(numpy.shape(xMtx)[1])
    wMtx = (numpy.linalg.inv((xMtx.T * xMtx) + (lamda * identityMatrix))) * ((xMtx.T * yMtx))
    return wMtx

# Definition of getMeanSqdErr function (Mean Squared Error)
def get_MeanSqdErr(xMtx, yMtx, wMtx):
    estYMtx = xMtx * wMtx
    diffYMtx = yMtx - estYMtx
    diffSqdMtx = numpy.square(diffYMtx)
    meanSqdErr = numpy.sum(diffSqdMtx) / numpy.shape(xMtx)[0]
    return meanSqdErr

# Definition of plot_MeanSqdErr function - plots the Mean Squared Error for Training and Test data
def plot_MeanSqdErr(mseTest, title, subplotIndex):
    plt.subplot(subplotIndex)
    plt.plot(mseTest)
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error Values')

# Definition of addDefault function - adds default one to the dataset
def addDefault(xdata):
    xData_W_Ones = numpy.insert(xdata, 0, 1, axis = 1)
    return xData_W_Ones


### Main Program ###
figure = plt.figure( figsize=(12,10) )
filePath = '/Users/alexeysanko/Desktop/Fordham/3rd\ Semester/Data\ Mining/HW1'

# Initiate lists for the 6 Data Sets in the format shown below
# lists = [trainingFile, testFile, #ofFeatures, PlotTitle, legend(training), legend(test), 
# 		color(training), color(test), SubplotIndx]
ds_100_10 = ['train-100-10.csv','test-100-10.csv', 10, 'train-100-10 Vs test-100-10', 
    'train-100-10', 'test-100-10', 'blue', 'orange', 231]
ds_100_100 = ['train-100-100.csv' , 'test-100-100.csv', 100, 'train-100-100 Vs test-100-100', 
    'train-100-100', 'test-100-100', 'blue', 'orange', 232]
ds_1000_100 = ['train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 Vs test-1000-100', 
    'train-1000-100', 'test-1000-100', 'blue', 'orange', 233]
ds_50_1000_100 = ['train-50(1000)-100.csv', 'test-1000-100.csv', 100, 'train-50(1000)-100 Vs test-1000-100', 
    'train-50(1000)-100', 'test-1000-100', 'blue', 'orange', 234]
ds_100_1000_100 = ['train-100(1000)-100.csv','test-1000-100.csv', 100, 'train-100(1000)-100 Vs test-1000-100', 
    'train-100(1000)-100', 'test-1000-100', 'blue', 'orange', 235]
ds_150_1000_100 = ['train-150(1000)-100.csv','test-1000-100.csv', 100, 'train-150(1000)-100 Vs test-1000-100', 
    'train-150(1000)-100', 'test-1000-100', 'blue', 'orange', 236]
allInOne = [ds_1000_100]

# Number of Run Times
numRuns = 10
numRandSample = 100
subPltIndx = 131

for counter in range(len(allInOne)):
    complete_TrainData= numpy.genfromtxt(allInOne[counter][0], delimiter = ',', skip_header = 1)
    testData= numpy.genfromtxt(allInOne[counter][1], delimiter = ',', skip_header = 1)
    xTData = numpy.asmatrix(testData[:, range(0, allInOne[counter][2])])
    xTestData_W_Ones = addDefault(xTData)
    yTData = numpy.asmatrix( testData[:, [allInOne[counter][2]]])
    lambdaList = list([1,25,150])
    
    for lambdas in range(len(lambdaList)):
        print("Lambda Learning Curve = ", lambdaList[lambdas])
        print("Run Iterations with Random Data Samples")
        meanSqdErr_LstSum = []

        for i in range(0, numRuns):
            randNumLst = random.sample( range(1,1000), numRandSample)
            randNumLst.sort()
            meanSqdErrLst = []

            for randnum in range(len(randNumLst)):
                trainData = complete_TrainData[:randNumLst[randnum]]
                xTrainData = numpy.asmatrix(trainData[:, range(0, allInOne[counter][2])])
                xTrainData_W_Ones = addDefault( xTrainData)
                yTrainData = numpy.asmatrix(trainData[:, [allInOne[counter][2]]])
                wMtx = get_W_Matrix(xTrainData_W_Ones, yTrainData, lambdaList[lambdas])
                meanSqdError = get_MeanSqdErr(xTestData_W_Ones, yTData, wMtx)
                meanSqdErrLst.append(meanSqdError)
                
            meanSqdErr_LstSum += meanSqdErrLst
        
        meanSqdErrLstRShape = numpy.reshape(numpy.array(meanSqdErr_LstSum), (numRuns, numRandSample))
        avg_meanSqdErrLst = numpy.divide(meanSqdErrLstRShape.sum(axis = 0), numRuns) 
        print("Average Mean Squared Error for all Iterations was \n", avg_meanSqdErrLst)
        plot_MeanSqdErr(avg_meanSqdErrLst, "Lambda Learning Curve is " + str(lambdaList[lambdas]), subPltIndx)
        subPltIndx += 1

plt.subplots_adjust(hspace = 0.5, wspace = 0.6)
plt.show()
figure.savefig("Plot_Q3.pdf")
print("\nPlot successfully generated")