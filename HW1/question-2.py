# CISC 6930 - Data Mining
#
# Homework 1
# Q2: 
# From the plots in question 1, we can tell which value of Lambda is best for each dataset 
# once we know the test data and its labels. This is not realistic in real world applications. 
# In this part, we use cross validation (CV) to set the value for Lambda. Implement the 10-fold CV 
# technique discussed in class (pseudo code given in Appendix A) to select the best Lamnda value from 
# the training set.
#           (a) Using CV technique, what is the best choice of Lamdda value and the corresponding te:                    MSE for each of the six datasets
#           (b) How do the values for Lambda and MSE obtained from CV compare to the choice of Lambda 
#               and MSE in question 1(a)?
#           (c) What are the drawbacks of CV?
#           (d) What are the factors affecting the performance of CV? 
#
# Prof. Yijun Zhao
# 09/27/2019
# Alexey Sanko

import numpy as numpy
import matplotlib.pyplot as plt
import operator

# Function Definitions
# Definition of get_W_Matrix function - it gets the weighted matrix for the training dataset
def get_W_Matrix(xMtx, yMtx, lValue):
	# Initialice an Identity Matrix with xMatrix, yMatrix and a lambda Value
	iMatrix = numpy.identity(numpy.shape(xMtx)[1])
	weightMatrix = (numpy.linalg.inv((xMtx.T * xMtx) + (lValue * iMatrix))) * ((xMtx.T * yMtx))
	return weightMatrix

# Definition of getMeanSqdErr function (Mean Squared Error)
def get_MeanSqdErr(xMtx, yMtx, wMtx):
	estYMatrix = xMtx * wMtx
	diffYMatrix = yMtx - estYMatrix
	diffSqdMatrix = numpy.square(diffYMatrix)
	mean_Sqd_Err = numpy.sum(diffSqdMatrix) / numpy.shape(xMtx)[0]
	return mean_Sqd_Err

# Definition of plot_MeanSqdErr function - plots the Mean Squared Error for Training and Test data
def plot_MeanSqdErr(meanSqdErrTrain, meanSqdErrTest,lamdaList, title, trainLgnd, testLgnd, trainColor, testColor, subplotIndx):
	plt.subplot(subplotIndx)
	plt.plot(lamdaList,meanSqdErrTrain)
	plt.plot(lamdaList,meanSqdErrTest)
	plt.title(title)
	plt.gca().set_color_cycle([trainColor, testColor])
	plt.legend([trainLgnd, testLgnd])
	plt.xlabel('Lambda Values')
	plt.ylabel('Mean Squared Error Values')

# Definition of addDefault function - adds default one to the dataset
def addDefault(xdata):
	xDataWithDefOnes = numpy.insert(xdata, 0, 1, axis=1)
	return xDataWithDefOnes


### Main Program ###
# Ploting the figure
figure = plt.figure(figsize = (14, 10))
filePath = '/Users/alexeysanko/Desktop/Fordham/3rd\ Semester/Data\ Mining/HW1'

# Initiate lists for the 6 Data Sets in the format shown below
# lists = [trainingFile, testFile, #ofFeatures, PlotTitle, legend(training), legend(test), 
# 		color(training), color(test), SubplotIndx]
dset_100_10 = ['train-100-10.csv','test-100-10.csv', 10, 'train-100-10 Vs test-100-10', 
	'train-100-10', 'test-100-10', 'blue', 'green', 231]
dset_100_100 = ['train-100-100.csv' , 'test-100-100.csv', 100, 'train-100-100 Vs test-100-100', 
	'train-100-100', 'test-100-100', 'blue', 'green', 232]
dset_1000_100 = ['train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 Vs test-1000-100', 
	'train-1000-100', 'test-1000-100', 'blue', 'green', 233]
dset_50_1000_100 = ['train-50(1000)-100.csv', 'test-1000-100.csv', 100, 'train-50(1000)-100 Vs test-1000-100', 
	'train-50(1000)-100', 'test-1000-100', 'blue', 'green', 234]
dset_100_1000_100 = ['train-100(1000)-100.csv','test-1000-100.csv', 100, 'train-100(1000)-100 Vs test-1000-100', 
	'train-100(1000)-100', 'test-1000-100', 'blue', 'green', 235]
dset_150_1000_100 = ['train-150(1000)-100.csv','test-1000-100.csv', 100, 'train-150(1000)-100 Vs test-1000-100', 
	'train-150(1000)-100', 'test-1000-100', 'blue', 'green', 236]
allInOne = [dset_100_10, dset_100_100, dset_1000_100, dset_50_1000_100, dset_100_1000_100, dset_150_1000_100]

# Number of Folds for Cross Validation
numOfFolds = 10

for counter in range(len(allInOne)):
    print('\nRunning Cross Validation for Traing DataSet ->', allInOne[counter][0])
    trainData = numpy.genfromtxt(allInOne[counter][0], delimiter = ',', skip_header = 1)
    testData = numpy.genfromtxt(allInOne[counter][1], delimiter = ',', skip_header = 1)
    xTrainData = numpy.asmatrix(trainData[:, range(0,allInOne[counter][2])])
    xTrainData_W_Ones = addDefault(xTrainData)
    yTrainData = numpy.asmatrix(trainData[:, [allInOne[counter][2]]])
    xTestData = numpy.asmatrix(testData[:, range(0,allInOne[counter][2])])
    xTestData_W_Ones = addDefault(xTestData)
    yTestData = numpy.asmatrix(testData[:, [allInOne[counter][2]]])
    
    # Split xData into 10 matrix for Cross Validation (will be used as a Traing Test set for it)
    trainData_Split = numpy.split(trainData, numOfFolds)
    lambdaList = list(range(0,151))

    # meanSqdErrTestLst = [lambdaValue, ith Fold , Mean Square Error]
    meanSqdErrTestLst = []

    for lambdas in range(len(lambdaList)):
        trainDataTemp = None
        trainDataCV = None 
        testDataCV = None
        meanSqdErr_SumFolds = 0

        # Loop for Cross Validation
        for i in range(len(trainData_Split)):
            # Create and fill taining data for Cross Validation concatenating split matrix
            trainDataTemp = numpy.delete(trainData_Split, [i], 0)
            testDataCV = trainData_Split[i]
            s_TrainData = numpy.shape(trainDataTemp)
            trainDataCV = numpy.reshape(trainDataTemp, (s_TrainData[0] * s_TrainData[1], s_TrainData[2]))
            # xMatrix
            xTrainDataCV = numpy.asmatrix(trainDataCV[:, range(0,allInOne[counter][2])])
            xTrainData_CV_W_Ones = addDefault(xTrainDataCV)
            xTestDataCV = numpy.asmatrix(testDataCV[:, range(0,allInOne[counter][2])])
            xTestData_CV_W_Ones = addDefault(xTestDataCV)
            # yMatrix
            yTrainDataCV = numpy.asmatrix(trainDataCV[:, [allInOne[counter][2]]])
            yTestDataCV = numpy.asmatrix(testDataCV[:, [allInOne[counter][2]]])
            wMatrixCV = get_W_Matrix(xTrainData_CV_W_Ones, yTrainDataCV, lambdaList[lambdas])
            meanSqdErr = get_MeanSqdErr(xTestData_CV_W_Ones, yTestDataCV, wMatrixCV)
            meanSqdErr_SumFolds += meanSqdErr
        # print('Average Mean Squared Error for all folds for lambda =', lambdaList[lambdas] , ' is = ' ,mseSumForAllFolds/numOfFolds)
        meanSqdErrTestLst.append([lambdaList[lambdas] , meanSqdErr_SumFolds/numOfFolds])  

    least_meanSqdErr_Indx = min(meanSqdErrTestLst, key = operator.itemgetter(1))
    print("After Cross Validation the Least Mean Squared Error is ", 
        least_meanSqdErr_Indx[1], " for Lambda = ", least_meanSqdErr_Indx[0])
    print("Best choice lambda obtained is ", least_meanSqdErr_Indx[0])

    wMatrix = get_W_Matrix(xTrainData_W_Ones, yTrainData, least_meanSqdErr_Indx[0] )
    meanSqdErr = get_MeanSqdErr(xTestData_W_Ones, yTestData, wMatrix )
    print("For best choice lambda", least_meanSqdErr_Indx[0], " the corresponding test set Mean Squared Error is ", meanSqdErr)
    print("")
