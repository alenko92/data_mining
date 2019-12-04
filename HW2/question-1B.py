# CISC 6930 - Data Mining
#
# Homework 1
# Q1:
# Implement L2 regularized linear regression algorithm with Lambda ranging from 0 to 150 (integers only) 
# For each of the 6 dataset, plot both the training set MSE and the test set MSE as a function of Lambda
# (x-axis) in one graph.
# 		(b) For each of datasets 100-100, 50(1000)-100, 100(1000)-100, provide an additional
# 			graph with Lambda ranging from 1 to 150.
#
# Prof. Yijun Zhao
# 09/27/2019
# Alexey Sanko

import numpy as numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
	plt.legend([trainLgnd, testLgnd])
	plt.xlabel('Lambda Values')
	plt.ylabel('Mean Squared Error Values')

# Definition of addDefault function - adds default one to the dataset
def addDefault(xdata):
	xDataWithDefOnes = numpy.insert(xdata, 0, 1, axis=1)
	return xDataWithDefOnes


### Main Program ###
figure = plt.figure(figsize = (14, 10))
filePath = '/Users/alexeysanko/Desktop/Fordham/3rd\ Semester/Data\ Mining/HW1'

# Initiate lists for the 6 Data Sets in the format shown below
# lists = [trainingFile, testFile, #ofFeatures, PlotTitle, legend(training), legend(test), 
# 		color(training), color(test), SubplotIndx]
ds_100_100 = ['train-100-100.csv' , 'test-100-100.csv', 100, 'train-100-100 Vs test-100-100', 
	'train-100-100', 'test-100-100', 'blue', 'orange', 311]
ds_50_1000_100 = ['train-50(1000)-100.csv', 'test-1000-100.csv', 100, 'train-50(1000)-100 Vs test-1000-100', 
	'train-50(1000)-100', 'test-1000-100', 'blue', 'orange', 312]
ds_100_1000_100 = ['train-100(1000)-100.csv','test-1000-100.csv', 100, 'train-100(1000)-100 Vs test-1000-100', 
	'train-100(1000)-100', 'test-1000-100', 'blue', 'orange', 313]
allInOne = [ds_100_100, ds_50_1000_100, ds_100_1000_100 ]

# Regularized Linear Regression Algorithm
for counter in range(len(allInOne)):
	print("\nL2 regularized linear regression for " + allInOne[counter][0] + ' and ' + allInOne[counter][1])
	trainData = numpy.genfromtxt(allInOne[counter][0], delimiter = ',', skip_header = 1)
	testData = numpy.genfromtxt(allInOne[counter][1], delimiter = ',', skip_header = 1)
	xTrain = numpy.asmatrix(trainData[:, range(0, allInOne[counter][2])])
	xTrain_W_Ones = addDefault(xTrain)
	yTrain = numpy.asmatrix(trainData[:, [allInOne[counter][2]]])
	xTest = numpy.asmatrix(testData[:, range(0, allInOne[counter][2])])
	xTest_W_Ones = addDefault(xTest)
	yTest = numpy.asmatrix(testData[:, [allInOne[counter][2]]])

	# Lamda Values List
	lambdaList = list(range(0,151))
	meanSqdErr_TrainLst = []
	meanSqdErr_TestLst = []
	wMatx = None

	# Lambda Loop
	for innerCounter in range(len(lambdaList)):
		wMatrix = get_W_Matrix(xTrain_W_Ones, yTrain, lambdaList[innerCounter])
		# Using the same W Matrix to get the MSE for Training & Test DataSet
		meanSqdErr_TrainLst.append(get_MeanSqdErr(xTrain_W_Ones, yTrain, wMatrix))
		meanSqdErr_TestLst.append(get_MeanSqdErr(xTest_W_Ones, yTest, wMatrix))
	
	print("Training Data Set: Least Mean Squared Error value is ", numpy.min(meanSqdErr_TrainLst), 
		" for lambda = ", numpy.argmin(meanSqdErr_TrainLst))
	print("Test Data Set: Least Mean Squared Error value is ", numpy.min(meanSqdErr_TestLst), 
		" for lambda = ", numpy.argmin(meanSqdErr_TestLst))

	plot_MeanSqdErr(meanSqdErr_TrainLst, meanSqdErr_TestLst,lambdaList, allInOne[counter][3], 
		allInOne[counter][4], allInOne[counter][5], allInOne[counter][6], allInOne[counter][7], 
		allInOne[counter][8])

plt.subplots_adjust(hspace = 0.5, wspace = 0.5) 
plt.show()
figure.savefig("Plot_Q1B.pdf")
print("\nPlot successfully generated")
