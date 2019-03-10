"""
DecisionTree gives 85% accuracy over testSize = 0.4 

"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
import random
import xlwt 
from xlwt import Workbook 
import csv

mnist = pd.read_csv("train.csv").values
print("size of MNIST: ",len(mnist))

# My MNIST model (from 10% to 70%)

size = len(mnist)
A_train = []
A_test = []
itr = []
for i in range(7):
	print("EPOCH :",(i+1))
	itr.append(i+1)
	prop = (i+1)*10
	s1 = 0.01*(size*prop)
	s2 = (size - s1)
	print("Training Size: (in %)",(prop))
	print("Train size: ",s1)
	print("Testing size: ",s2)


	clf = DecisionTreeClassifier()
	X = mnist[:,1:]
	Y = mnist[:,0]

	xtrain, xtest, train_label, test_label = tts(X,Y,test_size = (1-(i+1)/10))
	# #training set
	# xtrain = mnist[0:s1,1:]
	# train_label = mnist[0:s1,0]

	xtrain = 
	clf.fit(xtrain,train_label)
	predictions = clf.predict(xtrain)
	count = 0
	for i in range(len(predictions)):
		if predictions[i] == train_label[i]:
			count += 1
	print("Training Accuracy: {:.3f}%".format(100*count/len(predictions)))
	A_train.append(100*count/len(predictions))

	# #testing data
	# xtest = mnist[s1:,1:]
	# test_label = mnist[s1:,0]

	predictions = clf.predict(xtest)

	count = 0
	for i in range(len(predictions)):
		if predictions[i] == test_label[i]:
			count += 1

	print("Testing Accuracy: {:.3f}%".format(100*count/len(predictions)))
	A_test.append(100*count/len(predictions))

	print("\n\n")

acc = np.vstack((A_train,A_test))
report = np.vstack((itr,acc))

print("Report: ")
print(np.transpose(report))

plt.plot(np.arange(1,8),A_train,linewidth=2.0,color='r',label="Train Acc.")
plt.plot(np.arange(1,8),A_test,linewidth=2.0,color='b',label = "Test Acc.")
legend = plt.legend(loc="upper right",shadow=True)
plt.show()

print("\nLive Simulations of random samples: ")
for i in range(10):
	x = random.randint(0,1000)
	d = xtest[x]
	d.shape = (28,28)
	plt.imshow(255-d,cmap='gray')
	plt.show()
	print("Predicted Value of Image: ",clf.predict([xtest[x]]))



# ************************************************************************************************************************************
# ************************************************************************************************************************************
# Training the complete 100% MNIST dataset for Testing
# mnist = np.random.shuffle(mnist)


wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 

features = mnist[:,1:]
labels = mnist[:,0]
clf = DecisionTreeClassifier()
clf.fit(features,labels)
mnist_test = pd.read_csv("test.csv").values
# for i in range(10):
# 	print(mnist_test[i])
sheet1.write(0,0,'ImageId')
sheet1.write(0,1,'Label')
print("Training Model....")
predictions = clf.predict(mnist_test)
print("Training Complete...")

print("Writing predicitions")


for i in range(len(predictions)):
	sheet1.write(i+1,0,i+1)
	sheet1.write(i+1,1,int(predictions[i]))

wb.save('pred_randomForest.xls')

print("Done")
