'''
Gives 99.112% accuracy 
'''

import tensorflow as tf 
import xlwt 
from xlwt import Workbook 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import pandas as pd 

(x_train,y_train), (x_test,y_test) = (tf.keras.datasets.mnist.load_data())

# print(x_test[0].shape)
# print(x_test[0])
print("Train Size: ",len(x_train))
print("Test Size: ",len(x_test))

# reshape the dataset to 4 dims (keras needs 4d numpy arrays)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# normalize the features (divide by 255)
x_train = x_train/255
x_test = x_test/255


# create sequential model and add layers
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())	# flatten to form the FC layer
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=10)

wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 

# c = 0
# for i in range(len(x_test)):
# 	image_index = i
# 	# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# 	pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# 	# print(pred.argmax())

# 	sheet1.write(i+1,0,i+1)
# 	sheet1.write(i+1,1,int(pred.argmax()))
# 	if pred.argmax() == y_test[i]:
# 		c+=1
# wb.save('pred_CNN.xls')
# acc = 100*c/len(x_test)
# print

print("Testing....")
test_f = pd.read_csv("test.csv").values

for i in range(len(test_f)):
	image_index = i
	# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
	pred = model.predict(test_f[image_index].reshape(1, 28, 28, 1))
	# print(pred.argmax())

	sheet1.write(i+1,0,i+1)
	sheet1.write(i+1,1,int(pred.argmax()))
wb.save('pred_CNN.xls')
print("Done!")







