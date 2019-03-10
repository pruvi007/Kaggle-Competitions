'''
Gives 74% accuracy
'''
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import csv

def handle_non_numerics(df):
    columns = df.columns.values
    
    for col in columns:
        text_to_num = {}
        
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            content = df[col].values.tolist()
            unique = set(content)
            
            x = 0
            for un in unique:
                if un not in text_to_num:
                    text_to_num[un] = x
                    x += 1
            new = []
            for item in df[col]:
                new.append( text_to_num[item] )
            df[col] = new
            
    return df


df = pd.read_csv("train.csv")
# print(df.head())


label = df['Survived']


# print(df.columns)
df.drop(['Name','Survived','Cabin'],axis=1,inplace=True)
# print(df)

df = handle_non_numerics(df)
df = df.fillna(0)
# print(df.head())

X = np.array(df).astype(float)

# X = preprocessing.scale(X)
# for i in range(10):
# 	print(X[i])

X_train,X_test,Y_train,Y_test = train_test_split(X,label,test_size=0.3)
X_train = X_train.astype(float)
X_train = preprocessing.scale(X_train)

# print(len(X_train))
# print(len(Y_train))

# print(len(X_test))
# print(len(Y_test))



# # print(test)
mlp = MLPClassifier(hidden_layer_sizes=(128,128,128),max_iter=1000)
mlp.fit(X_train,Y_train)
predictions = mlp.predict(X_test)





	

# cnf_mat = confusion_matrix(label,predictions)
# print(cnf_mat)
# report = classification_report(label,predictions)
# print(report)

# print(predictions)
y = np.array(Y_test)
# print(y)

correct = 0

for i in range(len(y)):
	if y[i] == predictions[i]:
		correct += 1
	

acc = 100*correct/len(X_test)
print(acc)



# kaggle files
train = pd.read_csv("train.csv")
train = handle_non_numerics(train)
op = train['Survived']
train.drop(['Name','Survived'],axis=1,inplace=True)
train = train.fillna(0)
train = train.astype(float)
train = preprocessing.scale(train)


test = pd.read_csv("test.csv")
test = handle_non_numerics(test)
test.drop(['Name'],axis=1,inplace=True)
test = test.fillna(0)


mlp = MLPClassifier(hidden_layer_sizes=(128,128,128),max_iter=1000)
mlp.fit(train,op)
predictions = mlp.predict(test)

print(len(predictions))
with open('predict_mlp.txt','w') as f:
	for i in range(len(predictions)):
		print(predictions[i],file=f,end='\n')
