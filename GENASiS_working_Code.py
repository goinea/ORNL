# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:46:44 2022

@author: adamg
"""
from sklearn.neural_network import MLPRegressor,MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#                                       WORKS!
df = pd.read_csv(r"C:\Users\adamg\Documents\Keras_Formatted_EOS_Data.csv" , header=None) #file contains no header info
print(f"Read in {len(df)} rows")
df.head()

df.replace("?", 10000, inplace=True) #10,000 is way beyond the range of columns provided so acts as an outlier

X_1 = np.array(df.drop([1], 1)) #last column contains label, so ignore it when creating X
#print(f"{X}")
y_1 = np.array(df[1]) #second column is a label which is our y(Pressure)
#                                     Using the Kaggle Scaling
#df.drop([0], 1, inplace=True)
df.head()
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=41)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.25, random_state=43)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print(f"Accuracy of MLP Classifier is:{clf.score}")
print("clf = ",clf.score)
print("X_train = ",X_train.shape)
print("X_test = ",X_test.shape)
print("y_train = ",y_train.shape)
print("y_test = ",y_test.shape)
#                               WORKS!
#X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.25, random_state=43)
nn_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=43, max_iter=1000, learning_rate='adaptive')

y_train=y_train.astype('int')
y_test=y_test.astype('int')
X_test=X_test.astype('int')
X_train=X_train.astype('int')
nn_model.fit(X_train, y_train)
nn_accuracy = nn_model.score(X_test, y_test)
print(f"Accuracy of MLP Classifier is:{nn_accuracy}")
