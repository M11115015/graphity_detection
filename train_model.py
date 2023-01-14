import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib as jb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

raw_data1 = pd.read_csv('benign_graphtheory.csv', header=None)
raw_data2 = pd.read_csv('malware_graphtheory.csv', header=None)
raw_data1 = raw_data1.to_numpy()
raw_data2 = raw_data2.to_numpy()
train_X = np.empty([raw_data1.shape[0]+raw_data2.shape[0],23], dtype = float)
train_Y = np.empty([raw_data1.shape[0]+raw_data2.shape[0],1], dtype = int)
count=0
for i in range(0,raw_data1.shape[0]):
  train_X[count] = raw_data1[i,2:]
  if raw_data1[i,1]==0:
    train_Y[count]=0
  elif raw_data1[i,1]>0:
    train_Y[count]=1
  count+=1
for i in range(0,raw_data2.shape[0]):
  train_X[count] = raw_data2[i,2:]
  if raw_data2[i,1]==0:
    train_Y[count]=0
  elif raw_data2[i,1]>0:
    train_Y[count]=1
  count+=1

#---------------------SVM-----------------------------
#clf=svm.SVC()  
#parameters = {
#    'C': [1, 10, 100],
#    'gamma':[0.001,0.0001]
#} 
#grid_search = GridSearchCV(clf, parameters, cv=10)
#grid_search.fit(train_X, train_Y)
#print(grid_search.best_params_)
#print(grid_search.best_score_)
#jb.dump(clf,'GraphTheory_detection_SVM')

#-----------------Random forest------------------------
#parameters = {
#    'n_estimators': [80, 85, 90, 95, 100, 105, 110, 120, 125, 130]
#} 
#RFC = RandomForestClassifier(n_estimators=105) 
#RFC.fit(train_X,train_Y.flatten())
#grid_search = GridSearchCV(RFC, parameters, cv=10)
#grid_search.fit(train_X, train_Y)
#print(grid_search.best_params_)
#print(grid_search.best_score_)
#score=np.mean(cross_val_score(RFC,train_X,train_Y,cv=10))
#print(score)
#jb.dump(RFC,'GraphTheory_detection_RF')

#---------------------KNN-----------------------------
knn=KNeighborsClassifier(n_neighbors=4)  
knn.fit(train_X,train_Y.flatten())
#parameters = {
#    'n_neighbors': [3,4,5,6,7,8,9,10]
#} 
#grid_search = GridSearchCV(knn, parameters, cv=10)
#grid_search.fit(train_X, train_Y)
#print(grid_search.best_params_)
#print(grid_search.best_score_)
jb.dump(knn,'/home/b10704118/Model_detection/GraphTheory_detection_KNN')