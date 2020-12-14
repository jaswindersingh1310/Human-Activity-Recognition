


import numpy as np



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os              
from math import exp
from random import seed
from random import random

import warnings

pd.set_option('display.max_columns', None)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

dir = r'C:\Users\vinil\OneDrive - Oklahoma A and M System\Documents\Acads\Machine Learning\Project\HAPT\Test'


os.chdir(dir)

## Reading Train data file
df = pd.read_csv('data_ML.csv')

## Reading Test data file
df_Test = pd.read_csv('data_Test.csv')

# Dropping unnecessary columns which came while reading
df_Test.drop('Unnamed: 0', axis=1, inplace=True)

df.drop('Unnamed: 0', axis=1, inplace=True)


## Plot target in test data
df_Test['activity'].value_counts().plot(kind='bar')

activities=['STANDING','WALKING','LAYING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS','SITTING']

## Subsetting data to contain target from only 6 categories
df_subset_Test = df_Test.loc[df_Test['activity'].isin(activities)]



## Plot target in train data
df['activity'].value_counts().plot(kind='bar')

activities=['STANDING','WALKING','LAYING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS','SITTING']

## Subsetting train data to contain target from only 6 categories
df_subset = df.loc[df['activity'].isin(activities)]




df_subset['activity'].value_counts()


## Prepare Train X and Y data
X= df_subset.drop(['subID','activity'], 1)
Y = df_subset[['activity']]

## Prepare Test X and Y data
X_test= df_subset_Test.drop(['subID','activity'], 1)
Y_test = df_subset_Test[['activity']]


## Standardizing Train and Test X data
from sklearn.preprocessing import StandardScaler
SC_Train = StandardScaler().fit(X)

X_train = SC_Train.transform(X)

X_test = SC_Train.transform(X_test)


## This is to try TSNE for visualizing after we reduce dimensions to 2 or 3
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=3, random_state=0)
# tsne_obj= tsne.fit_transform(X_train)

# tsne_obj_test= tsne.fit_transform(X_test)
# X_new_Test = pd.DataFrame(data=tsne_obj_test)

# X_new_Train = pd.DataFrame(data=tsne_obj)



## Predictive models GBM< SVM, RF are built


RF = [RandomForestClassifier(n_estimators = 50, random_state  = 1300),
            RandomForestClassifier(n_estimators = 150, random_state  = 1300),
            RandomForestClassifier(n_estimators = 200, random_state  = 1300),
            RandomForestClassifier(n_estimators = 400, random_state  = 1300),
            RandomForestClassifier(n_estimators = 500, random_state  = 1300)]




GB_n = [GradientBoostingClassifier(n_estimators = 400, random_state  = 1300, learning_rate=0.005),
            GradientBoostingClassifier(n_estimators = 400, random_state  = 1300, learning_rate=0.01),
            GradientBoostingClassifier(n_estimators = 400, random_state  = 1300, learning_rate=0.05),
            GradientBoostingClassifier(n_estimators = 400, random_state  = 1300, learning_rate=0.1),
            GradientBoostingClassifier(n_estimators = 400, random_state  = 1300, learning_rate=0.2)]


svm_model = [SVC(kernel = 'linear', C = 1),
                    SVC(kernel = 'poly', C = 1),
                    SVC(kernel = 'rbf', C = 1),
                    SVC(kernel = 'sigmoid', C = 1)]


acc_RF_n = []

acc_SVC_n = []

acc_GB=[]



for RFM in RF :
    fit = RFM.fit(X_train, Y)
    pred = fit.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    acc_RF_n.append(accuracy)

for GB in GB_n :
    fit = GB.fit(X_train, Y)
    pred = fit.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    acc_GB.append(accuracy)
    
    

for svm in svm_model :
    fit = svm.fit(X_train, Y)
    pred = fit.predict(X_test)
    accuracy = accuracy_score(Y_test, pred)
    acc_SVC_n.append(accuracy)



labels = ('linear','ploy','rbf','sigmoid')
y_pos = np.arange(len(labels))

plt.bar(y_pos, acc_SVC_n, align='center')
plt.xticks(y_pos, labels)
plt.ylim(min(acc_SVC_n)- 0.01 , max(acc_SVC_n) +0.01)
plt.ylabel('ACCURACY PERCENTAGE')


labels = ('n_estimators-50','n_estimators-150','n_estimators-200','n_estimators-400','n_estimators-500')
y_pos = np.arange(len(labels))

plt.bar(y_pos, acc_RF_n, align='center')
plt.xticks(y_pos, labels)
plt.ylim(min(acc_RF_n)- 0.01 , max(acc_RF_n) +0.01)
plt.ylabel('ACCURACY PERCENTAGE')



labels = ('learning_rate=0.005','learning_rate=0.01','learning_rate=0.05','learning_rate=0.1','learning_rate=0.2')
y_pos = np.arange(len(labels))
plt.figure(figsize=(20,10))
plt.bar(y_pos, acc_GB, align='center')
plt.xticks(y_pos, labels)
plt.ylim(min(acc_GB)- 0.01 , max(acc_GB) +0.01)
plt.ylabel('ACCURACY PERCENTAGE')
plt.xlabel('n-estimators- 400')

## Dimensionality reduction PCA and SVD are used for dimensionality reduction and then predictive models are built





pca = [PCA(n_components=100),
       PCA(n_components=150), 
       PCA(n_components=200)]

svd = [TruncatedSVD(n_components=100, n_iter=7, random_state=42),
      TruncatedSVD(n_components=150, n_iter=7, random_state=42),
      TruncatedSVD(n_components=200, n_iter=7, random_state=42)]

## PCA
for RFM in RF :
    for pca_s in pca:
        pca_fit = pca_s.fit(X_train)
        X_train_PCA = pca_fit.transform(X_train)
        X_test_PCA = pca_fit.transform(X_test)       
        fit = RFM.fit(X_train_PCA, Y)
        pred = fit.predict(X_test_PCA)
        accuracy = accuracy_score(Y_test, pred)
        acc_RF_n.append(accuracy)


for GB in GB_n :
    for pca_s in pca:
        pca_fit = pca_s.fit(X_train)
        X_train_PCA = pca_fit.transform(X_train)
        X_test_PCA = pca_fit.transform(X_test)       
        fit = GB.fit(X_train_PCA, Y)
        pred = GB.predict(X_test_PCA)
        accuracy = accuracy_score(Y_test, pred)
        acc_GB.append(accuracy)
        
for svm in svm_model :
    for pca_s in pca:
        pca_fit = pca_s.fit(X_train)
        X_train_PCA = pca_fit.transform(X_train)
        X_test_PCA = pca_fit.transform(X_test)       
        fit = svm.fit(X_train_PCA, Y)
        pred = fit.predict(X_test_PCA)
        accuracy = accuracy_score(Y_test, pred)
        acc_SVC_n.append(accuracy)

## PLOT
labels = ('linear-100pca','linear-150pca','linear-200pca','poly-100pca','poly-150pca','ploy-200pca','rbf-100pca','rbf-150pca','rbf-200pca','sigmoid-100pca','sigmoid-150pca','sigmoid-200pca')
y_pos = np.arange(len(label))

plt.bar(y_pos, acc_SVC_n, align='center')
plt.xticks(y_pos, label)
plt.ylim(min(acc_SVC_n)- 0.01 , max(acc_SVC_n) +0.01)
plt.ylabel('ACCURACY PERCENTAGE')        
        
## SVD     
for RFM in RF :
    for svd_s in svd:
        svd_fit = svd_s.fit(X_train)
        X_train_SVD = svd_fit.transform(X_train)
        X_test_SVD = svd_fit.transform(X_test)       
        fit = RFM.fit(X_train_SVD, Y)
        pred = fit.predict(X_test_SVD)
        accuracy = accuracy_score(Y_test, pred)
        acc_RF_n.append(accuracy)


for GB in GB_n :
    for svd_s in svd:
        svd_fit = svd_s.fit(X_train)
        X_train_SVD = svd_fit.transform(X_train)
        X_test_SVD = svd_fit.transform(X_test)       
        fit = RFM.fit(X_train_SVD, Y)
        pred = fit.predict(X_test_SVD)
        accuracy = accuracy_score(Y_test, pred)
        acc_GB.append(accuracy)


        
for svm in svm_model :
    for svd_s in svd:
        svd_fit = svd_s.fit(X_train)
        X_train_SVD = svd_fit.transform(X_train)
        X_test_SVD = svd_fit.transform(X_test)       
        fit = RFM.fit(X_train_SVD, Y)
        pred = fit.predict(X_test_SVD)
        accuracy = accuracy_score(Y_test, pred)
        acc_SVC_n.append(accuracy)
        

        
        
## PLOT
labels = ('linear-100svd','linear-150svd','linear-200svd','poly-100svd','poly-150svd','ploy-200svd','rbf-100svd','rbf-150svd','rbf-200svd','sigmoid-100svd','sigmoid-150svd','sigmoid-200svd')
y_pos = np.arange(len(label))

plt.bar(y_pos, acc_SVC_n, align='center')
plt.xticks(y_pos, label)
plt.ylim(min(acc_SVC_n)- 0.01 , max(acc_SVC_n) +0.01)
plt.ylabel('ACCURACY PERCENTAGE')  






