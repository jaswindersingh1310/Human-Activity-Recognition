# -*- coding: utf-8 -*-
"""ML_project_HAR_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/129sUtZT5JwiEwEmUQEWCEoDmQ_TZ6FC4

Import Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import os
import mlxtend

"""Import Data"""

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

dir_path = '/content/drive/MyDrive/ML_osu/project'

os.chdir(dir_path)
filelist = os.listdir()

labels = pd.read_table('/content/drive/MyDrive/ML_osu/labels.txt', header=None, names=['expID','userID', 'activityID', 'labelStart', 'labelEnd'], sep=' ')

"""1. Use label file to slice the experiment's accerlometer and gyroscope files. 
   2. Append them them to create a master dataset
"""

# Use label file to slice the experiment's accerlometer and gyroscope files. 
df  = pd.DataFrame()
path = dir_path
files = [f for f in os.listdir(path) if os.path.isfile(f)]

for f in files:
    exp = f.split('_')[1][3:5]
    user = f.split('_')[2][4:6]
    aORg = f.split('_')[0]
      
    labels_seg = labels[(labels['expID']==int(exp)) & (labels['userID'] == int(user))]
    labstartList =  labels_seg['labelStart']
    labelendList = labels_seg['labelEnd']
    activityList = labels_seg['activityID']
    rawfile = pd.read_table(f, sep = ' ',names=['x','y','z'], header=None)
    rawfile['expID'] = exp
    rawfile['acc_gyr'] = aORg


    for i in range(len(labstartList)):
      temp = rawfile[labstartList.iloc[i]:labelendList.iloc[i]]
      temp['activityID'] = activityList.iloc[i]
      df = pd.concat([df, temp], axis=0) #Append them them to create a master dataset

Fs = 50 # Frequency of the accerlometer

dfBackup  = df

# subset the data for activities #1 to #6 that includes the static and dynamic activities
df_subset  = dfBackup[(dfBackup['activityID']==1 ) | (dfBackup['activityID']==2)| (dfBackup['activityID']==3)| (dfBackup['activityID']==4)| (dfBackup['activityID']==5)| (dfBackup['activityID']==6) ]

df_subset = df_subset.reset_index(drop=True)

df_subset['activity_name'] = df_subset['activityID'].replace([1,2,3,4,5,6],['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING',])

activities = df_subset['activity_name'].value_counts().index

df_subset['acc_gyr'].value_counts()

### CONVERTING FROM THE LONG TABLE FORMAT TO WIDE TABLE
# Subset the dataset to accerlometer and gyroscope dataset
df_acc = df_subset[df_subset['acc_gyr']=='acc']
df_gyr = df_subset[df_subset['acc_gyr']=='gyro']

df_acc = df_acc.rename(columns={"x": "x_acc", "y": "y_acc", "z": "z_acc"})
df_gyr = df_gyr.rename(columns={"x": "x_gyr", "y": "y_gyr", "z": "z_gyr"})

# Time Series Plot of one second of various activities
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, df_acc.index, df_acc['x_acc'], 'X-Axis')
    plot_axis(ax1, df_acc.index, df_acc['y_acc'], 'Y-Axis')
    plot_axis(ax2, df_acc.index, df_acc['z_acc'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in activities:
    data_for_plot = df_acc[(df_acc['activity_name'] == activity)][:Fs*1]
    plot_activity(activity, data_for_plot)

df_acc.head()

# converting to wide table with 6 features
df_final = pd.concat([df_acc.reset_index(drop=True).drop(['activity_name','acc_gyr','activity_name', 'expID'], axis=1), df_gyr.reset_index(drop=True).drop(['activity_name','acc_gyr','activity_name', 'activityID', 'expID'], axis=1)], axis=1)

df_final.columns

"""## standardize the data"""

# creating input feature matrix and labels
X = df_final[['x_acc', 'y_acc', 'z_acc',  'x_gyr', 'y_gyr', 'z_gyr']]
y = df_final['activityID']

# Standarize the x,y,z measures
scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x_acc', 'y_acc', 'z_acc',  'x_gyr', 'y_gyr', 'z_gyr'])
scaled_X['activityID'] = y.values

scaled_X

"""Frame Preparation"""

### FRAME PREPARATION
import scipy.stats as stats

# Defining Constants and calculations
Fs = 50 # frequency of accerlometer/gyroscope
t = 1 #batch size of sample
frame_size = Fs*t
hop_size = int(Fs*t*0.7) # 50% hop

# Function to create frames with time: 't' and hop size: 'h'
def get_frames(df, frame_size, hop_size):

    N_FEATURES = 6

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x_acc = df['x_acc'].values[i: i + frame_size]
        y_acc = df['y_acc'].values[i: i + frame_size]
        z_acc = df['z_acc'].values[i: i + frame_size]

        x_gyr = df['x_gyr'].values[i: i + frame_size]
        y_gyr = df['y_gyr'].values[i: i + frame_size]
        z_gyr = df['z_gyr'].values[i: i + frame_size]

        
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['activityID'][i: i + frame_size])[0][0]
        frames.append([x_acc,y_acc,z_acc,x_gyr,y_gyr,z_gyr])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

# Create features and labels
X, y =  get_frames(scaled_X, frame_size, hop_size)

X.shape, y.shape

# Train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234, stratify = y)

X_train.shape , X_test.shape

X_train.shape[0]

"""reshape the data to make each sample a 3 d"""

X_train[0].shape, X_test[0].shape

X_test[0]

# reshape the train and test for 2D CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

X_train[0].shape, X_test[0].shape

"""2D CNN model"""

# 2D CNN model Architecture
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
model.add(Dropout(0.1)) # drop 10% neurons

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Compile
model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model fit
history = model.fit(X_train, y_train, epochs = 20, validation_data= (X_test, y_test), verbose=1)

!pip install mlxtend==0.17

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_test)

mat = confusion_matrix(y_test, y_pred) # Create Confususion matrix
plot_confusion_matrix(conf_mat=mat, class_names=activities , show_normed=True, figsize=(7,7)) # plot confusion matrix