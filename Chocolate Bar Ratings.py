#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:16:10 2017

@author: johnnyhsieh
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('flavors_of_cacao.csv',engine = 'python')
dataset.head()
dataset.isnull().sum()
dataset.dropna()
dataset = dataset.replace(np.NaN, 'others')
dataset
x_train = dataset.drop(labels = ['Rating'],axis=1)
y_train = dataset['Rating']
perc = x_train['Cocoa\nPercent'].str.replace('%','')
x_train['Cocoa\nPercent'] = perc.astype(float)/100





#perc.strip('%').astype(float)/100

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
le = LabelEncoder()
x_train = x_train.values 
x_train[:,0] = le.fit_transform(x_train[:,0])
x_train[:,1] = le.fit_transform(x_train[:,1])
x_train[:,3] = le.fit_transform(x_train[:,3])
x_train[:,5] = le.fit_transform(x_train[:,5])
x_train[:,6] = le.fit_transform(x_train[:,6])
x_train[:,7] = le.fit_transform(x_train[:,7])

ohe = OneHotEncoder(categorical_features = [0,1,3,5,6,7])
x_train = ohe.fit_transform(x_train).toarray()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train
                                                 ,test_size = 0.2
                                                 ,random_state = 0)


import keras
from keras.models import Sequential,optimizers
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping
#from keras.layers.normalization import BatchNormalization

Regressor = Sequential()
Adam = optimizers.Adam(lr = 0.0001)
Regressor.add(Dense(units = 2048,activation = 'relu',kernel_initializer='uniform'
                    ,input_dim = 1672))

Regressor.add(Dense(units = 2048,activation = 'relu',kernel_initializer='uniform'))

Regressor.add(Dense(units = 1024,activation = 'relu',kernel_initializer='uniform'))

Regressor.add(Dense(units = 1024,activation = 'relu',kernel_initializer='uniform'))

Regressor.add(Dense(units = 512,activation = 'relu',kernel_initializer='uniform'))

Regressor.add(Dense(units = 256,activation = 'relu',kernel_initializer = 'uniform'))

Regressor.add(Dense(units = 128,activation = 'relu',kernel_initializer='uniform'))

Regressor.add(Dense(units= 64,activation = 'relu',kernel_initializer='uniform'))
Regressor.add(Dense(units= 32,activation = 'relu',kernel_initializer='uniform'))
Regressor.add(Dense(units = 16,activation = 'relu',kernel_initializer='uniform'))
Regressor.add(Dense(units= 8,activation = 'relu',kernel_initializer='uniform'))
Regressor.add(Dense(units = 1,activation = 'linear',kernel_initializer='uniform'))
Regressor.compile(optimizer = Adam,loss = 'mse',metrics = ['mae'])
#es = EarlyStopping(monitor = 'val_loss',patience = 10,mode = 'auto')
Regressor.fit(x_train,y_train,batch_size = 42, epochs = 200,shuffle = True
              ,validation_data = (x_test,y_test))
reslut = Regressor.predict(x_train)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(np.float64(y_train), color ='red', label = 'real rating')
plt.plot(reslut, color = 'blue', label = 'predict rating')
plt.title('chcolate bar ratings')
plt.ylabel('Rating')
plt.legend
plt.show()




