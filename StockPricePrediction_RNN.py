# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:03:21 2021

@author: Nithya K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing the dataset
#import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:, 1:2].values


# Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

#scale the training data
training_set_scaled = sc.fit_transform(training_set)

# Creating a datastructure with 60 timesteps and 1 output
#this will get last 60 records (rolling - 0 index column means the scaled stock price itself)
# the output value for this record is the current record y_train

X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

#REshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize the RNN
regressor = Sequential()

#add lstm and dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#add second lstm and dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#add third lstm and dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#add fourth lstm and dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#add output layer
regressor.add(Dense(units=1))

#compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fit the regressor
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Making predictions

#get real data 
dataset_test = pd.read_csv('Google_Stock_Price_Train.csv')

real_stock_price = dataset_test.iloc[:, 1:2].values

# Get predictions
dataset_total = pd.concat(dataset_train['Open'], dataset_test['Open'], axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test):].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 60+len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)


# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#predict
predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the results
plt.plot(real_stock_price,color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Comparison of Real and Predicted Google Stock Prices')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
