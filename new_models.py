# https://youtu.be/tepxdcepTbY
"""
@author: Sreenivas Bhattiprolu

Code tested on Tensorflow: 2.2.0
    Keras: 2.4.3

dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""

import numpy as np
from keras import Sequential
import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import yfinance as yf
#from datetime import datetime
Stock = "AAPL"
if os.path.exists(Stock + ".csv"):
    df = pd.read_csv(Stock + ".csv", index_col=0)
else:
    df = yf.Ticker(Stock)
    df = df.history(period="max")
    df.to_csv(Stock + ".csv")
df = pd.read_csv(Stock + '.csv')
#Read the csv file
# print(df.head()) #7 columns, including the Date. 

#Separate dates for future plotting
import datetime

def str_to_datetime(s):
  split = s.split('-')
  split2 = split[2].split(' ')
  year, month, day = int(split[0]), int(split[1]), int(split2[0])
  return datetime.datetime(year=year, month=month, day=day)

df['Date'] = pd.to_datetime(df['Date'])
train_dates = df['Date']
# print(train_dates.tail(15)) #Check last few dates. 

#Variables for training
cols = list(df)[1:6]
#Date and volume columns are not used in training. 
# print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)


# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])  # Prendre seulement la valeur cible (par exemple, la colonne "Open")


trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print(trainX)
print(trainY)
#In my case, trainX has a shape (12809, 14, 5). 
#12809 because we are looking back 14 days (12823 - 14 = 12809). 
#Remember that we cannot look back 14 days until we get to the 15th day. 
#Also, trainY has a shape (12809, 1). Our model only predicts a single value, but 
#it needs multiple variables (5 in my example) to make this prediction. 
#This is why we can only predict a single day after our training, the day after where our data ends.
#To predict more days in future, we need all the 5 variables which we do not have. 
#We need to predict all variables if we want to do that. 

# define the Autoencoder model
from keras import layers
model = Sequential()
model.add(layers.LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(layers.LSTM(32, activation='relu', return_sequences=False))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()
#Predicting...
#Libraries that will help us extract only business days in the US.
#Otherwise our dates would be wrong when we look back (or forward).  
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
#Remember that we can only predict one day in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last day in our dataset.
def predict_future(model, input_data, n_future, scaler, n_features):
    predictions = []
    current_input = input_data
    
    for _ in range(n_future):
        current_input = current_input.reshape((1, n_past, n_features))
        prediction = model.predict(current_input)
        predictions.append(prediction[0])  # Stocker la prédiction
        # Mettre à jour l'entrée avec la nouvelle prédiction (ajouter à la fin et supprimer la plus ancienne)
        current_input = np.append(current_input[:, 1:, :], [[prediction[0]]], axis=1)
    
    # Convertir les prédictions en format numpy array
    predictions = np.array(predictions)
    
    # Inverser la transformation pour revenir à l'échelle originale
    predictions_copies = np.repeat(predictions, n_features, axis=-1)
    return scaler.inverse_transform(predictions_copies)[:, 0]

n_future_days = 15
input_sequence = trainX[-1]  # Utiliser la dernière séquence connue pour commencer la prédiction
y_pred_future = predict_future(model, input_sequence, n_future_days, scaler, df_for_training.shape[1])

# Convertir les prédictions en dataframe
forecast_dates = pd.date_range(list(train_dates)[-1], periods=n_future_days, freq=us_bd).tolist()
df_forecast = pd.DataFrame({'Date': forecast_dates, 'Open': y_pred_future})

# Visualiser
sns.lineplot(x='Date', y='Open', data=original, label='Original')
sns.lineplot(x='Date', y='Open', data=df_forecast, label='Forecast')
plt.title(f'{Stock} Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.xticks(rotation=45)
plt.show()

# n_past = 16
# n_days_for_prediction=15  #let us predict past 50 days

# predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
# print(predict_period_dates)

# #Make prediction
# prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

# #Perform inverse transformation to rescale back to original range
# #Since we used 5 variables for transform, the inverse expects same dimensions
# #Therefore, let us copy our values 5 times and discard them after inverse transform
# prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
# y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# # Convert timestamp to date
# forecast_dates = [date.date() for date in predict_period_dates]
    
# df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
# df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


# original = df[['Date', 'Open']].copy()
# original['Date']=pd.to_datetime(original['Date'])
# original = original.loc[original['Date'] >= str_to_datetime('2020-5-1')]
# print(original)
# print(df_forecast)
# sns.lineplot(x='Date', y='Open', data=original, label='Original')
# sns.lineplot(x='Date', y='Open', data=df_forecast, label='Forecast')
# plt.show()