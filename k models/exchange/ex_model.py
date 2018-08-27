"""
...
Aug 28 2018
"""
import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from methods import *

# Load dataset.
series = load_dataset()

# Transform to stationary data. (To Delta 1)
raw_values = series.values
# diff would have length = len(raw_value) - 1 as it's taking the gaps.
diff_values = difference(raw_values, interval=1)

# Transform
# Use the current gap of differencing to predict the next gap differencing.
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# Split Data Set
train, test = supervised_values[0: 700], supervised_values[700:]

# Scaling
scaler, train_scaled, test_scaled = scale(train, test)

# Fit model
lstm_model = fit_lstm(train_scaled, 1, 10, 4)

# Reshape to the shape of input tensor to network.
# Then feed into the network and make predication.  
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# For test data
pred = list()
for i in range(len(test_scaled)):
    # Make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    pred.append(yhat)

rmse = np.sqrt(
    mean_squared_error(
        raw_values[: 700], pred
    )
)

print(f"Test RMSE: {rmse}")
