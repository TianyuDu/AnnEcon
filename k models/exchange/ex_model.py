"""
Univariate Version of 
"""
import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from methods import *


config = {
    "batch_size": 1,
    "epoch": 30,
    "test_ratio": 0.3
}

# Load dataset.
series = load_dataset(dir="../data/DEXCHUS.csv")

# Transform to stationary data to Delta 1
raw_values = series.values
# diff would have length = len(raw_value) - 1 as it's taking the gaps.
diff_values = difference(raw_values, lag=1)

# Transform
# Use the current gap of differencing to predict the next gap differencing.
sup = gen_sup_learning(diff_values, 4)
sup = sup.values

# Split Data Set
total_sample_size = len(sup)
test_size = int(total_sample_size * config["test_ratio"])

print(
    f"Total sample found {total_sample_size}, {test_size} will be used as test set."
)
train, test = sup[0: -test_size], sup[-test_size:]

# Scaling
scaler, train_scaled, test_scaled = scale(train, test)

# Fit model
model = fit_lstm(
    train_scaled, 
    batch_size=config["batch_size"], 
    epoch=config["epoch"], 
    neurons=128
)

# Reshape to the shape of input tensor to network.
# Then feed into the network and make predication.  
train_reshaped = train_scaled.reshape(
    train_scaled.shape[0], 1, train_scaled.shape[1]
)

test_reshaped = test_scaled

train_pred = model.predict(train_reshaped[:, :, 1:], batch_size=1)

# For test data
pred = list()
for i in range(len(test_scaled)):
    # Make one-step forecast
    X, y = test_scaled[i, 1:], test_scaled[i, 0]
    yhat = forecast_lstm(model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    pred.append(yhat)

rmse = np.sqrt(
    mean_squared_error(
        raw_values[-test_size:], pred
    )
)

print(f"Test RMSE: {rmse}")

train_pred_recons = list()
for i in range(len(train_pred)):
    X, y = train_scaled[i, 1:], train_scaled[i, 0]
    yhat = train_pred[i]
    yhat = invert_scale(scaler, X, yhat)
    yhat += raw_values[i-1]
    train_pred_recons.append(yhat)

test_pred = [None] * len(train_pred) + pred

plt.plot(raw_values, alpha=0.6, linewidth=0.6, label="Actual Data")
plt.plot(train_pred_recons, alpha=0.6, linewidth=0.6, label="Train Pred")
plt.plot(test_pred, alpha=0.6, linewidth=0.6, label="Test Pred")

plt.legend([
    "Actual Data",
    "Pred. on training set",
    "Pred on testing set"
])

plt.show()
