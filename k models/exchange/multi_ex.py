"""
Multivariate Version of prediciton.
"""
import sys
import datetime
from datetime import datetime

import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

from typing import Union, List

import config
import containers
import methods
from containers import *
from methods import *


file_dir = "./data/exchange_rates/exchange_rates_Daily.csv"

def load_multi_ex(file_dir: str) -> pd.DataFrame:
    dataset = pd.read_csv(file_dir, delimiter="\t", index_col=0)
    # Cleaning Data
    dataset.dropna(how="any", axis=0, inplace=True)
    dataset.replace(to_replace=".", value=np.NaN, inplace=True)
    dataset.fillna(method="ffill", inplace=True)
    dataset = dataset.astype(np.float32)
    # DEXVZUS behaved abnomally
    dataset.drop(columns=["DEXVZUS"], inplace=True)
    return dataset

c = MultivariateContainer(
    file_dir, 
    "DEXCAUS", 
    load_multi_ex, 
    {
        "max_lag": 3, 
        "train_ratio": 0.9,
        "time_steps": 14
    })

# Visualize raw data
def visualize_raw(data: pd.DataFrame, action: Union["save", "show"]) -> None:
    plt.close()
    plt.figure()
    values = data.values
    num_series = values.shape[1]
    wid = int(np.ceil(np.sqrt(num_series)))
    for i in range(num_series):
        plt.subplot(wid, wid, i+1)
        name = data.columns[i]
        plt.plot(values[:, i], alpha=0.6, linewidth=0.6)
        plt.title(name, y=0.5, loc="right")
    if action == "show":
        plt.show()
    elif action == "save":
        plt.savefig("raw.svg")


model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=128,
    input_shape=(c.train_X.shape[1], c.train_X.shape[2]),
    return_sequences=True
))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(1))

# TODO: change loss metric func.
model.compile(loss="mse", optimizer="adam")

epochs = int(input("Training epochs >>> "))

pin = str(datetime.datetime.now())

hist = model.fit(
    c.train_X, 
    c.train_y, 
    epochs=epochs, 
    batch_size=32)

## Testing Data

yhat = model.predict(c.test_X, verbose=1)
yhat = c.scaler_y.inverse_transform(yhat)
acty = c.scaler_y.inverse_transform(c.test_y)

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
plt.plot(acty, linewidth=0.6, alpha=0.6, label="actual")
plt.legend()
plt.savefig(f"./figure/{pin}_test.svg")

## Training Data

yhat = model.predict(c.train_X, verbose=1)
yhat = c.scaler_y.inverse_transform(yhat)
acty = c.scaler_y.inverse_transform(c.train_y)

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
plt.plot(acty, linewidth=0.6, alpha=0.6, label="actual")
plt.legend()
plt.savefig(f"./figure/{pin}_train.svg")

# yhat = model.predict(c.train_X)
# plt.close()
# plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
# plt.plot(c.train_y, linewidth=0.6, alpha=0.6, label="actual")
# plt.legend()
# plt.show()
