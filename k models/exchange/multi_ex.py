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
from models import *

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

# Setting up parameters.
CON_config = {
    "max_lag": 3,
    "train_ratio": 0.9,
    "time_steps": 14
}

NN_config = {
    "batch_size": 32,
    "validation_split": 0.1,
    "nn.lstm1": 64,
    "nn.lstm2": 32,
    "nn.dense1": 16
}

container = MultivariateContainer(
    file_dir, 
    "DEXCAUS", 
    load_multi_ex,
    CON_config)

model = MultivariateLSTM(container, NN_config)
model.fit_model(epochs=5)

time_stamp = str(datetime.datetime.now())

## Testing Data

yhat = model.predict(model.container.test_X)
yhat = model.container.invert_difference(yhat, range(4617, 5130), fillnone=True)

# FIXME: fix the prediction problem: output are nearly zeros

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
plt.plot(model.container.ground_truth_y, linewidth=0.6, alpha=0.6, label="actual")
plt.legend()
plt.show()
# plt.savefig(f"./figure/{pin}_test.svg")


## Training Data

yhat = model.predict(model.container.train_X)
acty = model.container.scaler_y.inverse_transform(model.container.train_y)
yhat = model.container.invert_difference(yhat, range(4617), fillnone=False)

plt.close()
plt.plot(yhat, linewidth=0.6, alpha=0.6, label="yhat")
# plt.plot(acty, linewidth=0.6, alpha=0.6, label="actual")
plt.plot(model.container.ground_truth_y, linewidth=0.6, alpha=0.6, label="actual")
plt.legend()
plt.show()
# plt.savefig(f"./figure/{pin}_train.svg")

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
