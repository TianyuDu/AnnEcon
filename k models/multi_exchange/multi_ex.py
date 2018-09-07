"""
Multivariate Version of prediciton.
"""
import sys
from datetime import datetime

import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

from typing import Union, List

sys.path.append("/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange")
import config
import containers
import methods


file_dir = "/Users/tianyudu/Documents/Github/AnnEcon/k models/multi_exchange/data/exchange_rates/exchange_rates_Daily.csv"

def load_multi_ex(file_dir: str) -> pd.DataFrame:
    dataset = pd.read_csv(file_dir, delimiter="\t", index_col=0)
    # Cleaning Data
    dataset.dropna(how="any", axis=0, inplace=True)
    dataset.replace(to_replace=".", value=np.NaN, inplace=True)
    dataset.fillna(method="ffill", inplace=True)
    dataset = dataset.astype(np.float32)
    return dataset

dataset = load_multi_ex(file_dir)
dataset.drop(columns=["DEXVZUS"], inplace=True)  # DEXVZUS behaved abnomally
dataset.describe()
print(dataset.head())

# set target series
target = "DEXCAUS"

values = dataset.values
print(values.shape)

num_obs, num_series = dataset.values.shape

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

visualize_raw(dataset, action="save")

# Move target to last column
y = dataset[target]
dataset.drop(columns=[target], inplace=True)
dataset = pd.concat([dataset, y], axis=1)
values = dataset.values

# Scaling data
scaler = sklearn.preprocessing.StandardScaler()
scaled = scaler.fit_transform(values)

def gen_sup(
    data: np.ndarray, max_lag=1, 
    var_names: List[str]=None, dropnan=True):

    n_vars = data.shape[1]
    y = data[:, -1]
    y = pd.DataFrame(y)
    df = pd.DataFrame(data)
    all_frams = list()

    for i in range(1, max_lag+1): 
        shifted = df.shift(i)
        cols = var_names[:]  # Retrive var names.
        for j in range(len(cols)):  # Rename columns in form var(t-i)
            cols[j] = f"{cols[j]}(t-{i})"
        shifted.columns = cols
        all_frams.append(shifted)

    all_frams.append(y)
    result = pd.concat(all_frams, axis=1)
    res_cols = list(result.columns)
    res_cols[-1] = f"(*Target){target}(t)"
    result.columns = res_cols

    assert len(s.columns) == max_lag * (n_vars - 1) + 1
    
    return result

# cols, names = list(), list()
# pd.shift(i) gives Lag-i variable on given time step.

reframed = gen_sup(scaled, 1, 1)

# split test and training
train_ratio = 0.7
train_size = int(num_obs * train_ratio)

train, test = values[:train_size, :], values[train_size:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

assert np.all(y == np.concatenate(
    [train_y.reshape(-1), test_y.reshape(-1)], axis=0))

# Formating: time_steps = 1
# @ [samples, timesteps, features]
time_steps = 1
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
