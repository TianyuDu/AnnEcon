"""
Multivariate Version of prediciton.
"""
import sys
import datetime

import keras
import pandas as pd
import numpy as np
import matplotlib
# TODO: for mac OS: os.name == "posix" and sys.platform == "darwin"
# Use this identifier to automatically decide the following.
on_server = bool(int(input("Training on server wihtout graphic output? [0/1] >>> ")))
if on_server:   
    matplotlib.use(
        "agg",
        warn=False,
        force=True
        )
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
    "nn.lstm1": 256,
    "nn.lstm2": 128,
    "nn.dense1": 64
}

container = MultivariateContainer(
    file_dir,
    "DEXCAUS",
    load_multi_ex,
    CON_config)

model = MultivariateLSTM(container, NN_config)
model.fit_model(epochs=int(input("Training epochs >>> ")))
model.save_model()