"""
Draft file for experimental purpose
"""
from typing import Tuple

import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import config
import containers
import methods
import models
from config import *
from containers import *
from methods import *
from models import *

file_dir = "/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange/DEXCHUS.csv"
print(f"Loading CSV file from {file_dir}")
series = methods.load_dataset(dir=file_dir)

c = containers.UnivariateContainer(series, config=config.data_proc_config)

m = UnivariateLSTM(c, config=neural_network_config)

# Stacked LSTM
data_dim = c.sup_num_target
timesteps = c.num_fea
# num_classes = 10  # Change to regression problem.

m2 = keras.Sequential()
m2.add(keras.layers.LSTM(
    32, return_sequences=True,
    input_shape=(timesteps, data_dim)
))
m2.add(keras.layers.LSTM(32, return_sequences=True))
m2.add(keras.layers.LSTM(32))
m2.add(keras.layers.Dense(1, activation="tanh"))

m2.compile(
    loss=keras.losses.MSE,
    optimizer="adam",
    metrics=[keras.metrics.mae])