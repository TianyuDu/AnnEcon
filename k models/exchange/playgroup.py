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


