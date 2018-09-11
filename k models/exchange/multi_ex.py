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
from config import *

file_dir = "./data/exchange_rates/exchange_rates_Daily.csv"

# Setting up parameters.
container = MultivariateContainer(
    file_dir,
    "DEXCAUS",
    load_multi_ex,
    CON_config)

model = MultivariateLSTM(container, NN_config)
model.fit_model(epochs=int(input("Training epochs >>> ")))
model.save_model()
