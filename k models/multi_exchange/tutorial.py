"""
Tutorial: Multivariate Time Series Forecasting with LSTMs in Keras.
"""
import sys
from datetime import datetime

import keras
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append("/Users/tianyudu/Documents/Github/AnnEcon/k models/exchange")
import config
import containers
import methods


def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")

dataset = pd.read_csv(
    "./data/PRSA.csv",
    parse_dates=[["year", "month", "day", "hour"]],
    index_col=0,
    date_parser=parse)

# Drop number column, clean the data frame.
dataset = dataset.drop(columns=["No"])
dataset.columns = [
    "pollution", "dew", "temp", "press", 
    "wnd_dir", "wnd_spd", "snow", "rain"]
dataset.index.name = "date"
dataset["pollution"].fillna(0, inplace=True)

# Drop hr=0 to hr=23 (first 24 hrs.)
dataset = dataset[24:]
dataset.to_csv("./data/pollution.csv")
# Data cleaned, create new csv file to store the new data.

# load dataset
dataset = pd.read_csv("./data/pollution.csv", header=0, index_col=0)
values = dataset.values

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group], linewidth=0.6, alpha=0.9)
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()


# LSTM Data Preparation
