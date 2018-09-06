"""
Tutorial: Multivariate Time Series Forecasting with LSTMs in Keras.
"""
import sys
from datetime import datetime

import keras
import pandas as pd
from matplotlib import pyplot as plt
import sklearn

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
dataset = pd.read_csv("./data/pollution.csv", header=0, index_col=0, engine="c")
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
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input seq. (t-n, ..., t-1)
    # sup learning over n lagged vars.
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f"var{j+1, i}") for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]
    
    # put it all together.
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values.
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Load dataset
dataset = pd.read_csv("./data/pollution.csv", header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = sklearn.preprocessing.LabelEncoder()
# encode wind_direction
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype(np.float32)
# normalize
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed =series_to_supervised(scaled, 1, 1) 
