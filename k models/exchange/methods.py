"""
Methods for CPI prediction model.
"""
import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def load_dataset(dir: str="./data/DEXCHUS.csv") -> pd.Series:
    # Read csv file, by default load exchange rate data
    # CNY against USD (1 USD = X CNY)
    series = pd.read_csv(
        "./data/DEXCHUS.csv",
        header=0,
        index_col=0,
        squeeze=True
    )
    # In Fred CSV file, nan data is represented by "."
    series = series.replace(".", np.nan)
    series = series.astype(np.float32)
    
    print(f"Found {np.sum(series.isna())} Nan data point(s), linear interpolation is applied.")
    series = series.interpolate(method="linear")
    print("Summary on Data:")
    print(series.describe())
    return series




def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    col_names = ["Current"] + [f"L{i}" for i in range(1, lag+1)]
    df.columns = col_names
    return df


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    ar = np.array(new_row)
    ar = ar.reshape(1, len(ar))
    inverted = scaler.inverse_transform(ar)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        neurons,
        batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
        stateful=True
    ))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    for _ in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size,
                  verbose=1, shuffle=False)
        model.reset_states()
    return model


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]
