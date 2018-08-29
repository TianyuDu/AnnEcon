"""
Methods for CPI prediction model.
"""
import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def load_dataset(dir: str) \
    -> pd.Series:
    """
        Read csv file, by default load exchange rate data
        CNY against USD (1 USD = X CNY)
    """
    series = pd.read_csv(
        dir,
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

def gen_sup_learning(data: np.ndarray, lag: int=1, nafill: object=0) \
    -> pd.DataFrame:
    """
        Generate superized learning problem.
        Transform the time series problem into a supervised learning
        with lag values as the training input and current value
        as target.
    """
    df = pd.DataFrame(data)
    # Each shifting creates a lag var: shift(n) to get lag n var.
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns = [df] + columns
    df = pd.concat(columns, axis=1)
    df.fillna(nafill, inplace=True)
    col_names = ["L0/current/target"] + [f"L{i}" for i in range(1, lag+1)]
    df.columns = col_names
    return df


def difference(dataset: np.ndarray, lag: int=1) \
    -> pd.Series:
    diff = list()
    for i in range(lag, len(dataset)):
        value = dataset[i] - dataset[i - lag]
        diff.append(value)
    return pd.Series(diff)


def inverse_difference(history, yhat, lag=1):
    return yhat + history[-lag]


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


def fit_lstm(train, batch_size, epoch, neurons):
    """
    """
    # The first column is 
    X, y = train[:, 1:], train[:, 1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        neurons,
        batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
        stateful=True
    ))
    model.add(keras.layers.Dense(1), name="layer")
    model.compile(loss="mean_squared_error", optimizer="adam")
    # for i in range(epoch):
    #     model.fit(
    #         X, 
    #         y, 
    #         batch_size=batch_size,
    #         validation_split=0.3,
    #         verbose=1, 
    #         shuffle=False
    #     )
    #     model.reset_states()
    model.fit(
        X, y,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.3,
        verbose=1,
        shuffle=False
    )
    return model


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

def reshape_and_split(data: np.ndarray, tar_idx: int=0) \
    -> (np.ndarray, np.ndarray):
    """
    Reshaped dataset into shape (*, 1, *) to fit in the input
    layer of model.
    tar_idx is the index of column (one output sequence in this model)
    contains
    """
    obs, fea = data.shape

    reshaped = data.reshape(obs, 1, fea)
    idx = list(range(fea))
    idx.remove(tar_idx)

    res_X = reshaped[:, :, idx]
    res_y = reshaped[:, :, [tar_idx]]
    
    return res_X, res_y