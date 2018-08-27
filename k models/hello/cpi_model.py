import numpy as np
import pandas as pd
import matplotlib
import keras
from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler


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
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Load Dataset
series = pd.read_csv(
    "./data/CPIAUCSL.csv",
    header=0,
    index_col=0,
    squeeze=True
)

# Transform to stationary data.
raw_values = series.values
diff_values = difference(raw_values, interval=1)

# Transform
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# Split Data Set
train, test = supervised_values[0: 700], supervised_values[700:]

# Scaling
scaler, train_scaled, test_scaled = scale(train, test)

# Fit model
lstm_model = fit_lstm(train_scaled, 1, 100, 4)

# Reshape to the shape of input tensor to network.
# Then feed into the network and make predication.  
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# For test data
pred = list()
for i in range(len(test_scaled)):
    # Make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    pred.append(yhat)

rmse = sqrt(
    sklearn.metrics.mean_squared_error(
        raw_values[: 700], pred
    )
)

print(f"Test RMSE: {rmse}")