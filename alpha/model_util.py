"""
Meta methods and classes for model to use.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

from predefine import *
from data_util import *


def add_regularization(loss: tf.Tensor, para: "ParameterControl") -> tf.Tensor:
    """
    Add regularization term to he loss tensor.
    """
    l2 = float(para.nn["reg_para"]) * sum(
        tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
    )
    return tf.add(loss, l2)


def gen_loss_tensor(
    y_hat: tf.Tensor,
    y: tf.Tensor,
    metric: str="mse",
    ) -> (tf.Tensor, str):
    """
    Generate specific type of loss tensor based on predicted and ground truth tensor.

    metric:
        sse := sum square error.
        mse := mean squared error.
        rmse := root mean squared error.
        mae := mean absolute error.
    """
    loss_pack = {
        "sse": tf.reduce_sum(tf.square(y_hat - y), name="loss_sse"),
        "mse": tf.reduce_mean(tf.square(y_hat - y), name="loss_mse"),
        "rmse": tf.sqrt(tf.reduce_mean(tf.square(y_hat - y)), name="loss_rmse"),
        "mae": tf.reduce_sum(tf.abs(y_hat - y), name="loss_mae")
    }
    return loss_pack[metric], metric


def load_data(
    target: str,
    source: str
    ) -> (pd.Series, np.ndarray):
    """
    source:
        fred := download data from fred database.
        local := load data from local csv file.
    """
    fred_url_package = {
        "SP500": "https://fred.stlouisfed.org/series/SP500/downloaddata/SP500.csv",
        "CPIAUCSL": "https://fred.stlouisfed.org/series/CPIAUCSL/downloaddata/CPIAUCSL.csv"
    }
    if source == "fred":
        print("Fetching data from Fred database...")
        try:
            url = fred_url_package[target]
            data = pd.read_csv(url, delimiter=",", index_col=0)
        except KeyError:
            raise SeriesNotFoundError(
                "Time series requested not found in data base.")
    elif source == "local":
        print("Loading data from local file...")
        try:
            data = pd.read_csv(target, delimiter=",", index_col=0)
        except FileNotFoundError:
            raise SeriesNotFoundError(
                "Local time series not found.")
    else:
        raise SeriesNotFoundError("Data source speficied not is not allowed.")

    # Create data series collection.
    ts = pd.Series(np.ravel(data.values), data.index, dtype=str)
    if any(ts == "."):
        print("Missing data found, interpolate the missing data.")
    ts[ts == "."] = np.nan  # Replace missing data with Nan.
    ts = ts.astype(np.float32)
    ts = ts.interpolate()  # Interpolate missing data.

    series = np.array(ts)
    series = series.reshape(-1, 1)
    print("Done.")
    return ts, series


def test_data(
        series: np.ndarray,
        forecast: int,
        num_periods: int) -> (np.ndarray, np.ndarray):
    """
    Generating test data.
    """
    print("Generating testing data...")
    test_x_setup = series[-(num_periods + forecast):]
    test_x = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    test_y = series[-num_periods:].reshape(-1,num_periods,1)
    print("Done.")
    return test_x, test_y


def test_data_panel(
        panel: np.ndarray,
        target: np.ndarray,
        forecast: int,
        num_periods: int) -> (np.ndarray, np.ndarray):
    """
    Generating test data for panel data.
    """
    print("Generating testing data...")
    test_x_setup = panel[-(num_periods + forecast):, :]
    test_x = test_x_setup[:num_periods, :].reshape(-1, num_periods, panel.shape[1], 1)
    test_y = target[-num_periods:].reshape(-1, num_periods, 1)
    print("Done.")
    return test_x, test_y


def visualize(
        y_data: np.ndarray,  # Ground True value.
        y_pred_train: np.ndarray,  # Predicted value on training set.
        y_pred_test: np.ndarray,  # Predicted value on testing set.
        on_server: bool=False  # If on AWS server.
            ) -> None:
    """
    Visualize result.
    """

    # Visualize test set.
    pred_test = [None] * len(np.ravel(y_data))
    pred_test[-len(np.ravel(y_pred_test)):] = np.ravel(y_pred_test)

    # plt.plot(pd.Series(np.ravel(y_data)), alpha=0.6, linewidth=0.5)
    # plt.plot(pd.Series(pred), alpha=0.8, linewidth=0.5)

    # if not on_server:
    #     plt.show()

    now_str = datetime.strftime(datetime.now(), "%Y_%m_%d_%s")
    # plt.savefig(f"./figure/result{now_str}_test.svg", format="svg")
    # plt.close()

    pred_train = [None] * len(np.ravel(y_data))
    pred_train[0:len(y_pred_train)] = y_pred_train

    fig, ax = plt.subplots()

    ax.set_title(f"Model Training Result{now_str}")

    ax.plot(pd.Series(np.ravel(y_data)), "C0", alpha=0.6, linewidth=0.5, label="Actual Data")
    ax.plot(pd.Series(np.ravel(pred_train)), "C1", alpha=0.6, linewidth=0.5, label="Prediction on Training Data")
    ax.plot(pd.Series(np.ravel(pred_test)), "C2", alpha=0.6, linewidth=0.5, label="Prediction on Test Data")

    ax.legend()

    plt.savefig(f"./figure/result{now_str}_all.svg", format="svg")


def visualize_error(loss_record: np.ndarray) -> None:
    """
    Visualize error and training progress.
    """
    raise NotImplementedError
