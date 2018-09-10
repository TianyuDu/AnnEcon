"""
This file contains all container object used in model training.
Container objects are designed to store all necessary data 
in model training.
"""
import datetime
import warnings
from typing import Tuple, Union

import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class BaseContainer():
    """
    Basic class for various data container object.
    """

    def __init__(self):
        self.proc_method = ["diff"]
        print("Base Container Initialized.")

    def check_config(self, cf: dict) -> bool:
        """
            Check if the configuration dictionary fed is legal.
        """
        assert cf["method"] in self.proc_method, "Data Processing method fed not avaiable."

        if cf["method"] == "diff":
            assert type(
                cf["diff.lag"]) is int and cf["diff.lag"] > 0, "diff.lag should be a positive integer."
            assert type(
                cf["diff.order"]) is int and cf["diff.order"] >= 0, "diff.order should be a non-negative integer."

        assert type(
            cf["test_ratio"]) is float and 0 <= cf["test_ratio"] < 1, "test ratio should be a float between 0 and 1."

        assert type(
            cf["lag_for_sup"]) is int and cf["lag_for_sup"] >= 1, "lag_for_sup should be an integer greater or equal to 1."

        print("Configuration check passed.")
        return True


class UnivariateContainer(BaseContainer):
    """
    The Univariate Data Container: for time series prediction.
    Used in problem where only one time series are fed in.

    Series form: y[*] with shape (N,)
    SLP form:
        Data in: previous observations. y[t-1], y[t-2], y[t-3], etc.
        Data out: predicted yhat[t]
    """

    def __init__(
            self,
            series: pd.Series,
            config: dict = {  # This is the default config for rapid testing.
                "method": "diff",
                "diff.lag": 1,
                "diff.order": 1,
                "test_ratio": 0.2,
                "lag_for_sup": 3
            }) -> None:

        super(UnivariateContainer, self).__init__()
        assert self.check_config(
            config), "Config fed in did not pass the configuration check."
        self.config = config
        print("Configuration loaded.")

        # Input format: shape=(num_obs,)
        assert len(series.shape) == 1, \
            f"UnivariateDataContainer: Series feed is expected to have shape=(n,) as univariate series, \
            but shape={len(series.shape)} received instead."
        self.series = series
        self.num_obs = len(self.series)

        self.raw = self.series.values.reshape(self.num_obs, 1)
        print(
            f"Univariate series with {self.num_obs} obs received.")

        self.differenced = self._difference(
            self.raw, lag=self.config["diff.lag"], order=self.config["diff.order"]
        )

        self.sup_set, self.tar_idx = self._gen_sup_learning(
            self.differenced, total_lag=self.config["lag_for_sup"])
        print(
            f"Supervised Learning Problem Generated with target index {self.tar_idx}")

        self.sup_set = self.sup_set.values
        self.num_fea = self.sup_set.shape[1] - 1
        self.sup_num_target = 1

        self.sample_size = len(self.sup_set)
        self.test_size = int(self.sample_size * config["test_ratio"])
        self.train_size = int(self.sample_size - self.test_size)
        assert self.sample_size == self.test_size + self.train_size

        # Split data
        # Note: idx0 = obs idx, idx1 = feature idx
        # Note: feature idx = 0 --> target

        self.train_X, self.train_y, self.test_X, self.test_y \
            = self._split_data(
                self.sup_set,
                tar_idx=self.tar_idx
            )

        print(f"""\tTest and training data spliting finished:
            train X shape: {self.train_X.shape},
            train y shape: {self.train_y.shape},
            test X shape: {self.test_X.shape},
            test y shape: {self.test_y.shape}""")

        # Scale the training data.
        # Scaler are created based on training data set. NOT the whole dataset.
        self.scaler_in, self.train_X_scaled = self._scale(self.train_X)
        self.scaler_out, self.train_y_scaled = self._scale(self.train_y)

        self.test_X_scaled = self.scaler_in.transform(self.test_X)
        self.test_y_scaled = self.scaler_out.transform(self.test_y)

    def __str__(self) -> str:
        # TODO: Add shapes of train/test Xy sets to report string.
        repr_str = f"""\t{str(type(self))} object at {hex(id(self))}
            =========================================
            Raw Data:
                Dataset size: {self.num_obs} obs.
            =========================================
            Supervised Learning problem generated:
                Total sample size: {self.sample_size} obs.
                Training set size: {self.train_size} obs.
                Testing set size: {self.test_size} obs.
                ======================================
                training set X(in) shape: {self.train_X.shape}
                training set y(out) shape: {self.train_y.shape}
                testing set X(in) shape: {self.test_X.shape}
                testing set y(out) shape: {self.test_y.shape}
        """
        return repr_str

    def __repr__(self):
        return self.__str__()

    def _split_data(self, data: np.ndarray, tar_idx: Union[int, list] = 0) -> Tuple[np.ndarray]:
        """
        Split data into X,y,train,test sets.
        Return: (train_X, train_y, test_X, test_y), univariate.
        """
        train, test = data[:self.train_size], data[self.train_size:]

        assert train.shape[1] == self.num_fea + 1, \
            f"Got train shape: {train.shape}, expected num col: {self.num_fea} + 1"
        fea_idx = list(range(train.shape[1]))
        fea_idx.remove(tar_idx)

        train_X = train[:, fea_idx]
        train_y = train[:, tar_idx].reshape(-1, 1)

        test_X = test[:, fea_idx]
        test_y = test[:, tar_idx].reshape(-1, 1)

        return (train_X, train_y, test_X, test_y)

    def _gen_sup_learning(self, data: np.ndarray, total_lag: int = 1, nafill: object = 0.0) \
            -> (pd.DataFrame, int):
        """
            Generate superized learning problem.
            Transform the time series problem into a supervised learning
            with lag values as the training input and current value
            as target.
        """
        df = pd.DataFrame(data)
        # Each shifting creates a lag var: shift(n) to get lag n var.
        columns = [df.shift(i) for i in range(1, total_lag+1)]
        columns = [df] + columns
        df = pd.concat(columns, axis=1)
        df.fillna(nafill, inplace=True)
        col_names = ["L0/current/target"] + \
            [f"L{i}" for i in range(1, total_lag+1)]
        df.columns = col_names
        tar_idx = 0
        return (df, tar_idx)

    def _difference(self, data: np.ndarray, lag: int = 1, order: int = 1) -> np.ndarray:
        """
            Note: set lag=1 & order=0 to use the original data.
        """
        if order != 0:
            diff = list()
            for i in range(lag, len(data)):
                val = data[i] - data[i - lag]
                diff.append(val)
            diff = np.array(diff)
            diff = diff.reshape(-1, 1)
            return self._difference(diff, lag, order-1)
        return data

    def _invert_difference(self, data: np.ndarray, idx: int) -> np.ndarray:
        """
            For initial stationarity removal order=1 only.
            #TODO: add higher order of differencing support: using recursion
        """
        # idx: index in supervised learning problem (differenced set)
        # Note: Use result[i] and raw[i + 1 - lag] to predict raw[i + 1]
        assert self.config["diff.order"] == 1, \
            "Initial stationarity removal differencing with order higher than 1 are not yet supported."

        lag = self.config["diff.lag"]
        if idx - lag >= 0:
            return self.raw[idx - lag + 1] + data
        else:
            return data

    def _scale(self, data: np.ndarray) -> (
            sklearn.preprocessing.StandardScaler,
            np.ndarray):
        """
        Transform the data using Standard scaler.
        x'[i] := (x[i] - avg(x))/std(x)
        return (scaler, scaled_data)
        """
        scaler = sklearn.preprocessing.StandardScaler().fit(data)
        data_scaled = scaler.transform(data)
        return scaler, data_scaled

    def invert_scale_y(self, data: np.ndarray):
        """
        Invert scaling the ouput y-hat generated by model.
        return inverted_data
        """
        # Assert data type to be univariate time series data with shape (n,) or (n, 1)
        assert len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1), \
            f"Unexpected data array feed, should be in shape (n,) or (n,1). Get {data.shape}"
        data = data.reshape(-1, 1)
        return self.scaler_out.inverse_transform(data)

    def reconstruct(self, data: np.ndarray):
        # TODO: reconstruct.
        pass

    def get_combined_scaled(self) -> Tuple[np.ndarray]:
        """
        Return splited and scaled training/testing (y+X)
        # TODO: remove this method and directly call from model.
        """
        train_scaled = np.concatenate(
            [self.train_y_scaled, self.train_X_scaled],
            axis=1
        )

        test_scaled = np.concatenate(
            [self.test_y_scaled, self.test_X_scaled],
            axis=1
        )

        return (train_scaled, test_scaled)


class MultivariateContainer(BaseContainer):
    """
        Multivariate data container for RNN prediction.
    """

    def __init__(
            self,
            file_dir: str,
            target_col: str,
            load_data: callable,
            config: dict = {
                "max_lag": 3,
                "train_ratio": 0.9,
                "time_steps": 14
            }):
        ## ======== Pre-requiste ========
        # Load configuration.  # TODO: Add check config method.
        self.config = config

        self.dataset = load_data(file_dir)
        assert type(
            self.dataset) is pd.DataFrame, f"Illegal object returned by data retrieving method, expected: pd.DataFrame, got: {type(self.dataset)}"

        assert target_col in self.dataset.columns, f"Target column {target_col} cannot be found in DataFrame loaded."
        self.target_col = target_col
        self.ground_truth_y = self.dataset[target_col].values
        print("Dataset loaded in multi-variate container.")

        self.diff_dataset = self.dataset.diff()
        self.diff_dataset.fillna(0.0, inplace=True)
        self.values = self.diff_dataset.values # TODO: Add inverting methods.
        self.num_obs, self.num_fea = self.values.shape
        print(
            f"\tDataset with {self.num_obs} observations and {self.num_fea} variables. Dataset shape={self.dataset.shape}")
        print(f"\tTarget variable: {self.target_col}")

        # self.scaler = sklearn.preprocessing.StandardScaler()  # TODO: change scaler, so that it's only scale over training data.
        # self.scaled = self.scaler.fit_transform(self.values)

        print(f"Generating supervised learning problem...")
        self.X, self.y, self.scaler_X, self.scaler_y = self.generate_supervised_learning(
            data=self.diff_dataset,
            time_steps=self.config["time_steps"]
        )

        self.train_size = int(
            self.config["train_ratio"] * self.num_obs)  # Get training set size

        (self.train_X, self.train_y, self.test_X, self.test_y) = self.split_data(
            self.X,
            self.y,
            train_size=self.train_size
            )

        # self.reshape_data(time_steps=self.config["time_steps"])  # TODO: remove.

    def generate_supervised_learning(
            self,
            data: pd.DataFrame,
            time_steps: int) -> (np.array, np.array):
        """
        Convert data to a supervised learning problem, reframed the data points into 
        [sample, time_step, feature] format.  # TODO: write doc.
        Return X @[sample, time_steps=max_lag, num_fea] and y.
        """

        num_obs, num_fea = data.shape
        print(
            f"Creating supervise learning problem with {num_fea} variables and total {time_steps} lagged variables.")

        y = data[self.target_col]

        scaler_X = sklearn.preprocessing.StandardScaler()
        scaler_y = sklearn.preprocessing.StandardScaler()

        data = pd.DataFrame(scaler_X.fit_transform(data.values))
        y = pd.DataFrame(scaler_y.fit_transform(y.values.reshape(-1, 1)))

        value = data.values

        X = [None] * num_obs  # Create placeholder for input feed.

        for t in range(num_obs):
            if t - time_steps < 0:  # If there are not enough look back values.
                X[t] = None
            else:
                # Retrive past observations on all concurrent series (including the target).
                X[t] = value[t - time_steps: t, :]

        X = [sub for sub in X if sub is not None]  # Drop training data without enough look back values.
        X = np.array(X)

        assert X.shape == (num_obs - time_steps, time_steps, num_fea)
        
        y = y[-(X.shape[0]):]  # Drop first few target data. 
        y = np.array(y)
        y = y.reshape(-1, 1)

        print(
            f"Supervised Learning Set Generated: X = {X.shape} in format [sample, time_steps, features], y = {y.shape}")
        return X, y, scaler_X, scaler_y

    def split_data(self, X, y, train_size: int) -> Tuple[np.array]:
        """
        Generate training and testing data, both input X and target y.
        """

        # scaler = sklearn.preprocessing.StandardScaler()
        # X = scaler.fit_transform(X)

        train_X = X[:train_size, :, :]
        train_y = y[:train_size, :]

        test_X = X[train_size:, :, :]
        test_y = y[train_size:, :]

        print(f"Split data into training and testing sets \
        \n\t train_X = {train_X.shape} \
        \n\t train_y = {train_y.shape} \
        \n\t test_X = {test_X.shape} \
        \n\t test_y = {test_y.shape}")

        return (train_X, train_y, test_X, test_y)
