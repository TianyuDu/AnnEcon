"""
# TODO: add discription here
"""
import datetime
import warnings

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
    def check_config(self, cf: dict) -> bool:
        print("Configuration check passed.")
        return True


class UnivariateContainer(BaseContainer):
    """
    # TODO: add doc and change class name.
    Data container
    """

    def __init__(
            self,
            raw: np.ndarray,
            config: dict={
                "method": "diff",
                "diff_lag": 1,
                "diff_order": 1,
                "test_ratio": 0.2,
                "lag_for_sup": 32
            }):
        assert self.check_config(config)
        self.config = config
        # Input format: num_obs * num_fea
        assert len(raw.shape) == 2, \
            f"UnivariateDataContainer: Raw value is expected to have dim=2, dim={len(raw.shape)} received instead."
        print(
            f"UnivariateDataContainer: Raw values received: {raw.shape[0]} observations with {raw.shape[1]} features.")
        self.num_obs, self.num_fea = raw.shape
        self.differenced = self._difference(
            raw, lag=self.config["diff_lag"], order=self.config["diff_order"]
        )

        self.sup_set = self._gen_sup_learning(
            self.differenced, total_lag=self.config["lag_for_sup"])
        self.sup_set = self.sup_set.values

        self.sample_size = len(self.sup)
        self.test_size = int(self.sample_size * config["test_ratio"])
        self.train_size = int(self.sample_size - self.test_size)

        # Split data
        # Note: idx0 = obs idx, idx1 = feature idx
        # Note: feature idx = 0 --> target
        self.train_X, self.train_y, self.test_X, self.test_y = self._split_data(self.sup)
        
        print(f"""
            Test and training data spliting finished:
            train X shape: {self.train_X.shape},
            train y shape: {self.train_y.shape},
            test X shape: {self.test_X.shape},
            test y shape: {self.test_y.shape}""")

        # Scale the training data.
        # self.scaler, self.diff_scaled = self._scale(self.differenced)

    def __str__(self) -> str:
        repr_str = f"""Univariate Data Contrainer at {hex(id(self))}
            with {self.num_obs} obs and {self.num_fea} features,
            Supervised Learning problem generated.
            Total sample size: {self.sample_size} obs.
            Training set size: {self.train_size} obs.
            Testing set size: {self.test_size} obs.
        """
        return repr_str
    
    def __repr__(self):
        self.__str__()


    def _split_data(self):
        pass # FIXME: Stopped here Sep. 1 2018


    def _gen_sup_learning(self, data: np.ndarray, total_lag: int=1, nafill: object=0.0) \
            -> pd.DataFrame:
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
        return df

    def _difference(self, data: np.ndarray, lag: int=1, order: int=1) -> np.ndarray:
        """
        Note: set lag=1 & order=0 to use the original data.
        """
        if order != 0:
            diff = list()
            for i in range(lag, len(data)):
                val = data[i] - data[i - lag]
                diff.append(val)
            diff = np.array(diff)
            diff = diff.reshape(-1, self.num_fea)
            return self._difference(diff, lag, order-1)
        return data

    def _invert_difference(self, data: np.ndarray) -> np.ndarray:
        pass

    def _scale(self, data: np.ndarray) \
            -> (sklearn.preprocessing.StandardScaler, np.ndarray):
        scaler = sklearn.preprocessing.StandardScaler().fit(data)
        data_scaled = scaler.transform(data)
        return scaler, data_scaled

    def _invert_scale(self, data: np.ndarray):
        pass

    def reconstruct(self, data: np.ndarray):
        pass
