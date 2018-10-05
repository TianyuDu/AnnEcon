"""
This file contains all container object used in model training.
Container objects are designed to store all necessary data
in model training.
"""
import datetime
import warnings
from typing import Tuple, Union


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
        assert cf["method"] in self.proc_method, \
            "Data Processing method fed not avaiable."

        if cf["method"] == "diff":
            assert type(
                cf["diff.lag"]) is int and cf["diff.lag"] > 0, \
                "diff.lag should be a positive integer."
            assert type(
                cf["diff.order"]) is int and cf["diff.order"] >= 0, \
                "diff.order should be a non-negative integer."

        assert type(
            cf["test_ratio"]) is float and 0 <= cf["test_ratio"] < 1, \
            "test ratio should be a float between 0 and 1."

        assert type(
            cf["lag_for_sup"]) is int and cf["lag_for_sup"] >= 1, \
            "lag_for_sup should be an integer greater or equal to 1."

        print("Configuration check passed.")
        return True
