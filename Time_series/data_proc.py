'''
Data Processing Functions
Mar 21 2018
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from time import time


def create_series(
    data: "pandas.core.frame.DataFrame",
    method: str = "MA", # Default handling method: moving average.
    slices: int = 100 # Number of slices used when using MA
    ) -> "pandas.core.series.Series":
    """
    Preprocessing for Fred database, to eliminate '.' data.
    """
    data[data == "."] = np.nan
    ts = pd.Series(
        data=data.values[:, 1].astype(np.float32),
        index=data.values[:, 0]
        )

    if method == "MA":
        break_points = np.linspace(0, len(ts), slices+1).astype(np.int64)
        
        for i in range(slices):
            begin, end = break_points[i], break_points[i+1]
            seg_avg = np.mean(ts[begin: end])
            ts[begin: end][ts.isna()] = seg_avg

    return ts


def gen_test_data(series, forecast, num_periods, TS) -> None:
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX, testY
