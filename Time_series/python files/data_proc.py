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


def gen_test_data(TS, forecast, num_periods):
    num_inputs = TS.shape[1]
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, num_inputs)
    testY = TS[-(num_periods):, 0].reshape(-1, num_periods, 1)
    return testX, testY


def import_data():
    cpi_data = pd.read_csv("CPIAUCSL.csv", sep=",")
    mprime_data = pd.read_csv("MPRIME.csv", sep=",")
    gs1_data = pd.read_csv("GS1.csv", sep=",")

    (cpi_ts,
        mprime_ts,
        gs1_ts) = (
            create_series(cpi_data),
            create_series(mprime_data),
            create_series(gs1_data))
    assert len(cpi_ts) == len(mprime_ts) == len(gs1_ts)
    combined = pd.concat([cpi_ts, mprime_ts, gs1_ts], axis=1)
    combined.rename(
        columns={0: "CPI", 1: "MPrime", 2: "GS1"},
        inplace=True
        )
    return combined


def gen_multi_series(
                   sources: dict,
                   target: str,
                   freq: str="M",
                   global_start: str="1959-01-01",
                   global_end: str="2018-02-01"
                   ):
    """
    Generate the pandas series object that contains
    all information we have as predictor.
    """
    print("Loading time series data from sources...")
    assert target in sources.keys(),\
    "Target is not in key of sources dict."

    periods = pd.period_range(
                              global_start,
                              global_end,
                              freq=freq
                              )
    collect_data = dict()
    collect_ts = list()
    target_data = dict()
    target_ts = None

    for data_name in sources.keys():
        data_info = sources[data_name]
        file_name = data_info[0]
        freq = data_info[1]

        data = pd.read_csv(file_name, sep=",")
        # Create series object over all data.
        ts = pd.Series(
                       data.values[:, 1],
                       index=data.values[:, 0]
                       )
        # Take the subsection within range
        ts = ts[global_start: global_end]
        assert len(ts) == len(periods),\
        "Series data length should be equal: Error found in {}".format(file_name)
        ts = pd.Series(ts.values, index=periods)
        ts[ts == "."] = np.nan
        ts = ts.astype(np.float32)
        ts = ts.interpolate()
        # Save generated objects.
        if data_name == target:
            target_data[data_name] = ts.values
            target_ts = ts
        collect_ts.append(ts)
        collect_data[data_name] = ts.values
    # Combined time series (input X)
    df_x = pd.DataFrame(collect_data, dtype=np.float32)
    df_y = pd.DataFrame(target_data, dtype=np.float32)
    print("Done.")
    return (
            df_x,  # Values of X.
            collect_ts,  # List of time series of X.
            df_y,  # Values of Y.
            target_ts  # List of time series of Y.
            )
























