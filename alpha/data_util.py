"""
Data Utility.
"""
from typing import List, Dict
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn
from sklearn import preprocessing
# import matplotlib.pyplot as plt
from warnings import warn

from model_util import *
from predfine import *


class Panel:
    """
    Panel object is used to store panel data / multi-variate time series data.
    ** Panel is used when all data are stored in excel
    """
    def __init__(self, source, sheet=7, data_type="excel"):
        """
        sheet=7: default sheet for monthly data downloaded from fred.
        """
        if data_type == "excel":
            print(f"@Panel: Loading {data_type} data from {source}...")

            self.df = pd.read_excel(source, sheet_name=sheet, header=0, index_col=0)

            int_method = "linear"
            print(f"\tInterpolating data using {int_method}...")
            self.df = self.df.interpolate(method=int_method)

            print("\tDropping Nan columns...")
            self.df = self.df.dropna(axis=1)

            print(f"\tSuccessfully loaded data with {self.df.shape[0]} time steps {self.df.shape[1]} variables.")

# TODO: Delete experimental code
p = Panel("./data/fred.xls")


class TimeTable:
    """
    Time table object is used to store mutivariate time series in our model.
    ** TimeTable is used when different time series data are in different csv files
        (in the same format!! e.g. download from fred data base.)
    """
    freq: str  # Global frequency of panel data.
    missing_idx: np.ndarray  # Array of boolean, True if any of data missed in the time step.
    series_collection: dict  # Collection of time series.
    size: int  # Number of variables recorded.
    table: pd.core.frame.DataFrame  # DataFrame indexed by datetime storing all variables.

    def __init__(self, var_list: List[str], source: Dict[str, str], freq: str="MS"):
        self.freq = freq
        print("@TimeTable: Initializing time series...\n\t-->Series to be loaded: ")
        for var in var_list:
            print("\t\t--> "+var)

        self.series_collection = dict()
        data_frame_collection = dict()

        for var in var_list:
            assert var in source.keys(), "Variable/indicator requested are not provided "
            try:
                df = pd.read_csv(source[var], delimiter=",", index_col=0)
                df.index = pd.to_datetime(df.index)
                data_frame_collection[var] = df.resample(self.freq).ffill()
                # For time series predicting, we use forward fill method to meet the avaiablity of data.

                series = pd.Series(np.ravel(df.values), index=pd.to_datetime(df.index))
                self.series_collection[var] = series.resample(self.freq).ffill()
            except FileNotFoundError:
                warn(f"\t@TimeTable: Time series {var} cannot be loaded, action: skipped.")

        self.size = len(self.series_collection)
        print(f"\t@TimeTable: {self.size} series loaded successfully.")

        print("\t@Timetable: merging timetable.")
        merged = list(data_frame_collection.values())[0]
        for i in range(1, self.size):
            var = var_list[i]
            merged = pd.merge(merged,
                              data_frame_collection[var],
                              how="left",
                              left_index=True,
                              right_on="DATE")

        self.missing_idx = np.any(merged.isnull().values, 1)
        self.table = merged

        print(f"\t@TimeTable: total time step sampled: {self.table.values.shape[0]}")
        print(f"\t@TimeTable: number of time step containing missing data: {np.sum(self.missing_idx)}")

        percentage_missing = np.sum(self.missing_idx) / self.table.values.shape[0] * 100
        if percentage_missing > 5:
            warn("@TimeTable: more than 0.05 of total time stamp containing at least one missing data. ")

        print(f"\t@Timetable: percentage of missing time steps: {percentage_missing: .4}%")
        print("@TimeTable: Time table object initialized successfully.")

    def remove_missing(self, threshold=1):
        """
        Remove time stamp with more than $threshold missing data in the panel.
        """
        qualified = np.sum(self.table.isnull().values, 1) >= threshold
        print(f"@Timetable.remove_missing: {np.sum(qualified)}\
         with more than {threshold} missing data will be removed.")
        self.table = self.table[np.logical_not(qualified)]
