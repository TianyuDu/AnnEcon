"""
Ref: https://lectures.quantecon.org/py/pandas_panel.html
"""
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from warnings import warn

cpi = pd.read_csv("./data/CPIAUCSL.csv", delimiter=",", index_col=0)
rgdp = pd.read_csv("./data/RGDP.csv", delimiter=",", index_col=0)
unrate = pd.read_csv("./data/UNRATE.csv", delimiter=",", index_col=0)

cpi.index = pd.to_datetime(cpi.index)
rgdp.index = pd.to_datetime(rgdp.index)
unrate.index = pd.to_datetime(unrate.index)

merged = pd.merge(cpi, unrate, how="left", left_index=True, right_on="DATE")

# Filter out Nan values.
# For each time stamp, check all of data at this time stamp is not Nan.
merged = merged[
    np.all(np.logical_not(merged.isnull().values), 1)
]

ts_cpi = pd.Series(np.ravel(cpi.values), index=pd.to_datetime(cpi.index))
ts_rgdp = pd.Series(np.ravel(rgdp.values), index=pd.to_datetime(rgdp.index))
ts_unrate = pd.Series(np.ravel(unrate.values), index=pd.to_datetime(unrate.index))


# ==== Experimental Functions ====
var_list = ["CPIAUCSL", "UNRATE", "RGDP"]
source = {"CPIAUCSL": "./data/CPIAUCSL.csv",
          "RGDP": "./data/RGDP.csv",
          "UNRATE": "./data/UNRATE.csv"}


class TimeTable:
    """
    [Experimental] Create time serie table containing indicators.
    """
    def __init__(self, var_list: List[str], source: Dict[str, str], freq: str="MS"):
        """
            Loading 
        """
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
                self.series_collection[var] = series
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
        print(f"@Timetable.remove_missing: {np.sum(qualified)} with more than {threshold} missing data will be removed.")
        self.table = self.table[np.logical_not(qualified)]

tt = TimeTable(var_list, source)
tt.remove_missing()



















