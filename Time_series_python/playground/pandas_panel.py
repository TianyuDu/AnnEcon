"""
Ref: https://lectures.quantecon.org/py/pandas_panel.html
"""
import matplotlib.pyplot as plt
import matplotlib
from warnings import warn
from typing import List, Dict
import numpy as np
import pandas as pd

cpi = pd.read_csv("./Time_series_python/playground/data/CPIAUCSL.csv", delimiter=",", index_col=0)
rgdp = pd.read_csv("./Time_series_python/playground/data/RGDP.csv", delimiter=",", index_col=0)
unrate = pd.read_csv("./Time_series_python/playground/data/UNRATE.csv", delimiter=",", index_col=0)

cpi.index = pd.to_datetime(cpi.index)
rgdp.index = pd.to_datetime(rgdp.index)
unrate.index = pd.to_datetime(unrate.index)

merged = pd.merge(cpi, unrate, how="left", left_index=True, right_on="DATE")

# Filter out Nan values.
# For each time stamp, check all of data at this time stamp is not Nan.
merged = merged[
    np.all(np.logical_not(merged.isnull().values), 1)
]
"""
Ref: https://lectures.quantecon.org/py/pandas_panel.html
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

cpi = pd.read_csv("./Time_series_python/playground/data/CPIAUCSL.csv", delimiter=",", index_col=0)
rgdp = pd.read_csv("./Time_series_python/playground/data/RGDP.csv", delimiter=",", index_col=0)
unrate = pd.read_csv("./Time_series_python/playground/data/UNRATE.csv", delimiter=",", index_col=0)

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


def load_time_table(var_list: List[str],
                    source: Dict[str, str],
                    freq: str="M"):
    """
    [Experimental] Create time serie table containing indicators.
    """
    print("@load_time_table: Loading time series...\n\tSeries to be loaded: ")
    for var in var_list:
        print("\t\t"+var)

    series_collection = dict()
    for var in var_list:
        assert var in source.keys(), "Variable/indicator requested are not provided "
        try:
            data = pd.read_csv(source[var], delimiter=",", index_col=0)
            series = pd.Series(np.ravel(data.values), index=pd.to_datetime(data.index))
            series_collection[var] = series
        except FileNotFoundError:
            warn(f"@load_time_table: Time series {var} cannot be loaded, action: skipped.")
    print(f"@load_time_table: {len(series_collection)} series loaded successfully.")






