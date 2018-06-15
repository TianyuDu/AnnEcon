import numpy as np
import pandas as pd
from datetime import datetime
from warnings import warn

fred_url_package = {
	"SP500": "https://fred.stlouisfed.org/series/SP500/downloaddata/SP500.csv"
}

def fetch_fred_single(target: str):
	try:
		url = fred_url_package[target]
	except KeyError:
		warn("Time series requested cannot be found in data base.")

	data.read_csv(url, delimiter=",", index_col=0)
	ts = pd.Series(np.ravel(data.values), data.index)
	ts[ts == "."] = np.nan
	ts = ts.astype(np.float32)
	ts = ts.interpolate()

	TS = np.array(ts)

	return ts, TS




