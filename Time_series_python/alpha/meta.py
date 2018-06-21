import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing


class SeriesNotFoundError(Exception):
	"""
	The Error Risen to indicate
	"""
	pass


class parameter_control():
	"""
	This logger is used to collect all user-specified
	parameters in training.
	"""
	on_server: int  # If on AWS server.
	num_periods: int
	f_horizon: int
	learning_rate: float
	epochs: int

	def __init__(self):
		if int(input("Use default parameters [0/1]: ")):
			print("Setting parameters to default...")
			self.on_server = int(
				input("On Server? [0/1]: "))
			self.num_periods = 24
			self.f_horizon = 1
			self.learning_rate = 0.001
			self.epochs = 5000
		else:
			self.get_parameters()

	def get_parameters(self):
		print("Please input training parameters")
		self.on_server = int(
			input("On Server? [0/1]: "))
		self.num_periods = int(
			input("Number of periods looking back in traning: ")
			)
		self.f_horizon = int(
			input("Numer of periods to forecast: ")
			)
		self.learning_rate = float(
			input("Learning rate: ")
			)
		self.epochs = int(
			input("Training epochs: ")
			)



fred_url_package = {
	"SP500": "https://fred.stlouisfed.org/series/SP500/downloaddata/SP500.csv",
	"CPIAUCSL": "https://fred.stlouisfed.org/series/CPIAUCSL/downloaddata/CPIAUCSL.csv"
}


def fetch_fred_single(target: str):
	print("Fetching data from Fred database...")
	try:
		url = fred_url_package[target]
	except KeyError:
		raise SeriesNotFoundError("Time series requested cannot be found in data base.")

	data = pd.read_csv(url, delimiter=",", index_col=0)
	ts = pd.Series(np.ravel(data.values), data.index, dtype=str)
	ts[ts == "."] = np.nan
	ts = ts.astype(np.float32)
	ts = ts.interpolate()

	TS = np.array(ts)
	TS = TS.reshape(-1, 1)
	print("Done.")
	return ts, TS


def fetch_local_single(dir: str):
	print("Fetching data form local database...")
	try:
		data = pd.read_csv(dir, delimiter=",", index_col=0)
	except FileNotFoundError:
		raise SeriesNotFoundError("Local time series cannot be found.")

	ts = pd.Series(np.ravel(data.values), data.index, dtype=str)
	ts[ts == "."] = np.nan
	ts = ts.astype(np.float32)
	ts = ts.interpolate()

	TS = np.array(ts)
	TS = TS.reshape(-1, 1)
	print("Done.")
	return ts, TS

# def gen_loss_tensor(
# 	y_hat: tf.Tensor,
# 	y: tf.Tensor,
# 	metric: str="mse"
# 	) -> (tf.Tensor, str):
# 	loss_pack = {
# 		"sse": tf.reduce_sum(tf.square(outputs - y), name="loss_sse"),
# 		"mse": tf.reduce_mean(tf.square(outputs - y), name="loss_mse"),
# 		"rmse": tf.sqrt(tf.reduce_mean(tf.square(outputs - y)), name="loss_rmse")
# 	}
# 	return loss_pack[metric], metric

# def test_data(series, forecast, num_periods):
# 	print("Generating testing data...")
# 	test_x_setup = TS[-(num_periods + forecast):]
# 	testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
# 	testY = TS[-(num_periods):].reshape(-1,num_periods,1)
# 	print("Done.")
# 	return testX, testY
