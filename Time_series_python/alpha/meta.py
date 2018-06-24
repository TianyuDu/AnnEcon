import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt


class ParameterControl():
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
		self.on_server = int(
			input("On Server? [0/1]: "))
		if int(input("Use default parameters [0/1]: ")):
			print("Setting parameters to default...")
			self.num_periods = 24
			self.f_horizon = 1
			self.learning_rate = 0.001
			self.epochs = 100
		else:
			self.get_parameters()
		self.load_customized_parameters()

	def get_parameters(self):
		print("Please input training parameters")
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

	def load_customized_parameters(self):
		self.nn_inputs = 1
		self.nn_hidden = [64, 128]
		self.nn_output = 1
		self.nn_reg_para = 0.005

class SeriesNotFoundError(Exception):
	"""
	The Error Risen to indicate
	"""
	pass


def add_regularization(loss: tf.Tensor, para: "ParameterControl") -> (tf.Tensor):
	l2 = float(para.nn_reg_para) * sum(
	    tf.nn.l2_loss(tf_var)
	        for tf_var in tf.trainable_variables()
	        if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
	)
	return loss + l2

def gen_loss_tensor(
	y_hat: tf.Tensor,
	y: tf.Tensor,
	metric: str="mse",
	) -> (tf.Tensor, str):
	loss_pack = {
		"sse": tf.reduce_sum(tf.square(y_hat - y), name="loss_sse"),
		"mse": tf.reduce_mean(tf.square(y_hat - y), name="loss_mse"),
		"rmse": tf.sqrt(tf.reduce_mean(tf.square(y_hat - y)), name="loss_rmse")
	}
	return loss_pack[metric], metric


def load_data(
	target: str,
	source: str
	) -> (pd.Series, np.ndarray):
	"""
		source:
			"fred": download data from fred database.
			"local": load data from local csv file.
	"""
	fred_url_package = {
		"SP500": "https://fred.stlouisfed.org/series/SP500/downloaddata/SP500.csv",
		"CPIAUCSL": "https://fred.stlouisfed.org/series/CPIAUCSL/downloaddata/CPIAUCSL.csv"
	}
	if source == "fred":
		print("Fetching data from Fred database...")
		try:
			url = fred_url_package[target]
			data = pd.read_csv(url, delimiter=",", index_col=0)
		except KeyError:
			raise SeriesNotFoundError(
				"Time series requested not found in data base.")
	elif source == "local":
		print("Loading data from local file...")
		try:
			data = pd.read_csv(target, delimiter=",", index_col=0)
		except FileNotFoundError:
			raise SeriesNotFoundError(
				"Local time series not found.")
	else:
		raise SeriesNotFoundError(
			"Data source speficied not is not allowed.")

	# Create data series collection.
	ts = pd.Series(np.ravel(data.values), data.index, dtype=str)
	if any(ts == "."):
		print("Missing data found, interpolate the missing data.")
	ts[ts == "."] = np.nan  # Replace missing data with Nan.
	ts = ts.astype(np.float32)
	ts = ts.interpolate()  # Interpolate missing data.

	TS = np.array(ts)
	TS = TS.reshape(-1, 1)
	print("Done.")
	return ts, TS


def test_data(
	series,
	forecast,
	num_periods,TS
	) -> (np.ndarray, np.ndarray):
	print("Generating testing data...")
	test_x_setup = TS[-(num_periods + forecast):]
	testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
	testY = TS[-(num_periods):].reshape(-1,num_periods,1)
	print("Done.")
	return testX, testY


def visualize(
	y_data: np.ndarray,  # Ground True value.
	y_pred_train: np.ndarray,  # Predicted value on training set.
	y_pred_test: np.ndarray,  # Predicted value on testing set.
	on_server: bool=False  # If on AWS server.
	) -> None:

	# Visualize test set.
	pred = [None] * len(np.ravel(y_data))
	pred[-len(np.ravel(y_pred_test)):] = np.ravel(y_pred_test)

	plt.plot(pd.Series(np.ravel(y_data)), alpha=0.6, linewidth=0.5)
	plt.plot(pd.Series(pred), alpha=0.8, linewidth=0.5)

	if not on_server:
		plt.show()

	now_str = datetime.strftime(datetime.now(), "%Y_%m_%d_%s")
	plt.savefig(f"./figure/result{now_str}_test.svg", format="svg")
	plt.close()

	full = [None] * len(np.ravel(y_data))
	full[0:len(y_pred_train)] = y_pred_train
	plt.plot(pd.Series(np.ravel(y_data)), alpha=0.6, linewidth=0.5)
	plt.plot(pd.Series(np.ravel(full)), alpha=0.6, linewidth=0.5)
	plt.savefig(f"./figure/result{now_str}_all.svg", format="svg")

def visualize_error(loss_record: np.ndarray):
	pass
