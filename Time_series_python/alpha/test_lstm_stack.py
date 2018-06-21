# https://medium.com/@anthony_sarkis/
# tensorboard-quick-start-in-5-minutes-e3ec69f673af

# General Settings.
on_server = int(input("On Server? [0/1]: "))

print("Loading Packages...")

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
if on_server:
	# If we are on a server without graphic output.
	matplotlib.use(
		"agg",
		warn=False,
		force=True
		)

import matplotlib.pyplot as plt
from datetime import datetime
from time import time
from sklearn import preprocessing

# Load predefined classes.
from meta import *
# import data_methods

print("Done.")

# Fetch data from Fred.

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


ts, TS = fetch_local_single("./data/CPIAUCSL.csv")

scaler = preprocessing.StandardScaler().fit(TS)

TS = scaler.transform(TS)

# ts, TS = fetch_fred_single("CPIAUCSL")

num_periods = 24  # Number of periods lookingback.
f_horizon = 1  # Forecasting period.

x_data = TS[:(len(TS) - (len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1: (len(TS) - (len(TS) % num_periods)) + 1]
y_batches = y_data.reshape(-1, num_periods, 1)


def test_data(series, forecast, num_periods):
	print("Generating testing data...")
	test_x_setup = TS[-(num_periods + forecast):]
	testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
	testY = TS[-(num_periods):].reshape(-1,num_periods,1)
	print("Done.")
	return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)

tf.reset_default_graph()

inputs = 1
hidden = [64, 128]
output = 1

# Input feed node.
X = tf.placeholder(
	tf.float32,
	[None, num_periods, inputs],
	name="input_label_feed_X")

# Output node.
y = tf.placeholder(tf.float32,
	[None, num_periods, output],
	name="output_label_feed_y")

multi_layers = [
	tf.nn.rnn_cell.BasicRNNCell(num_units=hidden[0]),
	tf.nn.rnn_cell.LSTMCell(num_units=hidden[1], cell_clip=100)
	]

multi_cells = tf.nn.rnn_cell.MultiRNNCell(multi_layers)

rnn_output, states = tf.nn.dynamic_rnn(
	multi_cells,
	inputs=X,
	dtype=tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden[-1]], name="stacked_rnn_output")
stacked_outputs = tf.layers.dense(stacked_rnn_output, output, name="stacked_outputs")

outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

# Select loss/ objective function.
def gen_loss_tensor(
	y_hat: tf.Tensor,
	y: tf.Tensor,
	metric: str="mse"
	) -> (tf.Tensor, str):
	loss_pack = {
		"sse": tf.reduce_sum(tf.square(outputs - y), name="loss_sse"),
		"mse": tf.reduce_mean(tf.square(outputs - y), name="loss_mse"),
		"rmse": tf.sqrt(tf.reduce_mean(tf.square(outputs - y)), name="loss_rmse")
	}
	return loss_pack[metric], metric

loss, loss_metric = gen_loss_tensor(outputs, y, metric="mse")


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = int(input("Training epochs: "))


with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	# tf.summary.histogram("loss", loss)
	# tf.summary.histogram("outputs", outputs)
	init.run()
	print("Tensors initialized.")
	print("Training...")
	begin_time = time()

	loss_record = [1]
	for ep in range(epochs):
		sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
		if ep % 100 == 0:
			quantified_loss = loss.eval(feed_dict={X: x_batches, y: y_batches})
			loss_record.append(quantified_loss)
			print(ep, f"\t{loss_metric}:", quantified_loss)
			print(f"\tLoss change (Negative -> Improvement): {(loss_record[-1] - loss_record[-2]) / loss_record[-2] * 100} %.")
			# print(f"\t\tLoss improvement {(loss_record[-1] - loss_record[-2]) / loss_record[-2]} %.")
	y_pred = sess.run(outputs, feed_dict={X: X_test})
	print(y_pred)
	writer.close()

	# print(
	# 	f"Finished. \nTrained for {epochs} epochs. \ntime taken: {time() - begin_time()} seconds")

y_pred = scaler.inverse_transform(y_pred)
y_data = scaler.inverse_transform(y_data)

pred = [None] * len(np.ravel(y_data))
pred[-len(np.ravel(y_pred)):] = np.ravel(y_pred)

plt.plot(pd.Series(np.ravel(y_data)), alpha=0.6, linewidth=0.7)
plt.plot(pd.Series(pred), alpha=0.8, linewidth=0.7)

if not on_server:
	plt.show()

now_str = datetime.strftime(datetime.now(), "%Y_%m_%d_%s")
plt.savefig(f"./figure/result{now_str}_all.svg", format="svg")

