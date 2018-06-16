# General Settings.
on_server = int(input("On Server? [0/1] >>> "))


import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
if on_server:
	# If we are on a server without graphic output.
	matplotlib.use('agg', warn=False, force=True)

import matplotlib.pyplot as plt
from datetime import datetime
import data_methods

# Fetch data from Fred.
ts, TS = data_methods.fetch_fred_single("SP500")

num_periods = 48
f_horizon = 1

x_data = TS[:(len(TS) - (len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1: (len(TS) - (len(TS) % num_periods)) + 1]
y_batches = y_data.reshape(-1, num_periods, 1)


def test_data(series, forecast, num_periods):
	test_x_setup = TS[-(num_periods + forecast):]
	testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
	testY = TS[-(num_periods):].reshape(-1,num_periods,1)
	return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)

tf.reset_default_graph()

inputs = 1
hidden = 128
output = 1

X = tf.placeholder(tf.float32,
	[None, num_periods, inputs])

y = tf.placeholder(tf.float32,
	[None, num_periods, output])

basic_cell = tf.contrib.rnn.LSTMCell(
	num_units=hidden,
	cell_clip=100,
	activation=tf.nn.relu)

rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)

outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

# loss = tf.reduce_sum(tf.square(outputs - y))
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 10000

with tf.Session() as sess:
	init.run()
	for ep in range(epochs):
		sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
		if ep % 100 == 0:
			mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
			print(ep, "\tMSE", mse)
	y_pred = sess.run(outputs, feed_dict={X: X_test})
	print(y_pred)


pred = [None] * len(np.ravel(y_data))
pred[-len(np.ravel(y_pred)):] = np.ravel(y_pred)

plt.plot(pd.Series(np.ravel(y_data)), alpha=0.6, linewidth=0.3)
plt.plot(pd.Series(pred))

if not on_server:
	plt.show()

now_str = datetime.strftime(datetime.now(), "%Y_%m_%d_%s")
plt.savefig(f"./figure/result{now_str}_all.svg", format="svg")

