# Alpha Version
# Tianyu Du

# Loading Packages
from meta import *
para = ParameterControl()

print("Loading Packages...")
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
if para.on_server:
	matplotlib.use(
		"agg",
		warn=False,
		force=True
		)  # If on a server, change matplotlib settings.
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing

print("Done.")

def main2():
	

def main():

	# Prepare data.
	ts, TS = load_data("./data/CPIAUCSL.csv", "local")

	scaler = preprocessing.StandardScaler().fit(TS)
	print("Scaler built.")

	TS = scaler.transform(TS)

	num_periods = para.num_periods
	f_horizon = para.f_horizon  # Forecasting period.

	x_data = TS[:(len(TS) - (len(TS) % num_periods))]
	x_batches = x_data.reshape(-1, num_periods, 1)

	y_data = TS[1: (len(TS) - (len(TS) % num_periods)) + 1]
	y_batches = y_data.reshape(-1, num_periods, 1)

	X_test, Y_test = test_data(TS, f_horizon, num_periods)

	tf.reset_default_graph()
	print("Default graph reset.")


	# Input feed node.
	X = tf.placeholder(
		tf.float32,
		[None, num_periods, para.nn_inputs],
		name="input_label_feed_X")

	# Output node.
	y = tf.placeholder(
		tf.float32,
		[None, num_periods, para.nn_output],
		name="output_label_feed_y")

	multi_layers = [
		tf.nn.rnn_cell.BasicRNNCell(num_units=para.nn_hidden[0]),
		tf.nn.rnn_cell.LSTMCell(num_units=para.nn_hidden[1], cell_clip=100)
		]

	multi_cells = tf.nn.rnn_cell.MultiRNNCell(multi_layers)

	rnn_output, states = tf.nn.dynamic_rnn(
		multi_cells,
		inputs=X,
		dtype=tf.float32)


	stacked_rnn_output = tf.reshape(
		rnn_output,
		[-1, para.nn_hidden[-1]],
		name="stacked_rnn_output"
		)

	stacked_outputs = tf.layers.dense(
		stacked_rnn_output,
		para.nn_output,
		name="stacked_outputs"
		)

	outputs = tf.reshape(
		stacked_outputs,
		[-1, num_periods, para.nn_output]
		)


	loss, loss_metric = gen_loss_tensor(outputs, y, metric="mse")

	loss = add_regularization(loss, para)

	loss_metric += " + reg"

	optimizer = tf.train.AdamOptimizer(learning_rate=para.learning_rate)
	training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		print("Starting session...")
		print("Starting writer...")
		writer = tf.summary.FileWriter("output", sess.graph)
		tf.summary.histogram("loss", loss)
		tf.summary.histogram("outputs", outputs)
		print("Initializing tensors...")
		init.run()
		print("Training...")
		begin_time = datetime.now()

		loss_record = [1]
		for ep in range(para.epochs):
			sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
			if ep % 100 == 0:
				quantified_loss = loss.eval(feed_dict={X: x_batches, y: y_batches})
				loss_record.append(quantified_loss)
				print(ep,
					f"\t{loss_metric}:", quantified_loss)
				print(
					f"\tLoss Improvement: {-1 * ((loss_record[-1] - loss_record[-2]) / loss_record[-2] * 100)} %.")
		y_pred_train = sess.run(outputs, feed_dict={X: x_batches})
		y_pred_test = sess.run(outputs, feed_dict={X: X_test})
		writer.close()

		print("Finished, time taken {}.".format(datetime.now() - begin_time))

	# Transform back
	y_data = scaler.inverse_transform(y_data)
	y_pred_train = scaler.inverse_transform(y_pred_train)
	y_pred_train = y_pred_train.reshape(-1,1)  # Expand the stacked inputs.
	y_pred_test = scaler.inverse_transform(y_pred_test)

	visualize(
		y_data,
		y_pred_train,
		y_pred_test,
		on_server=para.on_server)

if __name__ == "__main__":
	main()


