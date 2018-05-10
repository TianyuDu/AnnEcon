"""
Created: Mar 21 2018
Join Several Economic Indicators to assist the prediction.
Update Mar 24 2018
Version 2 created: Mar 30 2018
"""
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from time import time
from data_proc import *
from constants import *
from save_data import *


# Below: Import and change of packages depends on platform.
if int(input("On AWS Server? [0/1] >>> ")):
    matplotlib.use('agg', warn=False, force=True)

(x_data_raw,
 x_ts_raw,
 y_data_raw,
 y_ts_raw
 ) = gen_multi_series(
                      data_files,
                      "DEXCAUS",
                      freq=parameters["data_freq"],
                      global_start=parameters["global_start"],
                      global_end=parameters["global_end"]
                      )

x_data_raw = x_data_raw.values
y_data_raw = y_data_raw.values

num_periods = parameters["length_sample"]
f_horizon = parameters["forecast_horizon"]

num_sample = x_data_raw.shape[0]
# Total lenght of time series data we have.
num_feature = x_data_raw.shape[1]
print("[Data Info.]\
      \n\tTotal lenght of time series sample: {},\
      \n\tNumber of feature(s): {},\
      \n\tNumber of batches: {}."\
      .format(num_sample,
              num_feature,
              num_sample // num_periods)
      )

# Drop out extra data that cannot be fit into batches
x_data = x_data_raw[:(num_sample - num_sample % num_periods)]
x_batches = x_data.reshape(-1, num_periods, num_feature)

y_data = y_data_raw[1:(num_sample - num_sample % num_periods) + f_horizon, -1]
# Y data contains one period ahead, for forcasting.
y_batches = y_data.reshape(-1, num_periods, 1)

print("x batches size: {}\
      \ny batches size: {}"\
      .format(
              x_batches.shape,
              y_batches.shape
              ))

assert x_batches.shape[:-1] == y_batches.shape[:-1]

x_train = x_batches[:-parameters["num_test_batch"]]
y_train = y_batches[:-parameters["num_test_batch"]]

x_test = x_batches[-parameters["num_test_batch"]:]
y_test = y_batches[-parameters["num_test_batch"]:]

def check():
    print("Checking training set and test set...")
    check_collection = np.concatenate([x_train, x_test])
    assert np.all(check_collection == x_batches),\
    "Failed to split X batches into correct subsets."
    check_collection = np.concatenate([y_train, y_test])
    assert np.all(check_collection == y_batches),\
    "Failed to split Y batches into correct subsets."
    print("Done.")
check()

# ==== Below: Tensorflow ==== #
tf.reset_default_graph()

inputs = num_feature  # Single Vector input.
hidden = parameters["rnn_hidden"]
output = 1

print("Creating RNN Graph...")
x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

lstm_cell = tf.contrib.rnn.LSTMCell(
    num_units=hidden,
    activation=tf.nn.relu,
    cell_clip=100000.0,
    forget_bias=0.5,
    name="LSTM_cell"
    )

rnn_output, states = tf.nn.dynamic_rnn(
    lstm_cell,
    x,
    dtype=tf.float32
    )


stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))
# Applying Mean Absolute Percentage Error.
# loss = tf.reduce_mean(
#                       tf.abs(tf.divide(outputs - y, y))
#                       )

optimizer = tf.train.AdamOptimizer(
                                   learning_rate=parameters["rnn_learning_rate"]
                                   )
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = parameters["epochs"]

start_t = time()

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(
            training_op,
                feed_dict={
                    x: x_train,
                    y: y_train
                }
            )
        if ep % 10 == 0:
            mse = loss.eval(
                    feed_dict={
                        x: x_train,
                        y: y_train
                    }
                )
            print(
                  "{},\tTraining set SE: {}".\
                  format(ep, mse * 100))

    in_range_est = sess.run(
                            outputs,
                            feed_dict={x: x_train}
                            )
    y_pred = sess.run(
                      outputs, feed_dict={x: x_test}
                      )

print(
    "Ran {} epochs w/ {} hidden units in RNN Cell for {} seconds.".\
    format(
        epochs,
        hidden,
        time() - start_t
        )
    )

if int(input("Process saved result?[0/1] >>> ")):
    # Prediction on test set.
    test_pred = np.copy(y_pred).reshape(-1, )
    full_test_pred = np.array([None] * len(y_data))
    full_test_pred[-len(test_pred):] = test_pred
    full_test_pred = full_test_pred.astype(np.float32)

    # Prediction on training set.
    in_range_est = np.copy(in_range_est).reshape(-1, )
    full_train_pred = np.array([None] * len(y_data))
    full_train_pred[:len(in_range_est)] = in_range_est
    full_train_pred = full_train_pred.astype(np.float32)

    # Create directory to save result.
    result_dir = gen_result_dir(profile=0)
    os.system("mkdir ./{}"\
              .format(result_dir))

    np.savetxt(
               "./" + result_dir + "full_test_pred.csv",
               full_test_pred)
    np.savetxt(
               "./" + result_dir + "full_train_pred.csv",
               full_train_pred)
    np.savetxt("./" + result_dir + "y_data.csv",
               y_data)

    # Visualization
    x_range = range(len(y_data))
    plt.plot(
             x_range,
             y_data,
             label="Actual")
    plt.plot(
             x_range,
             full_test_pred,
             label="Predicted: Out of Range")
    plt.plot(
             x_range,
             full_train_pred,
             label="Predicted: Witin training Range")
    plt.legend(
               bbox_to_anchor=(1.05, 1),
               loc=4,
               borderaxespad=0.)
    plt.title("# Hidden RNN Cell * {}, ep = {}".format(hidden, epochs))
    if int(input("Show plot? [0/1] >>> ")):
        plt.show()
    if int(input("Save plot? [0/1] >>> ")):
        plt.savefig("./" + result_dir + "plot.png")





