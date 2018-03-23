"""
Created: Mar 21 2018
Join Several Economic Indicators to assist the prediction.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
if int(input("On server? [0/1] >>> ")):
    matplotlib.use('agg',warn=False, force=True)
else:
    import progressbar

import matplotlib.pyplot as plt
import random
from time import time
from data_proc import *

# random.seed(111)
# rng = pd.date_range(start="2000", periods=209, freq="M")
# ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
# ts.plot(c="b")

def import_data():
    cpi_data = pd.read_csv("CPIAUCSL.csv", sep=",")
    mprime_data = pd.read_csv("MPRIME.csv", sep=",")
    gs1_data = pd.read_csv("GS1.csv", sep=",")

    # print("Improted Data, from {} to {}".format(
    #         cpi_data["DATE"][0], cpi_data["DATE"][-1]
    #         )
    #     )

    (cpi_ts,
        mprime_ts,
        gs1_ts) = (
            create_series(cpi_data),
            create_series(mprime_data),
            create_series(gs1_data))
    assert len(cpi_ts) == len(mprime_ts) == len(gs1_ts)
    combined = pd.concat([cpi_ts, mprime_ts, gs1_ts], axis=1)
    combined.rename(
        columns={0: "CPI", 1: "MPrime", 2: "GS1"},
        inplace=True
        )
    return combined

ts = import_data()
TS = np.array(ts) # Value of time series
num_periods = 12

f_horizon = 1  # Forecasting range.

num_sample = len(TS)  # Total lenght of time series data we have.

x_data = TS[:(num_sample - num_sample % num_periods)]
# Drop out extra data that cannot be fit into batches
x_batches = x_data.reshape(-1, num_periods, 3)

y_data = TS[1:(num_sample - num_sample % num_periods) + f_horizon, 0]
# Y data contains one period ahead, for forcasting.
y_batches = y_data.reshape(-1, num_periods, 1)

X_test, y_test = gen_test_data(TS, f_horizon, num_periods)

num_batches = x_batches.shape[0]
x_train = x_batches[: num_batches - 1, :, :]
y_train = y_batches[: num_batches - 1, :, :]

tf.reset_default_graph()

inputs = 3  # Single Vector input.
hidden = int(input("Num. Hidden Layers in Cell >>> "))
output = 1

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.LSTMCell(
    num_units=hidden,
    activation=tf.nn.relu,
    cell_clip=100000.0,
    forget_bias=0.5,
    name="LSTM_cell"
    )

rnn_output, states = tf.nn.dynamic_rnn(
    basic_cell,
    X,
    dtype=tf.float32
    )

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = int(input("Epochs to run >>> "))

start_t = time()

# widgets = [
#         progressbar.Percentage(),
#         progressbar.Bar(),
#         progressbar.DynamicMessage('mes')
#         ]

# bar = progressbar.ProgressBar(
#     max_value=epochs,
#     widgets=widgets
#     )

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(
            training_op,
                feed_dict={
                    X: x_train,
                    y: y_train
                }
            )
        if ep % 10 == 0:
            mse = loss.eval(
                    feed_dict={
                        X: x_train,
                        y: y_train
                    }
                )
            print("{},\tMSE:{}".format(ep, mse))
            # bar.update(ep, mse=mse)
        # else:
        #     bar.update(ep)
    y_pred_train = sess.run(outputs, feed_dict={X: x_train})
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)

print(
    "Ran {} epochs w/ {} hidden units in RNN Cell for {} seconds.".format(
        epochs,
        hidden,
        time() - start_t)
    )

if int(input("Show forecast plot?[0/1] >>> ")):
    predict_p = np.copy(y_pred).reshape(num_periods, )
    full_p = np.zeros(len(TS), )
    full_p[:] = None
    full_p[-len(predict_p):] = predict_p
    # in_p = np.zeros(len(TS), )  # Within range prediction
    # in_p[:] = None
    # in_p[:len(y_pred_train)] = y_pred_train
    np.savetxt("full_p.csv", full_p)
    np.savetxt("ts.csv", TS)
    # np.savetxt("in_p.csv", in_p)

    plt.plot(range(len(TS)), TS[:, 0], label="Actual")
    # plt.plot(range(len(TS)), in_p, label="Predicted: Within Range")
    plt.plot(range(len(TS)), full_p, label="Predicted: Out of Range")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0.)
    plt.title("# Hidden RNN Cell * {}, ep = {}".format(hidden, epochs))
    if int(input("Show plot? [0/1] >>> ")):
        plt.show()
    if int(input("Save plot? [0/1] >>> ")):
        plt.savefig("plot.png")






