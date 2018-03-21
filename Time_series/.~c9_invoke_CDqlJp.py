'''
Created: Mar 17 2018
Modified: Mar 19 2018
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg',warn=False, force=True)
import matplotlib.pyplot as plt
import random
from time import time
from data_proc import *

# random.seed(111)
# rng = pd.date_range(start="2000", periods=209, freq="M")
# ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
# ts.plot(c="b")

data = pd.read_csv("DEXCAUS.csv", sep=",")
ts = create_series(data)

# plt.show()

TS = np.array(ts) # Value of time series
num_periods = 60  # Training size 70 days before.

f_horizon = 1  # Forecasting range.

num_sample = len(TS)  # Total lenght of time series data we have.

x_data = TS[:(num_sample - num_sample % num_periods)]
# Drop out extra data that cannot be fit into batches
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1:(num_sample - num_sample % num_periods) + f_horizon]
# Y data contains one period ahead, for forcasting.
y_batches = y_data.reshape(-1, num_periods, 1)

X_test, y_test = gen_test_data(TS, f_horizon, num_periods, TS)

num_batches = x_batches.shape[0]
x_train = x_batches[: num_batches - 1, :, :]
y_train = y_batches[: num_batches - 1, :, :]

tf.reset_default_graph()

inputs = 1  # Single Vector input.
hidden = 2048
output = 1

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, inputs])

basic_cell = tf.contrib.rnn.LSTMCell(
    num_units=hidden,
    activation=tf.nn.relu,
    name="LSTMCell unit"
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
        if ep % 1 == 0:
            mse = loss.eval(
                    feed_dict={
                        X: x_train,
                        y: y_train
                    }
                )
            print("{},\tMSE:{}".format(ep, mse))
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)

print("Ran {} epochs for {} seconds.".format(epochs, time() - start_t))

if bool(input("Show forecast plot?[0/1] >>> ")):
    predict_p = np.copy(y_pred).reshape(num_periods, )
    full_p = np.zeros(len(TS), )
    full_p[:] = None
    full_p[-len(predict_p):] = predict_p
    np.savetxt("full_p.csv", full_p)
    np.savetxt("ts.csv", TS)
    plt.plot(range(len(TS)), TS)
    plt.plot(range(len(TS)), full_p)
    if bool(input("Show plot? [0/1] >>> ")):
       plt.show()
    if bool(input("Save plot? [0/1] >>> ")):
       plt.savefig("plot.png")






