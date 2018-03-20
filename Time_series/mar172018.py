'''
Created: Mar 17 2018
Modified: Mar 19 2018
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from time import time

# random.seed(111)
# rng = pd.date_range(start="2000", periods=209, freq="M")
# ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
# ts.plot(c="b")

data = pd.read_csv("DEXCAUS.csv", sep=",")
data_value = data.values[:, 1]
data_value[data_value == "."] = "0.0"
ts = pd.Series(data=data_value.astype(np.float32), index=data.values[:, 0])

# plt.show()

TS = np.array(ts)
num_periods = 60  # Training size.

f_horizon = 1  # Forecasting range.

x_data = TS[:(len(TS) - len(TS) % num_periods)]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1:(len(TS) - len(TS) % num_periods) + f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)

def gen_test_data(series, forecast, num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = TS[-(num_periods):].reshape(-1, 20, 1)
    return testX, testY

X_test, y_test = gen_test_data(TS, f_horizon, num_periods)

tf.reset_default_graph()

inputs = 1  # Single Vector input.
hidden = 2048
output = 1

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(
    num_units=hidden,
    activation=tf.nn.relu
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


epochs = 200

start_t = time()

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(
            training_op,
                feed_dict={
                    X: x_batches,
                    y: y_batches
                }
            )
        if ep % 10 == 0:
            mse = loss.eval(
                    feed_dict={
                        X: x_batches,
                        y: y_batches
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
    plt.plot(range(len(TS)), TS)
    plt.plot(range(len(TS)), full_p)
    if bool(input("Show plot? [0/1] >>> ")):
        plt.show()
    else:
        plt.savefig("plot.png")






