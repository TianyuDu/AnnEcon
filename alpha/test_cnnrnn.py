"""
# TODO: add model back to model module.
"""
from typing import List, Dict
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn
from sklearn import preprocessing
# import matplotlib.pyplot as plt
from warnings import warn


from model_util import *
from data_util import *
from predefine import *
import reference_code.CRNN as CRNN


class BasicCnnRnnModel:
    """
    The ALPHA version of Stacked Rnn Model used for
    time series prediction.
    """
    init: tf.Operation  # Variable initializer.
    loss: tf.Tensor  # Loss tensor (objective function of minimization).
    loss_metric: str  # Type of loss function
    multi_cells: tf.nn.rnn_cell.MultiRNNCell  # RNN cells used to create the graph.
    optimizer: tf.train.Optimizer  # Optimizer
    output: tf.Tensor  # Output tensor
    rnn_output:  tf.Tensor  # RNN output tensor
    scaler: sklearn.preprocessing.data.StandardScaler  # The standard scaler for input data.
    stacked_outputs: tf.Tensor
    stacked_rnn_output: tf.Tensor
    X: tf.Tensor  # Input Feeding tensor.
    x_batches: np.ndarray  # Input data in batches.
    x_data: np.ndarray  # Raw input data.
    X_test: np.ndarray  # Test sample data.

    y: tf.Tensor  # Label tensor.
    y_batches: np.ndarray  # Input label data in batches.
    y_data: np.ndarray  # Raw input label data.
    Y_test: np.ndarray  # Test label data.

    def __init__(self, panel: "Panel", parameters: "ParameterControl"):
        """
        Initialize Stacked RNN Model.
        """
        print("\t@model: Initializing Stacked RNN model...")

        print("\t@model: Building scaler from panel...")
        try:
            raw = panel.df.values
            target = panel.df["UNRATE"].values
            target = target.reshape(-1, 1)
        except:
            raise PanelFailure

        assert not np.any(np.isnan(raw)), "Nan value found in panel."

        (total_steps, self.n_features) = raw.shape

        print(f"\t@model: {self.n_features} features with {total_steps}\
               time stamps (observations).")
        self.input_scaler = preprocessing.StandardScaler().fit(raw)
        self.output_scaler = preprocessing.StandardScaler().fit(target)

        print("\t@model: applying scaler...")
        raw = self.input_scaler.transform(raw)
        target = self.output_scaler.transform(target)

        print("\t@model: Preparing data...")

        self.x_data = raw[:(
            total_steps - (total_steps % parameters.num_periods)
            ), :]

        self.x_batches = self.x_data.reshape(-1, parameters.num_periods, self.n_features, 1)

        self.y_data = target[1: (
            len(target) - (len(target) % parameters.num_periods) + 1
            )]

        self.y_batches = self.y_data.reshape(-1, parameters.num_periods, 1)

        self.X_test, self.Y_test = test_data_panel(raw, target, parameters.f_horizon, parameters.num_periods)


        print("\t@model: Creating feeding nodes, dtype=float32...")
        # Input feed node.
        # self.conv_in = tf.placeholder(tf.float32, shape=self.x_batches.shape)
        self.conv_in = tf.placeholder(tf.float32, shape=[None, parameters.num_periods, self.n_features, 1])

        # Output node.
        self.y = tf.placeholder(
            tf.float32,
            [None, parameters.num_periods, parameters.nn["output"]],
            name="output_label_feed_y"
            )

        print("\t@model: Constructing CNN Layers, dtype=float32...")

        # Creating filter object
        self.filter1 = tf.Variable(
            tf.random_normal([9, 9, 1, 1]),
            name="CnnFilter1",
            dtype=tf.float32
            )

        self.filter2 = tf.Variable(
            tf.random_normal([3, 3, 1, 1]),
            name="CnnFilter2",
            dtype=tf.float32
            )

        self.conv1 = tf.nn.conv2d(
            input=self.conv_in,
            filter=self.filter1,
            strides=[1, 1, 4, 1],
            padding="SAME",
            data_format="NHWC",
            name="ConvLayer1"
            )

        self.pool1 = tf.nn.max_pool(
            value=self.conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 1, 4, 1],
            padding="SAME",
            name="MaxPoolLayer1"
            )

        self.conv2 = tf.nn.conv2d(
            input=self.pool1,
            filter=self.filter2,
            strides=[1, 1, 2, 1],
            padding="SAME",
            data_format="NHWC",
            name="ConvLayer2"
            )

        self.conv_out = tf.nn.max_pool(
            self.conv2,
            ksize=[1, 3, 3, 1],
            strides=[1, 1, 2, 1],
            padding="SAME",
            name="ConvOutputLayer"
            )

        # self.conv_out = tf.reshape(self.conv_out, shape=self.conv_out.shape[:-1])
        self.conv_out = tf.squeeze(self.conv_out, [-1])

        print("\t@model: Constructing RNN Layers, dtype=float32...")

        multi_layers = [
            tf.nn.rnn_cell.BasicRNNCell(
                num_units=parameters.nn["hidden"][0]),
            tf.nn.rnn_cell.GRUCell(
                num_units=parameters.nn["hidden"][1])
            ]

        self.multi_cells = tf.nn.rnn_cell.MultiRNNCell(multi_layers)

        self.rnn_output, self.states = tf.nn.dynamic_rnn(
            self.multi_cells,
            inputs=self.conv_out,
            dtype=tf.float32,
            parallel_iterations=512
            )

        self.stacked_rnn_output = tf.reshape(
            self.rnn_output,
            [-1, parameters.nn["hidden"][-1]],
            name="stacked_rnn_output"
            )

        self.stacked_outputs = tf.layers.dense(
            self.stacked_rnn_output,
            parameters.nn["output"],
            name="stacked_outputs"
            )

        self.outputs = tf.reshape(
            self.stacked_outputs,
            [-1, parameters.num_periods, parameters.nn["output"]]
            )

        print("\t@model: Building loss metric tensors with regularization term...")
        self.loss, self.loss_metric = gen_loss_tensor(
            self.outputs, self.y, metric="mse")

        # Add regularization term.
        self.loss = add_regularization(self.loss, parameters)
        self.loss_metric += "+reg"

        print("\t@model: Creating training operations...")
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=parameters.learning_rate)
        self.training_operation = None

        self.training_operation = self.optimizer.minimize(self.loss)

        print("\t@model: Creating initializer...")
        self.init = tf.global_variables_initializer()
        print("\t@model: Basic CNN-RNN model created.")





