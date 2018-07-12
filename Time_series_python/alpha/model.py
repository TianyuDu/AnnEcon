"""
Meta methods and classes for model to use.
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
from predfine import *


class StackedRnnModel:
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

    def __init__(self, series: np.ndarray, parameters: "ParameterControl"):
        """
        Initialize Stacked RNN Model.
        """
        print("\t@model: Initializing Stacked RNN model...")

        print("\t@model: Building scaler for series...")
        self.scaler = preprocessing.StandardScaler().fit(series)

        print("\t@model: Preparing data...")
        self.x_data = series[:(
            len(series) - (len(series) % parameters.num_periods)
            )]

        self.x_batches = self.x_data.reshape(-1, parameters.num_periods, 1)

        self.y_data = series[1: (
            len(series) - (len(series) % parameters.num_periods) + 1
            )]

        self.y_batches = self.y_data.reshape(-1, parameters.num_periods, 1)

        self.X_test, self.Y_test = model_util.test_data(series, parameters.f_horizon, parameters.num_periods)

        print("\t@model: Creating feeding nodes, dtype = float32...")
        # Input feed node.
        self.X = tf.placeholder(
            tf.float32,
            [None, parameters.num_periods, parameters.nn["inputs"]],
            name="input_label_feed_X"
            )
        # Output node.
        self.y = tf.placeholder(
            tf.float32,
            [None, parameters.num_periods, parameters.nn["output"]],
            name="output_label_feed_y"
            )

        multi_layers = [
            tf.nn.rnn_cell.BasicRNNCell(
                num_units=parameters.nn["hidden"][0]),
            tf.nn.rnn_cell.GRUCell(
                num_units=parameters.nn["hidden"][1])
            ]

        self.multi_cells = tf.nn.rnn_cell.MultiRNNCell(multi_layers)

        self.rnn_output, self.states = tf.nn.dynamic_rnn(
            self.multi_cells,
            inputs=self.X,
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
        self.loss, self.loss_metric = model_util.gen_loss_tensor(
            self.outputs, self.y, metric="mse")

        # Add regularization term.
        self.loss = model_util.add_regularization(self.loss, parameters)
        self.loss_metric += "+reg"

        print("\t@model: Creating training operations...")
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=parameters.learning_rate)
        self.training_operation = None

        self.training_operation = self.optimizer.minimize(self.loss)

        print("\t@model: Creating initializer...")
        self.init = tf.global_variables_initializer()


class ParameterControl:
    """
    This logger is used to collect all user-specified
    parameters in training.
    """
    epochs: int
    f_horizon: int  # Forecasting horizon.
    learning_rate: float  # Starting learning rate for training.

    nn: dict  # Dictionary containing parameter setting for neural network.
    # nn["inputs"]: int  # RNN input layer size.
    # nn["hiddens"]: List[int]  # RNN hidden unit sizes (mutliple layers in stack)
    # nn["output"]: int  # RNN output layer size.
    # nn["reg_para"]: float  # RNN regularization parameter.

    num_periods: int  # Number of periods looking back in training.
    on_server: bool  # If on AWS server.

    def __init__(self):
        self.on_server = int(
            input("@ParameterControl: Use graphic configuration for server? [0/1]: "))

        if int(input("@ParameterControl: Use default parameters? [0/1]: ")):
            # Use default parameters.
            self.load_default_parameters()
        else:
            # Ask user for parameters.
            self.get_parameters()

        # Load neural network settings.
        self.nn = self.load_nn_parameters()
        print("@ParameterControl: ParameterControl built successfully.")

    def load_default_parameters(self, config=0) -> None:
        """ Loading the default parameter for the parameter control pack
        """
        print("\t@ParameterControl: Loading default model parameters for test purpose...")
        print(f"\t\tConfiguration={config}")
        if config == 0:
            self.num_periods = 24
            self.f_horizon = 1
            self.learning_rate = 0.001
            self.epochs = 1500

    def get_parameters(self):
        """
        Promot users to specify the model parameters.
        """
        print("\t@ParameterControl: Please specify training parameters")

        self.num_periods = int(
            input("\t\tNumber of periods looking back in traning: ")
            )
        assert (self.num_periods > 0), "Number of periods looking back should be positive."

        self.f_horizon = int(
            input("\t\tNumer of periods to forecast: ")
            )
        assert (self.f_horizon > 0), "Number of forecasting periods should be positive."

        self.learning_rate = float(
            input("\t\tLearning rate: ")
            )
        assert (self.learning_rate > 0), "Learning rate should be positive."
        if self.learning_rate > 0.5:
            warn("Learning rate requested is greater than 0.5.")

        self.epochs = int(
            input("\t\tTraining epochs: ")
            )
        assert (self.epochs > 0), "Training epochs should be positive."
        if self.epochs < 100:
            warn("Training epochs is smaller than 100.")

    def load_nn_parameters(self) -> dict:
        """
        This method create the parameter pack for neural networks and return a dict containing all those parameters.
        """
        nn_para = dict()
        nn_para["inputs"] = 1
        nn_para["hidden"] = [128, 128]
        nn_para["output"] = 1
        nn_para["reg_para"] = 0.005
        return nn_para


class RnnFeatureExtractionRnn:
    """
    Reference:
    Title: Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data
    Keren 20 Jul. 2017

    Mechanism
    Extract features from a multivariate time series dataset (panel) and pass the result into a RNN.
    """
    def __init__(self):
        # TODO Jul 12 2018 Stop here

