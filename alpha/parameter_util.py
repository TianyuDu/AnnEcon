"""
Parameter Utility.
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
from predefine import *



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
            self.num_periods = 48
            self.f_horizon = 1
            self.learning_rate = 0.001
            self.epochs = int(input("Training epochs: "))

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
        nn_para["hidden"] = [256, 128]
        nn_para["output"] = 1
        nn_para["reg_para"] = 0.005
        return nn_para
