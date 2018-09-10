"""
Models.
"""

import numpy as np
import pandas as pd
import keras
import containers

class BaseModel():
    def __init__(self):
        self.core = None

    def __str__(self):
        keras.utils.print_summary(self.core)
        return f"""UnivariateLSTM model at {hex(id(self))}
        """

class UnivariateLSTM():
    """
    Univariate LSTM model with customized num of layers.
    """
    def __init__(
        self, 
        container: containers.UnivariateContainer,
        config: dict={
            "batch_size": 1,
            "epoch": 10,
            "neuron": [128]}):
        self.config = config
        self.container = container
        self.core = self._construct_lstm()

    def _construct_lstm(self):
        core = keras.Sequential()
        num_lstm_lys = len(self.config["neuron"])

        batch_size = self.config["batch_size"]
        neuron_units = self.config["neuron"]

        core.add(
            keras.layers.LSTM(
                units=neuron_units[0],
                batch_input_shape=(batch_size, 1, self.container.num_fea),
                stateful=True,
                name="lstm_layer_0_input"
        ))

        # TODO: deal with multiple LSTM layer issue
        for i in range(1, num_lstm_lys):
            core.add(
                keras.layers.LSTM(
                    units=neuron_units[i],
                    stateful=True,
                    name=f"lstm_layer_{i}"
            ))
        
        core.add(keras.layers.Dense(
            units=1,
            name="dense_output"
        ))

        core.compile(
            loss="mean_squared_error",
            optimizer="adam"
        )

        return core

    def fit_model(self):
        pass


class MultivariateLSTM():
    def __init__(self, container, config=None):
        
        _, self.time_steps, self.num_fea = container.train_X.shape
        print(f"MultivariateLSTM Initialized: \
        \n\t Time Step: {self.time_steps}\
        \n\t Feature: {self.num_fea}")

        self.container = container

        self.core = self._construct_lstm()
        
    def _construct_lstm(self):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=128,
            input_shape=(self.time_steps, self.num_fea),
            return_sequences=False
        ))
        # model.add(keras.layers.LSTM(units=16))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dense(1))
        model.compile(loss="mse", optimizer="adam")

        return model
    
    def predict(self, X_feed: np.ndarray=self.container.test_X):
        y_hat = self.core.predict()
