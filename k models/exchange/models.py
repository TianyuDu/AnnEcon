"""
Models
"""

import numpy as np
import pandas as pd
import keras
import containers


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

    def __str__(self):
        keras.utils.print_summary(self.core)
        return f"""UnivariateLSTM model at {hex(id(self))}
        """

    def __repr__(self):
        # TODO: write repr method.
        print("")
        print(keras.utils.print_summary(self.core))
        return f"id={hex(id(self))}"

    def fit_model(self):
        pass
