"""
Models.
"""
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import keras
import containers

class BaseModel():
    def __init__(self):
        self.core = None
        self.container = None
        self.config = None

    def __str__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model at {hex(id(self))}
        """
    
    def __repr__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model with data container {self.container}
        """


class UnivariateLSTM(BaseModel):
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


class MultivariateLSTM(BaseModel):
    def __init__(self, container, config=None) -> None:

        _, self.time_steps, self.num_fea = container.train_X.shape
        print(f"MultivariateLSTM Initialized: \
        \n\t Time Step: {self.time_steps}\
        \n\t Feature: {self.num_fea}")

        self.config = config

        self.container = container
        self.core = self._construct_lstm(self.config)
        
    def _construct_lstm(self, config: dict) -> keras.Sequential:
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=config["nn.lstm1"],
            input_shape=(self.time_steps, self.num_fea),
            return_sequences=False
        ))
        # model.add(keras.layers.LSTM(units=config["nn.lstm2"]))
        model.add(keras.layers.Dense(units=config["nn.dense1"]))
        model.add(keras.layers.Dense(1))
        model.compile(loss="mse", optimizer="adam")

        return model
    
    def update_config(self, new_config: dict) -> None:

        self.prev_config = self.config
        self.config = new_config
        self.core = self._construct_lstm(self.config)

    def fit_model(self, epochs: int=10) -> None:

        self.hist = self.core.fit(
            self.container.train_X,
            self.container.train_y,
            epochs=epochs,
            batch_size=32 if self.config is None else self.config["batch_size"],
            validation_split=0.1 if self.config is None else self.config["validation_split"]
        )
    
    def predict(
        self, X_feed: np.ndarray) -> np.ndarray:

        y_hat = self.core.predict(X_feed, verbose=1)
        y_hat = self.container.scaler_y.inverse_transform(y_hat)
        return y_hat  # y_hat returned used to compare with self.container.*_X directly.

    def save_model(self, file_dir: str=None) -> None:
        if file_dir is None:
            file_dir = f"./saved_models/{str(datetime.now())}"
        # Save model structure to JSON
        model_json = self.core.to_json()
        with open(f"{file_dir}.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save weight to h5
        self.core.save_weights(f"{file_dir}.h5")
        print(f"Save model weights to {file_dir}.h5/json")
    
    def load_model(self, file_dir: str) -> None:
        print(f"Load model from {file_dir}")
        # construct model from json
        json_file = open("{file_dir}.json", "r")
        model_file = json_file.read()
        json_file.close()
        self.core = keras.models.model_from_json(model_file)
        # load weights from h5
        self.core.load_weights(f"{file_dir}.h5", by_name=True)
        self.core.compile(loss="mse", optimizer="adam")
        