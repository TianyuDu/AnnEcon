"""
Models.
"""
import datetime
import numpy as np
import pandas as pd
import keras
import containers
import os

class BaseModel():
    def __init__(self):
        self.core = None
        self.container = None
        self.config = None
        self._gen_file_name()
    
    def _gen_file_name(self):
        """
        Generate the directory name to save all relevant files about
        Graphic representation of model,
        Model structure()
        """
        now = datetime.datetime.now()
        self.file_name = now.strftime("%Y%h%d_%H_%M_%s")

    def __str__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model at {hex(id(self))}
        """
    
    def __repr__(self):
        keras.utils.print_summary(self.core)
        return f"""{str(type(self))} model with data container {self.container}
        """

# TODO: Fix the univariate LSTM
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
        """
        Initialization method.
        """
        _, self.time_steps, self.num_fea = container.train_X.shape
        print(f"MultivariateLSTM Initialized: \
        \n\tTime Step: {self.time_steps}\
        \n\tFeature: {self.num_fea}")

        self.config = config

        self.container = container
        self.core = self._construct_lstm(self.config)
        self._gen_file_name()
        print(
            f"\tMultivariateLSTM: Current model will be save to ./saved_models/f{MultivariateLSTM}/")
        
    def _construct_lstm(self, config: dict, verbose: bool=True) -> keras.Sequential:
        """
        Construct the Stacked lstm model, 
        Note: Modify this method to change model configurations.
        # TODO: Add arbitray layer support. 
        """
        print("MultivariateLSTM: Generating LSTM model")
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=config["nn.lstm1"],
            input_shape=(self.time_steps, self.num_fea),
            return_sequences=True
        ))
        model.add(keras.layers.LSTM(units=config["nn.lstm2"]))
        model.add(keras.layers.Dense(units=config["nn.dense1"]))
        model.add(keras.layers.Dense(1))
        model.compile(loss="mse", optimizer="adam")

        if verbose:
            print("\tMultivariateLSTM: LSTM model constructed with configuration: ")
            keras.utils.print_summary(model)
        return model
    
    def update_config(self, new_config: dict) -> None:
        """
        Update the neural network configuration, and re-construct, re-compile the core.
        """
        # TODO: add check configuration method here.
        print("MultivariateLSTM: Updating neural network configuration...")
        self.prev_config = self.config
        self.config = new_config
        self.core = self._construct_lstm(self.config, verbose=False)
        print("\tDone.")

    def fit_model(self, epochs: int=10) -> None:
        start_time = datetime.datetime.now()
        print("MultivariateLSTM: Start fitting.")
        self.hist = self.core.fit(
            self.container.train_X,
            self.container.train_y,
            epochs=epochs,
            batch_size=32 if self.config is None else self.config["batch_size"],
            validation_split=0.1 if self.config is None else self.config["validation_split"]
        )
        finish_time = datetime.datetime.now()
        time_taken = finish_time - start_time
        print(f"\tFitting finished, {epochs} epochs for {str(time_taken)}")
    
    def predict(
        self, X_feed: np.ndarray) -> np.ndarray:

        y_hat = self.core.predict(X_feed, verbose=1)
        y_hat = self.container.scaler_y.inverse_transform(y_hat)
        return y_hat  # y_hat returned used to compare with self.container.*_X directly.

    def save_model(self, file_dir: str=None) -> None:
        if file_dir is None:
            # If no file directory specified, use the default one.
            file_dir = self.file_name

        # Try to create record folder.
        try:
            folder = f"./saved_models/{file_dir}/"
            os.system(f"mkdir {folder}")
            print(f"Experiment record directory created: {folder}")
        except:
            print("Current directory: ")
            _ = os.system("pwd")
            raise FileNotFoundError(
                "Failed to create directory, please create directory ./saved_models/")
        
        # Save model structure to JSON
        print("Saving model structure...")
        model_json = self.core.to_json()
        with open(f"{folder}model_structure.json", "w") as json_file:
            json_file.write(model_json)
        print("Done.")

        # Save model weight to h5
        print("Saving model weights...")
        self.core.save_weights(f"{file_dir}model_weights.h5")
        print("Done")
    
    def load_model(self, file_dir: str) -> None:
        """
        """
        if not file_dir.endwith("/"):  
            # Assert the correct format, file_dir should be 
            file_dir += "/"

        print(f"Load model from folder {file_dir}")

        # construct model from json
        print("Reconstruct model from Json file...")
        try:
            json_file = open(f"{file_dir}model_structure.json", "r")
        except FileNotFoundError:
            raise Warning(
                f"Json file not found. Expected: {file_dir}model_structure.json"
            )

        model_file = json_file.read()
        json_file.close()
        self.core = keras.models.model_from_json(model_file)
        print("Done.")

        # load weights from h5
        print("Loading model weights...")
        try:
            self.core.load_weights(f"{file_dir}model_weights.h5", by_name=True)
        except FileNotFoundError:
            raise Warning(
                f"h5 file not found. Expected: {file_dir}model_weights.h5"
            )
        print("Done.")
        self.core.compile(loss="mse", optimizer="adam")

    def summarize_training(self):
        """
        Summarize training result to string file.
        - Loss
        - Epochs
        - Time taken
        """
        raise NotImplementedError
    
    def visualize_training(self):
        """
        Visualize the training result:
        - Plot training set loss and validation set loss.
        """
        raise NotImplementedError


class MultivariateCnnLSTM(BaseModel):
    def __init__(self):
        pass
