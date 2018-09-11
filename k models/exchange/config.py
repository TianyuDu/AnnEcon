import pandas as pd
import numpy as np

"""
Configuration file for univariate time series model.
"""
# Configuration for neural network
neural_network_config = {
    "batch_size": 1,  # Batch size for training
    "epoch": 10,  # Training epochs
    "neuron": [128],  # Unit of neurons in layers in LSTM model.
}

# Configuration for data processing
# Configuration dictionary to create DataContainer Object
data_proc_config = {
    # Method to remove non-stationarity of time series
    # By default, create stationary time series using first
    # differencing.
    "method": "diff",  # {str}
    # Avaiable method:
    # "diff": remove non-stationarity by differncing.
    "diff.lag": 1,  # {int > 0}
    # Lag of differencing,
    # Notice: If set to zero, the 0 series will be returned (x[i] - x[i - 0] == 0)
    "diff.order": 1,  # {int >= 0}
    # Order of differencing,
    # Notice: 0 order differencing will return the original series.
    # For k order differncing, the generated series will have length N - k
    "test_ratio": 0.2,  # {0 <= float < 1}
    # Ratio of test set to evaluate neural network.
    "lag_for_sup": 3,  # {int > 0}
    # Total number of lag values used to generate
    # supervised learning problem (SLP).
    # Using m then the i-1 to i-m total m lagged variables
    # will be used to predict the i-index variable in the SLP.
}


# ================ Configuration for multivariate model.
def load_multi_ex(file_dir: str) -> pd.DataFrame:
    dataset = pd.read_csv(file_dir, delimiter="\t", index_col=0)
    # Cleaning Data
    dataset.dropna(how="any", axis=0, inplace=True)
    dataset.replace(to_replace=".", value=np.NaN, inplace=True)
    dataset.fillna(method="ffill", inplace=True)
    dataset = dataset.astype(np.float32)
    # DEXVZUS behaved abnomally
    dataset.drop(columns=["DEXVZUS"], inplace=True)
    return dataset


CON_config = {
    "max_lag": 3,
    "train_ratio": 0.9,
    "time_steps": 14
}

NN_config = {
    "batch_size": 32,
    "validation_split": 0.1,
    "nn.lstm1": 256,
    "nn.lstm2": 128,
    "nn.dense1": 64
}
