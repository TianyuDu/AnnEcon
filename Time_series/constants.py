


# In tuple value: 
# (0) File name of csv file.
# (1) Frequency of data.

# # For v1
# Database/source control.
# data_files = {
#     # "AMBNS": ("./data/AMBNS.csv", "M"),
#     "CPIAUCSL": ("./data/CPIAUCSL.csv", "M"),
#     # "GS1": ("./data/GS1.csv", "M"),
#     # "M2NS": ("./data/M2NS.csv", "M"),
#     # "MPRIME": ("./data/MPRIME.csv", "M"),
#     # "TB3MS": ("./data/TB3MS.csv", "M"),
#     "UNRATE": ("./data/UNRATE.csv", "M")
# }

# Model control parameters.
# parameters = {
#     "global_start": "1959-01-01",
#     "global_end": "2018-02-01",
#     "length_sample": 12,  # Length(in terms of time stamp) of RNN network.
#     "forecast_horizon": 1,  # Range of forecasting into future
#     "num_test_batch": 1,  # Number of batches that used as test set.
#     "rnn_hidden": 64,  # Number of hidden layer in RNN network.
#     "rnn_learning_rate": 0.001,  # Learning rate in RNN Training.
#     "epochs": 100 # Epochs to be ran.
# }

# For v2
parameters = {
    "global_start": "1971-01-04",
    "global_end": "2018-03-16",
    "data_freq": "B",
    "length_sample": 500,  # Length(in terms of time stamp) of RNN network.
    "forecast_horizon": 1,  # Range of forecasting into future
    "num_test_batch": 1,  # Number of batches that used as test set.
    "rnn_hidden": 256,  # Number of hidden layer in RNN network.
    "rnn_learning_rate": 0.001,  # Learning rate in RNN Training.
    "epochs": 100 # Epochs to be ran.
}

data_files = {
    "DEXCAUS": ("./data/DEXCAUS.csv", "B")
}
