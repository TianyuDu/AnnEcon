neural_network_config = {
    "batch_size": 1,
    "epoch": 10,
    "neuron": 128,
}

data_proc_config = {
    "method": "diff",
    "diff.lag": 1,
    "diff.order": 1,
    "test_ratio": 0.2,
    "lag_for_sup": 3,
    "target_idx": 0
}