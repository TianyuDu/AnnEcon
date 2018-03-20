import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List


def get_header(original_data_dir: str) -> List[str]:
    with open(original_data_dir) as file:
        header_str = file.readline()
        header_list = header_str.split(",")
    return header_list

data = pd.read_csv("play.csv", sep=",")

def position(index: str) -> int:
    """
    Helper function.
    """
    index = index.upper()
    return int(ord(index) - 65)

def generate_feature_columns(config=0):
    fc = list()

    fc.append(
        tf.feature_column.numeric_column("credit_limit")
        )


    fc.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "sex",
                vocabulary_list=(1, 2)
            ), 1
        )
    )

    fc.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "education",
                vocabulary_list=(1, 2, 3, 4)
                ), 1
            )
        )


    fc.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "marital_status",
                vocabulary_list=(1, 2, 3)
                ), 1
            )
        )

    age_boundary = [int(i) for i in np.linspace(14, 101, 11)]
    fc.append(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("age"),
            boundaries=age_boundary
            )
    )

    def rep_operation(month: str):
        return tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column(month + "_rep"),
            boundaries=[0, 7, 13]
            )

    def ppp_sta_operation(month: str):
        return (
            tf.feature_column.numeric_column(
                month + "_ppp"
                ),
            tf.feature_column.numeric_column(
                month + "_sta"
                )
            )

    for month in ["jan", "feb", "mar", "apr", "may", "jun"]:
        fc.append(rep_operation(month))
        for i in ppp_sta_operation(month):
            fc.append(i)

    return fc

label = np.copy(data.values[:, -1])
label = label.astype(np.uint32)
label = label.astype(np.int64)


def load_data(input_data: str, shuffle=True):
    print("load_data: starting loading data...")
    try:
        data = pd.read_csv(input_data, sep=",")
    except FileNotFoundError:
        raise Warning("load_data(): Target input_data not found")
    data_np = data.values[:]
    print("Creating input wrapper...")
    input_wrapper = tf.estimator.inputs.numpy_input_fn(
        x={
            "credit_limit": data_np[:, position("A")], # Credit
            "sex": data_np[:, position("B")].astype(np.int64),
            "education": data_np[:, position("C")].astype(np.int64),
            "marital_status": data_np[:, position("D")].astype(np.int64),
            "age": data_np[:, position("E")],

            "jan_rep": data_np[:, position("F")].astype(np.int64),
            "jan_ppp": data_np[:, position("G")],
            "jan_sta": data_np[:, position("H")],

            "feb_rep": data_np[:, position("I")].astype(np.int64),
            "feb_ppp": data_np[:, position("J")],
            "feb_sta": data_np[:, position("K")],

            "mar_rep": data_np[:, position("L")].astype(np.int64),
            "mar_ppp": data_np[:, position("M")],
            "mar_sta": data_np[:, position("N")],

            "apr_rep": data_np[:, position("O")].astype(np.int64),
            "apr_ppp": data_np[:, position("P")],
            "apr_sta": data_np[:, position("Q")],

            "may_rep": data_np[:, position("R")].astype(np.int64),
            "may_ppp": data_np[:, position("S")],
            "may_sta": data_np[:, position("T")],

            "jun_rep": data_np[:, position("U")].astype(np.int64),
            "jun_ppp": data_np[:, position("V")],
            "jun_sta": data_np[:, position("W")]
            },
        y=data_np[:,-1].astype(np.int64),
        num_epochs=1,
        shuffle=shuffle)
    print("Finished.")
    return input_wrapper

estimator = tf.estimator.DNNClassifier(
    feature_columns=generate_feature_columns(),
    hidden_units=[50, 30, 20],
    optimizer=tf.train.AdamOptimizer(
        learning_rate=0.0001
        ),
    n_classes=2, # We predict binary
    model_dir="./model_cache/"
    )

train_file = "dataf1.csv"
test_file = "play_test.csv"


estimator.train(input_fn=load_data(train_file, shuffle=True), steps=1000)

def get_performance(estimator, data_file: str, metric: str, shuffle_data=True):
    perf_dict = estimator.evaluate(
        input_fn=load_data(data_file, shuffle=shuffle_data)
        )
    try:
        target_score = perf_dict[metric]
    except KeyError:
        print("Performance metric not found, empty string will be returned.")
        target_score = ""
    return target_score
    
print("Accuracy on train file: {}".format(
    get_performance(estimator, train_file, "accuracy")
    ))    
    
print("Accuracy on test file: {}".format(
    get_performance(estimator, test_file, "accuracy")
    ))

