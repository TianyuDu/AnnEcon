import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List

data_dir = "cleaned_data.csv"
original_data_dir = "Credit card data for participants.csv"


def get_header(original_data_dir: str) -> List[str]:
    with open(original_data_dir) as file:
        header_str = file.readline()
        header_list = header_str.split(",")
    return header_list

def load_data(
    data_dir: str=data_dir,
    original_data_dir: str=original_data_dir):

    data_pd = pd.read_csv(data_dir, sep=",", header=None)
    data_np = data_pd.values

    headers = get_header(original_data_dir)
    num_features = len(headers)
    print("Header of input data {}".format(headers))

    feature_dict = dict()

    for i in range(num_features - 1): # Last column is label
        header = headers[i]
        data_tensor = tf.constant(data_np[:, i])
        dataset = tf.data.Dataset.from_tensors(data_tensor)
        feature_dict[header] = data_tensor


    # label = tf.constant(data_np[:, -1].astype(np.int64))
    label = tf.constant(data_np[:, -1].astype(np.int64))
    # label = tf.data.Dataset.from_tensors(label)

    return (feature_dict, label)


age = tf.feature_column.numeric_column("Age")
# credit_limit = tf.feature_column.numeric_column("Credit_Limit")


data = pd.read_csv("play.csv", sep=",")

def position(index: str) -> int:
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


def load_data(input_data: str, shuffle=False):
    print("load_data: starting loading data...")
    try:
        data = pd.read_csv(input_data, sep=",")
    except FileNotFoundError:
        raise Warning("load_data(): Target input_data not found")
    data_np = data.values
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

classifier = tf.estimator.DNNClassifier(
    feature_columns=generate_feature_columns(),
    hidden_units=[10, 25, 10],
    n_classes=2, # We predict binary
    model_dir="./model/temp/"
    )

train_file = "play.csv"
test_file = "play_test.csv"


classifier.train(input_fn=load_data(train_file, shuffle=True), steps=20)

accuracy_score = classifier.evaluate(
    input_fn=load_data(test_file)
    )
print(accuracy_score)




