import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List

print("Starting model...")

data_dir = "cleaned_data.csv"
original_data_dir = "Credit card data for participants.csv"

pred_file = "dataedu2.csv"
test_file = train_file = "dataedu1.csv"

test_file = train_file = input("File name of training file (*.csv) you would like to use: ")
pred_file = input("File name of prediction file (*.csv) you would like to use: ")


def get_header(original_data_dir: str) -> List[str]:
    with open(original_data_dir) as file:
        header_str = file.readline()
        header_list = header_str.split(",")
    return header_list

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
            boundaries=[0, 3, 6, 9, 13]
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

def load_predict_data(input_data: str, shuffle=False):
    print("load_data: starting loading data...")
    try:
        data = pd.read_csv(input_data, sep=",")
    except FileNotFoundError:
        raise Warning("load_data(): Target input_data not found")
    data_np = data.values
    print("Creating input wrapper...")
    
    def mean_normalize(x):
        return (x - np.mean(x)) / np.sum(x)
        
    input_wrapper = tf.estimator.inputs.numpy_input_fn(
        x={
            "credit_limit": mean_normalize(data_np[:, position("A")]), # Credit
            "sex": data_np[:, position("B")].astype(np.int64),
            "education": data_np[:, position("C")].astype(np.int64),
            "marital_status": data_np[:, position("D")].astype(np.int64),
            "age": data_np[:, position("E")],

            "jan_rep": data_np[:, position("F")].astype(np.int64),
            "jan_ppp": mean_normalize(data_np[:, position("G")]),
            "jan_sta": mean_normalize(data_np[:, position("H")]),

            "feb_rep": data_np[:, position("I")].astype(np.int64),
            "feb_ppp": mean_normalize(data_np[:, position("J")]),
            "feb_sta": mean_normalize(data_np[:, position("K")]),

            "mar_rep": data_np[:, position("L")].astype(np.int64),
            "mar_ppp": mean_normalize(data_np[:, position("M")]),
            "mar_sta": mean_normalize(data_np[:, position("N")]),

            "apr_rep": data_np[:, position("O")].astype(np.int64),
            "apr_ppp": mean_normalize(data_np[:, position("P")]),
            "apr_sta": mean_normalize(data_np[:, position("Q")]),

            "may_rep": data_np[:, position("R")].astype(np.int64),
            "may_ppp": mean_normalize(data_np[:, position("S")]),
            "may_sta": mean_normalize(data_np[:, position("T")]),

            "jun_rep": data_np[:, position("U")].astype(np.int64),
            "jun_ppp": mean_normalize(data_np[:, position("V")]),
            "jun_sta": mean_normalize(data_np[:, position("W")])
            },
        num_epochs=1,
        shuffle=shuffle)
    print("Finished.")
    return input_wrapper

def load_data(input_data: str, shuffle=False):
    print("load_data: starting loading data...")
    try:
        data = pd.read_csv(input_data, sep=",")
    except FileNotFoundError:
        raise Warning("load_data(): Target input_data not found")
    data_np = data.values
    print("Creating input wrapper...")
    
    def mean_normalize(x):
        return x
        return np.copy((x - np.mean(x)) / np.sum(x))
        
    input_wrapper = tf.estimator.inputs.numpy_input_fn(
        x={
            "credit_limit": mean_normalize(data_np[:, position("A")]),
            "sex": data_np[:, position("B")].astype(np.int64),
            "education": data_np[:, position("C")].astype(np.int64),
            "marital_status": data_np[:, position("D")].astype(np.int64),
            "age": data_np[:, position("E")],

            "jan_rep": data_np[:, position("F")].astype(np.int64),
            "jan_ppp": mean_normalize(data_np[:, position("G")]),
            "jan_sta": mean_normalize(data_np[:, position("H")]),

            "feb_rep": data_np[:, position("I")].astype(np.int64),
            "feb_ppp": mean_normalize(data_np[:, position("J")]),
            "feb_sta": mean_normalize(data_np[:, position("K")]),

            "mar_rep": data_np[:, position("L")].astype(np.int64),
            "mar_ppp": mean_normalize(data_np[:, position("M")]),
            "mar_sta": mean_normalize(data_np[:, position("N")]),

            "apr_rep": data_np[:, position("O")].astype(np.int64),
            "apr_ppp": mean_normalize(data_np[:, position("P")]),
            "apr_sta": mean_normalize(data_np[:, position("Q")]),

            "may_rep": data_np[:, position("R")].astype(np.int64),
            "may_ppp": mean_normalize(data_np[:, position("S")]),
            "may_sta": mean_normalize(data_np[:, position("T")]),

            "jun_rep": data_np[:, position("U")].astype(np.int64),
            "jun_ppp": mean_normalize(data_np[:, position("V")]),
            "jun_sta": mean_normalize(data_np[:, position("W")])
            },
        y=data_np[:,-1].astype(np.int64),
        num_epochs=1,
        shuffle=shuffle)
    print("Finished.")
    return input_wrapper

def create_test_files(input_data: str):
    data_pd = pd.read_csv(input_data, sep=",")
    data_np = data_pd.values
    num_sample = data_np.shape[0]
    test_data = data_np[
        np.random.randint(0, num_sample, int(0.3 * num_sample)),
        :]
    test_sample_file = "temp_file.csv"
    np.savetxt(test_sample_file, test_data)
    
    return test_sample_file

classifier = tf.estimator.DNNClassifier(
    feature_columns=generate_feature_columns(),
    hidden_units=[1024, 512, 256, 128, 64, 32],
    optimizer=tf.train.AdamOptimizer(
        learning_rate=0.0001
        ),
    activation_fn=tf.nn.relu,
    n_classes=2, # We predict binary
    model_dir="./model/temp/"
    )

classifier.train(input_fn=load_data(train_file, shuffle=True), steps=2000)

test_sample_file = create_test_files(train_file)
test_sample_file = test_file
accuracy_score = classifier.evaluate(
    input_fn=load_data(test_sample_file)
    )

print("Accuracy on test file {}%".format(100 * accuracy_score["accuracy"]))

pred_result = classifier.predict(load_predict_data(pred_file))

binary_result = [np.argmax(i["probabilities"]) for i in pred_result]


print("Accuracy on test file {}%".format(100 * accuracy_score["accuracy"]))

output_file_name = "output_{}".format(pred_file)
print("Prediction is saved to {}".format(output_file_name))
np.savetxt(output_file_name, binary_result)
