"""
Alpha version 2
"""
# Loading Packages
from model_util import *
from models import *
para = ParameterControl()
print("Loading Packages...")
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
import matplotlib
if para.on_server:
    matplotlib.use(
        "agg",
        warn=False,
        force=True
        )  # If on a server, change matplotlib settings.
import matplotlib.pyplot as plt
print("Done.")
from data_util import *


def main(parameters: "ParameterControl"):
    """
    Main operation.
    """
    print("@main: version alpha 2: single time series prediction, stacked.")
    # Prepare data.
    print("@main: preparing data...")
    print("\t@main: CPIAUCSL data will be loaded, as benchmark test data.")
    ts, ts_array = load_data("./data/CPIAUCSL.csv", "local")

    # Reset tensorflow
    print("@main: resetting default model ")
    tf.reset_default_graph()
    # Create model.
    print("@main: building model...")
    model = BasicCnnRnnModel(p, parameters=parameters)

    with tf.Session() as sess:
        print("@main: Starting session...")
        print("@main: Creating saver...")
        saver = tf.train.Saver()
        print("@main: Starting writer...")
        writer = tf.summary.FileWriter("output", sess.graph)
        tf.summary.histogram("loss", model.loss)
        tf.summary.histogram("outputs", model.outputs)
        print("@main: Initializing tensors...")
        model.init.run()
        print("@main: Training...")
        begin_time = datetime.now()

        loss_record = [1]  # Record loss history.

        for ep in range(para.epochs + 1):  # + 1 so that the training loss on t=epochs will be printed.
            sess.run(
                model.training_operation,
                feed_dict={
                    model.conv_in: model.X_train,
                    model.y: model.Y_train})

            if ep % 100 == 0:
                quantified_loss = model.loss.eval(
                    feed_dict={model.conv_in: model.X_train,
                               model.y: model.Y_train})
                loss_record.append(quantified_loss)
                print(ep,
                      f"\t{model.loss_metric}: {quantified_loss}")
                print(
                    f"\tLoss Improvement:\
                     {-1 * ((loss_record[-1] - loss_record[-2]) / loss_record[-2] * 100): .6}%")

        now_str = datetime.strftime(datetime.now(), "%Y_%m_%d_%s")
        saver.save(sess, f"./saved/{now_str}")
        # Create training set prediction,
        y_pred_train = sess.run(
            model.outputs, feed_dict={model.conv_in: model.X_train})
        y_pred_test = sess.run(
            model.outputs, feed_dict={model.conv_in: model.X_test})
        writer.close()

        print(f"Training finished, time taken {(datetime.now() - begin_time)}")

    # Transform back
    y_data = model.output_scaler.inverse_transform(model.y_data)
    y_pred_train = model.output_scaler.inverse_transform(y_pred_train)
    y_pred_train = y_pred_train.reshape(-1, 1)  # Expand the stacked inputs.
    y_pred_test = model.output_scaler.inverse_transform(y_pred_test)

    visualize(
        y_data,
        y_pred_train,
        y_pred_test,
        parameters.epochs,
        on_server=parameters.on_server)


if __name__ == "__main__":
    main(para)
