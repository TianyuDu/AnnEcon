"""
Prediction model for Wiki Stock prices
"""
# Loading Packages
from model_util import *
from models.wiki_price_cnnrnn import *
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


def main(parameters: "ParameterControl"):
    """
    Main operation.
    """
    print("WIKI_PRICE Prediction model loaded.")
    # Prepare data.
    print("@main: preparing data...")

    data_url = "https://s3.us-east-2.amazonaws.com/spikey/data/WIKI_PRICES.csv"

    print(f"@main: downloading data from \n\t{data_url}")

    df = pd.read_csv(
        data_url,
        index_col=1,
        header=0)

    # Reset tensorflow
    print("@main: resetting default model ")
    tf.reset_default_graph()
    # Create model.
    print("@main: building model...")

    target_str = "open"
    print(f"@main: target set to {target_str}")
    model = BasicCnnRnnModel(df, target_str, parameters=parameters)

    with tf.Session() as sess:
        print("@main: Starting session...")
        print("@main: Starting writer...")
        writer = tf.summary.FileWriter("output", sess.graph)
        tf.summary.histogram("loss", model.loss)
        tf.summary.histogram("outputs", model.outputs)
        print("@main: Initializing tensors...")
        model.init.run()
        print("@main: Training...")
        begin_time = datetime.now()

        loss_record = [1]  # Record loss history.

        for ep in range(para.epochs):
            sess.run(
                model.training_operation,
                feed_dict={
                    model.X: model.x_train,
                    model.y: model.y_train})

            if ep % 100 == 0:
                quantified_loss = model.loss.eval(
                    feed_dict={model.X: model.x_train,
                               model.y: model.y_train})
                loss_record.append(quantified_loss)
                print(ep,
                      f"\t{model.loss_metric}: {quantified_loss}")
                print(
                    f"\tLoss Improvement:\
                     {-1 * ((loss_record[-1] - loss_record[-2]) / loss_record[-2] * 100): .6}%")

        # Create training set prediction,
        y_pred_train = sess.run(
            model.outputs, feed_dict={model.X: model.x_train})
        y_pred_test = sess.run(
            model.outputs, feed_dict={model.X: model.X_test})
        writer.close()

        print(f"Training finished, time taken {(datetime.now() - begin_time)}")

    # Transform back
    y_data = model.scaler.inverse_transform(model.y_data)
    y_pred_train = model.scaler.inverse_transform(y_pred_train)
    y_pred_train = y_pred_train.reshape(-1, 1)  # Expand the stacked inputs.
    y_pred_test = model.scaler.inverse_transform(y_pred_test)

    visualize(
        y_data,
        y_pred_train,
        y_pred_test,
        on_server=parameters.on_server)


if __name__ == "__main__":
    main(para)
