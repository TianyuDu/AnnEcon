"""
Multivariate Version of prediciton.
"""
import sys
import datetime

import keras
import pandas as pd
import numpy as np
import matplotlib
# TODO: for mac OS: os.name == "posix" and sys.platform == "darwin"
# Use this identifier to automatically decide the following.
on_server = bool(int(input("Are you on a server wihtout graphic output? [0/1] >>> ")))
if on_server:   
    matplotlib.use(
        "agg",
        warn=False,
        force=True
        )
from matplotlib import pyplot as plt
import sklearn

from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import HoverTool
from bokeh.io import show, output_file

from typing import Union, List

import config
import containers
import methods
from containers import *
from methods import *
from models import *
from multi_config import *


def train_new_model():
    """
    Train a new model.
    """
    print(f"Control: Building new container from {file_dir}...")
    print(f"\tTarget is {target}")
    # Build up containers.
    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)
    print(chr(9608))

    print("Control: Building up models...")
    model = MultivariateLSTM(container, NN_config)
    print(chr(9608))

    model.fit_model(epochs=int(input("Training epochs >>> ")))
    
    save_destination = input("Folder name to save model? [Enter] Using default >>> ")
    print("Control: Saving model training result...")
    if save_destination == "":
        model.save_model()
    else:
        model.save_model(file_dir=save_destination)
    print(chr(9608))


def visualize_training_result():
    print(f"Contro;: Building up container from {file_dir}...")
    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)
    print(chr(9608))

    print("Control: Building empty model...")
    model = MultivariateLSTM(container, NN_config, create_empty=True)
    print(chr(9608))

    load_target = input("Model folder name >>> ")
    load_target = f"./saved_models/{load_target}/"
    print(f"Control: Loading model from {load_target}...")

    model.load_model(
        folder_dir=load_target
    )
    print(chr(9608))

    # Forecast testing set.
    yhat = model.predict(model.container.test_X)
    yhat = model.container.invert_difference(
        yhat, 
        range(
            model.container.num_obs-len(yhat), 
            model.container.num_obs
        ), 
        fillnone=True
    )
    # Forecast trainign set.
    train_yhat = model.predict(model.container.train_X)
    train_yhat = model.container.invert_difference(
        train_yhat, range(len(train_yhat)), fillnone=True
    )
    
    # Visualize
    plt.close()
    plt.plot(yhat, linewidth=0.6, alpha=0.6, label="Test set yhat")
    plt.plot(train_yhat, linewidth=0.6, alpha=0.6, label="Train set yhat")
    plt.plot(model.container.ground_truth_y,
            linewidth=1.2, alpha=0.3, label="actual")
    plt.legend()
    action = input("Plot result? \n\t[P] plot result. \n\t[S] save result. \n\t>>>")
    assert action.lower() in ["p", "s"], "Invalid command."

    if action.lower() == "p":
        plt.show()
    elif action.lower() == "s":
        fig_name = str(datetime.datetime.now())
        plt.savefig(f"./figure/{fig_name}.svg")
        print(f"Control: figure saved to ./figure/{fig_name}.svg")


def advanced_visualize():
    print(f"Control: Building up from container from {file_dir}")
    container = MultivariateContainer(
        file_dir,
        target,
        load_multi_ex,
        CON_config)
    print(chr(9608))

    print("Control: Building empty model...")
    model = MultivariateLSTM(container, NN_config, create_empty=True)
    print(chr(9608))

    load_target = input("Model folder name >>> ")
    load_target = f"./saved_models/{load_target}/"
    print(f"Control: Loading model from {load_target}...")

    model.load_model(
        folder_dir=load_target
    )
    print(chr(9608))

    print("Control: Building up forecasting...")
    test_yhat = model.predict(model.container.test_X)
    test_yhat = model.container.invert_difference(
        test_yhat,
        range(
            model.container.num_obs - len(test_yhat),
            model.container.num_obs
        ),
        fillnone=True
    )
    test_yhat = np.squeeze(test_yhat).astype(np.float32)

    train_yhat = model.predict(model.container.train_X)
    train_yhat = model.container.invert_difference(
        train_yhat, range(len(train_yhat)), fillnone=True
    )
    train_yhat = np.squeeze(train_yhat).astype(np.float32)

    output_file(f"{load_target}visualized.html")
    print(f"Saving plotting html file to {load_target}visualized.html...")
    pred_plot = figure(
        x_axis_label="Date", 
        y_axis_label="Value",
        x_axis_type="datetime",
        tools="lasso_select, box_select, pan")

    timeline = pd.DatetimeIndex(container.dataset.index)

    pred_plot.line(
        timeline,
        # range(len(model.container.ground_truth_y)),
        model.container.ground_truth_y,
        color="blue",
        alpha=0.7,
        legend="Actual values"
    )
    
    pred_plot.line(
        timeline,
        train_yhat,
        color="red",
        alpha=0.7,
        legend="Training set predictions"
    )

    pred_plot.line(
        timeline,
        test_yhat,
        color="green",
        alpha=0.7,
        legend="Testing set predictions"
    )

    show(pred_plot)


if __name__ == "__main__":
    print("""
    =====================================================================
    Hey, you are using the Multivariate Exchange Rate Forecasting Model
        This is a neural network developed to forecast economic indicators
        The model is based on Keras
    @Spikey
        Version. 0.0.1, Sep. 11 2018
    Important files
        Configuration file: ./multi_config.py
        Model definition file: ./models.py
    """)

    task = input("""
    What to do?
        [N] Train new model.
        [R] Restore saved model and continue training.
        [V] Visualize training result using matplotlib.
        [B] Visualize training result using bokeh.
        [Q] Quit.
    >>> """)
    assert task.lower() in ["n", "r", "v", "q", "b"], "Invalid task."
    if task.lower() == "n":
        train_new_model()
    elif task.lower() == "r":
        raise NotImplementedError
    elif task.lower() == "v":
        visualize_training_result()
    elif task.lower() == "b":
        advanced_visualize()
    elif task.lower() == "q":
        quit()



