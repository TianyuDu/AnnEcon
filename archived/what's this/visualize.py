"""
Contain visualization tools for model.
Created Mar 30 2018
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def man_plot():
    folder = input("result folder name: ")

    full_test_pred = pd.read_csv(
                                 "./{}/full_test_pred.csv".format(folder),
                                 sep=",",
                                 header=None)
    
    full_train_pred = pd.read_csv(
                                  "./{}/full_train_pred.csv".format(folder),
                                  sep=",",
                                  header=None)

    y_data = pd.read_csv(
        "./{}/y_data.csv".format(folder),
        sep=",",
        header=None)

    plt.plot(
             range(len(y_data)),
             y_data,
             full_test_pred,
             full_train_pred
             )


if __name__ == "__main__":
    man_plot()
