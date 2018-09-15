# Artificial Neural Networks & Economic Forecasting

## What's This?

This is a project focusing on applying computer science techniques on economics-related topics. Specifically, we are trying to build up a read-to-use and user-friendly toolbox for economic forecasting backend by neural networks.

In this project, neural networks are built to make predictions on economic and financial data. Codes in this project are mainly python scripts and typical python libraries are required for the model to be trained.

Models are implemented in  `Keras` , `Tensorflow` and `Matlab`  libraries. (`tensorflow` and `matlab` models are archived as we found models based on `keras`  are more robust compared to the alternative libraries.)



## Active Topics

### Exchange Rate Forecasting: Canadian Dollar against US dollar

#### About this Topic

* <u>Topic directory</u>: `./k models/exchange/`

* In this project, we feed multiple (26 in total) exchange rate time series into a recurrent neural network and use lagged values to predict the future value of CAD-USD exchange rate.

#### Packages Required

* <u>Core</u>: the model is built on `keras` (`version 2.2.2`)  with `tensorflow`  backend.
* <u>Data</u>: `numpy` , `pandas` and `sklearn` are required for data processing.
* <u>Visualization</u>: `matplotlib` and `bokeh` are required for visualization.

#### Uni-Variate Version (in progress)

* The baseline model takes historical time series data of exchange rate and make single or multiple step forecasting.

#### Multi-Variate Version (version 0.0.1 functional but full of bugs)

* In additional to the baseline model, extra time series are fed to the neural networks. The multi-variate version takes longer to be trained but can achieve higher accuracy.

#### How To Try This Model

1. Open your terminal

2. Change directory: `cd ./k\ models/exchange/`

3. And execute the script `python{$your_python3_version} ./multi_ex.py`

4. Then follow the prompt shown in terminal

#### Functions

<u>Working on this</u>

#### Future Updates

* Other economic indicators will be added.

#### Demo Results

##### Univariate Single Step Prediction Result

![sample_output](https://github.com/TianyuDu/AnnEcon/blob/master/sample_output.svg)

## Archived Topics

Archived codes can be found at `./archived/`

* `./archived/alpha/`  is a `tensorflow` based model to forecast macroeconomic indicators like price level and unemployment rate.
* `./archived/matlab code/`  is a `matlab`  based model for macroeconomic indicator forecasting.
* <u>Note</u>: Archived models are scripts Functionality of archived topics are not guaranteed. 

## About Training on GPU-Accelerated Servers

* All model based on `keras`  can be training using GPU-accelerated servers automatically once applicable. And it's been tested using AWS server with Nvidia Tesla V100 GPU.

* <u>Note</u>: I tested the training function of  `keras` (`tensorflow backend`) on Amazon Web Service with GPUs (Nvidia Tesla V100). The training efficiency might <u>not</u> be higher with GPU accelerated server compared with CPU server (16C32T) for some tasks.

## References

* All reference papers and books could be found in Mendeley group [AnnEcon](https://www.mendeley.com/community/annecon/). 

## Database

* St. Louis Fed (FRED) Economic data by [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org)
* IMF DataMapper by [International Monetary Fund](http://www.imf.org/external/datamapper/datasets)
* [World Bank Open Data](https://data.worldbank.org)
* [Global Financial Data](https://www.globalfinancialdata.com/)


