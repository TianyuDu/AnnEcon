# myDeepLearning by Spikey

## Contents

### Time series analysis and prediction with neural networks

#### Introduction

In this project, we use neural networks with `tensorflow` to predict <u>economic time series</u> data.



#### Feature Extraction

In simple versions of RNN models, the model predicts future movements of indicator solely based on <u>previous data</u>. 

**Working** multivariate case, we use <u>CNN</u> and the <u>activation/hidden layers of RNN</u> as the extracted features. The extracted features are then passed into a RNN (LSTM or GRU, depends on configuration) for forecasting.



#### Database

* St. Louis Fed (FRED) Economic data by [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org)

* IMF DataMapper by [International Monetary Fund](http://www.imf.org/external/datamapper/datasets)

* [World Bank Open Data](https://data.worldbank.org)




