% This module is to fetch data from Fred Database.

fileName = "UNRATE.csv";
data.main = csvread(fileName, 1, 1);

data.d1 = diff(data.main);  % First order difference in time series.
data.d2 = diff(data.main, 2);  % Second order difference in time series.
data.len= length(data.d2);  % Data length.

data.main = data.main(3: end);
data.d1 = data.d1(2: end);
data.stack = [data.main, data.d1, data.d2];