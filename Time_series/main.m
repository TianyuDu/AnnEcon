clear all;
clc;

%% Loading data from source;
filename = "CPIAUCSL.csv";
data = csvread(filename, 1, 1);
data = data'; % Row time time series data.

%% Initialize figure;
figure
plot(data)
xlabel("Time")
ylabel("Consumer Price Index")
% Consumer Price Index for All Urban Consumers: All Items
title("Consumer Price Index for All Urban Consumers: All Items")

%% Creating sub-dataset;
numTimeStepsTrain = floor(0.9 * numel(data));
XTrain = data(1: numTimeStepsTrain);
YTrain = data(2: numTimeStepsTrain + 1);
% One time step forward forcesting.
XTest = data(numTimeStepsTrain + 1: end -1);
YTest = data(numTimeStepsTrain + 2: end);

%% Standardize Data;
mu = mean(XTrain);
sig = std(XTrain);

XTrain = (XTrain - mu) ./ sig;
YTrain = (YTrain - mu) ./ sig;

XTest = (XTest - mu) ./ sig;

%% Setup LSTM;
inputSize = 1; % Dimension of input sequence.
numResponses = 1; % Dimension of output sequence.

numHiddenUnits.lstm1 = 16;
numHiddenUnits.lstm2 = 8;
numHiddenUnits.fc1 = 64;

layers = [...
	sequenceInputLayer(inputSize)
	lstmLayer(numHiddenUnits.lstm1)
	lstmLayer(numHiddenUnits.lstm2)
	fullyConnectedLayer(numHiddenUnits.fc1)
	fullyConnectedLayer(numResponses)
	regressionLayer
	];

opts = trainingOptions(...
	"adam",...
	"MaxEpochs", 250, ...
	"GradientThreshold", 1, ...
	"InitialLearnRate", 0.005, ...
	"LearnRateSchedule", "piecewise", ...
	"LearnRateDropPeriod", 125, ...
	"LearnRateDropFactor", 0.2, ...
	"Verbose", 0, ...
	"Plot", "training-progress");

%% Training;
net = trainNetwork(...
	XTrain, ...
	YTrain, ...
	layers, ...
	opts);

[net, YPred] = predictAndUpdateState(net, YTrain(end));

numTimeStepsTest = numel(XTest);

for i = 2:numTimeStepsTest
	[net, YPred(1, i)] = predictAndUpdateState(net, YPred(i - 1));
end

% Unstandardize.
YPred = sig * YPred + mu;

% error, from unstandardized data.
rmse = sqrt(mean((YPred - YTest) .^ 2));

%% Visualize;
% Main graph
figure
plot(data(1: numTimeStepsTrain));
hold on
idx = numTimeStepsTrain: (numTimeStepsTrain + numTimeStepsTest);
plot(idx, [data(numTimeStepsTrain) YPred], ".-");
hold off
xlabel("Date")
ylabel("CPI")
title("Forecast")
legend(["Observed" "Forecast"])

% Combined paramter indicator graph.
figure
subplot(2, 1, 1)
plot(YTest)
hold on
plot(YPred, ".-")
hold off
legend(["Observed", "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2, 1, 2)
stem(YPred - YTest)
xlabel("Date")
ylabel("Error")
title("RMSE=" + rmse)

%% Update netowrk state with observed values
net = resetState(net);
net = predictAndUpdateState(net, XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);

for i = 1:numTimeStepsTest
	[net, YPred(1, i)] = predictAndUpdateState(net, XTest(i));
end

%% Unstandardize





