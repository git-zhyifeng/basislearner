clear ; close all; clc
load('data');      % Get X,Y data matrices, consisting of 500 examples.
widths = [10 10];  % construct network with 2 layers of width 10, plus an output layer.
trainend = 250;    % Use first 250 examples in X,Y as a training set.
lambdaRange = 10.^(-3:1);
F = BuildNetwork(X,Y,widths,trainend);
Results = BuildOutputLayers(F,Y,trainend,widths,lambdaRange);