%%
%  ---------------------------------------------------------------------
%  Machine Learning : Classification of Forest Cover Types by using Neural Networks.
%  ---------------------------------------------------------------------
%
%   Copyright (c) 2015 Stefan Contiu (stefan.contiu@gmail.com)
%   Published under MIT License, http://opensource.org/licenses/MIT.
% 
% 
%  About:
%   TODO : add some overall info. refs to data source, kaggle, andrew ng.
% 
%%
clear; close all; clc

% Data files : TODO
trainingSetCsvFile = '';
validationSetCsvFile = '';
testSetCsvFile = '';
submissionCsvFile = '';

% Parameters
input_layer_size  = 54;   % 54 input features 
hidden_layer_size = 30;   % 25 hidden units
forest_covers = 7;        % 7 output nodes coresponding to the 7 classes

%  ---------------------------------------------------------------------
%  Load Data
%  ---------------------------------------------------------------------
fprintf('Loading Training Data ...\n')
X = csvread('f:\edu\Kaggle\Forest Cover Type Prediction\Data\s_train.csv');

% remove first row (features names) and first column (ids)
X(1,:) = [];
X(:,1) = [];

% separate the class from training data
lastColumnIndex = size(X, 2);
y = X(:,lastColumnIndex);
X(:,lastColumnIndex) = [];

%  ---------------------------------------------------------------------
%  Randomly initialize neural network weights
%  ---------------------------------------------------------------------
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, forest_covers);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  ---------------------------------------------------------------------
%  Training the neural network
%  ---------------------------------------------------------------------
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 500); % try different values for 50

%  should also try different values of lambda
lambda = 0.5;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   forest_covers, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 forest_covers, (hidden_layer_size + 1));

%  ---------------------------------------------------------------------
%  Computing Accuracy
%  ---------------------------------------------------------------------
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

validX = csvread('f:\edu\Kaggle\Forest Cover Type Prediction\Data\s_valid.csv');
validX(:,1) = [];
validXlastColumnIndex = size(validX, 2);
validY = validX(:,validXlastColumnIndex);
validX(:,validXlastColumnIndex) = [];
validPred = predict(Theta1, Theta2, validX);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(validPred == validY)) * 100);

%  ---------------------------------------------------------------------
%  Construct the submission
%  ---------------------------------------------------------------------
%testx = csvread('f:\edu\kaggle\forest cover type prediction\data\test.csv');
%testx(1,:) = [];
%ids = testx(:,1);
%testx(:,1) = [];
%results = predict(theta1, theta2, testx);
%results = [ids results];
%csvwrite('f:\edu\kaggle\forest cover type prediction\data\octave.csv', results);
