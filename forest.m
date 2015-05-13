%  ---------------------------------------------------------------------
%  Classification of Forest Cover Types by using Neural Networks
%  ---------------------------------------------------------------------
% 
%  TODO : add link to src of data set. Put smthg on overall context.

clear; close all; clc

% Parameters
input_layer_size  = 54;   % 54 input features 
hidden_layer_size = 25;   % 25 hidden units
forest_covers = 7;       % prev num_labels

%  ---------------------------------------------------------------------
%  Load Data
%  ---------------------------------------------------------------------
fprintf('Loading Training Data ...\n')
X = csvread('f:\edu\Kaggle\Forest Cover Type Prediction\Data\train.csv');

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

options = optimset('MaxIter', 50); % try different values for 50

%  should also try different values of lambda
lambda = 1;

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
%  Predict
%  ---------------------------------------------------------------------
                 
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
