function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, labels, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, labels, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(nn_params));

% check inputs
assert(size(X, 1) == size(labels, 1));

% Layers: 1st is the input layer, last is the output layer
% Reshape nn_params back into matrices for intermediate layers
layer_sizes = [input_layer_size, hidden_layer_size, num_labels];
Thetas = unpackThetas(nn_params, layer_sizes);

% Setup some useful variables
m = size(X, 1);
num_layers = length(layer_sizes);

% labels is an (m x 1) array of integers in the range 1:num_labels
% we cast it to an (m x num_labels) array, where each row is a perfect classifier output
y = zeros(m, num_labels);
idx = sub2ind(size(y), 1:m, labels(:)');
y(idx) = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% FF and store activations in all layers, including bias units:
a = cell(1, num_layers);
a{1} = [ones(m,1) X];  % each row is one example
for i = 1:num_layers-1
  a{i+1} = sigmoid(a{i}*Thetas{i}'); % compute next layer
  a{i+1} = [ones(m,1) a{i+1}]; % add bias unit
end %for
h = a{end}(:, 2:end); % each row of h is the output for the respective example (remove bias unit)

% Cost function
% mean is taken over DIM 1 - i.e. over training examples
% sum is taken over the remaining DIM 2 - i.e. over labels
J = -sum(mean(y.*log(h) + (1-y).*log(1-h)));
% Add regularization:
r = 0;
for i = 1:num_layers-1
  r = r + sum(sumsq(Thetas{i}(:,2:end)));  % skip first column - i.e. bias weights
end %for
J = J + lambda*r/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta = cell(1, num_layers);
delta{end} = h-y;
for i = num_layers-1:-1:2
  % BP from layer (i+1) to layer 1
  gprime_i = sigmoidGradientAtInverse(a{i}); % derivative of activation function in layer i
  bp = delta{i+1}*Thetas{i}; % backpropagation
  delta{i} = bp.*gprime_i; % multiply by individual derivatives
  delta{i} = delta{i}(:, 2:end); % remove delta at bias unit
end %for

% compute gradients for each Theta matrix
ThetaGradients = Thetas; % same size as Thetas
for i = 1:num_layers-1
  ThetaGradients{i} = delta{i+1}'*a{i}/m;
  % add reguralization for all except first column:
  ThetaGradients{i}(:, 2:end) = ThetaGradients{i}(:, 2:end)  + lambda*Thetas{i}(:, 2:end)/m;
end %for


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = packThetas(ThetaGradients{:});


end