function dg = sigmoidGradientAtInverse(g)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at inverse of g
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z, where sigmoid(z) = g


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

dg = g.*(1-g);

% =============================================================




end
