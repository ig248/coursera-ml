function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
if 0  % run optimization and show debug plots
Cmid = 1;
sigmamid = 0.1;
factor = 2;

C_grid = Cmid*factor.^(-4:4);
sigma_grid = sigmamid*factor.^(-4:4);

C_n = numel(C_grid);
sigma_n = numel(sigma_grid);

train_err      = zeros(C_n, sigma_n);
validation_err = zeros(C_n, sigma_n);

for c = 1:C_n
  C = C_grid(c);
  disp(['C = ', num2str(C)]);
  for s = 1:sigma_n
    sigma = sigma_grid(s);
    disp(['-- sigma = ', num2str(sigma)]);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    train_pred = svmPredict(model, X);
    valid_pred = svmPredict(model, Xval);
    train_err(c, s) = mean(double(train_pred ~= y));
    validation_err(c, s) = mean(double(valid_pred ~= yval));
  end  % for c
end  % for s


  train_err
  validation_err
  [C_mesh, sigma_mesh] = meshgrid(C_grid, sigma_grid);
  
  % note that transpose is needed as MESH treats first dimension as Y
  figure()
  mesh(C_mesh, sigma_mesh, train_err')
  xlabel('C')
  ylabel('sigma')
  zlabel('train error')
  set(gca,'XScale','log');
  set(gca,'YScale','log');
    keyboard
  
  figure()
  mesh(C_mesh, sigma_mesh, validation_err')
  xlabel('C')
  ylabel('sigma')
  zlabel('validation error')
  set(gca,'XScale','log');
  set(gca,'YScale','log');
  
  keyboard

  %% find C and sigma with smallest validation error
  [M,I] = min(validation_err(:));
  [I_row, I_col] = ind2sub(size(validation_err), I)
  C     = C_grid(I_row);
  sigma = sigma_grid(I_col);
end
  % =========================================================================

end
