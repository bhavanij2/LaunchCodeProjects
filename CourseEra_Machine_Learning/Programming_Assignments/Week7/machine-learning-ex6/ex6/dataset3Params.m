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
sigma = 0.3;

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




values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error_summary = zeros(64, 3);
index = 1;

for C_val = 1:8
  for sigma_val = 1:8
      C_local = values(C_val);
      sigma_local = values(sigma_val);
      error = predict_error(X, y, Xval, yval, C_local, sigma_local);
      
      error_summary(index, :) = [C_local, sigma_local, error];
      index = index + 1;
   endfor   
endfor

[min_error, row_index] = min(error_summary(:, 3));

C = error_summary(row_index, 1);
sigma = error_summary(row_index, 2);


% =========================================================================

end


function prediction_error = predict_error(X, y, Xval, yval, C, sigma)
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  predictions = svmPredict(model, Xval);

  prediction_error = mean(double(predictions ~= yval));
endfunction
