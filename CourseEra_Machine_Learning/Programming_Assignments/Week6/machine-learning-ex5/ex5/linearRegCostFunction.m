function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

prediction = X * theta; % 12 * 1

costfunc_without_reg = sum ((prediction - y ) .^ 2) / (2 * m);

reg = sum(theta(2:end) .^ 2 ) * lambda / (2 * m);

J = costfunc_without_reg + reg;

grad_without_reg = sum((prediction - y) .* X, 1) / m ; % 1 * 2

grad_for_theta_zero = grad_without_reg(1);
grad_for_theta_non_zero = grad_without_reg(2:end); 

if lambda > 0
   grad_for_theta_non_zero = grad_for_theta_non_zero + (lambda .* theta(2:end)' / m) ;   
endif

grad = [grad_for_theta_zero grad_for_theta_non_zero] ;

% =========================================================================

grad = grad(:);

end
