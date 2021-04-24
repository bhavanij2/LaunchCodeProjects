function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X * theta;
prediction = sigmoid(z);

J_if_y_is_1 = y .* log(prediction);
J_if_y_is_0 = (1-y) .* log( 1 - prediction);
cost_value = - ( sum ( J_if_y_is_1 + J_if_y_is_0 ) / m );

reg_value = lambda * ( sum(theta(2:end) .^ 2) / (2 * m) );

J = cost_value + reg_value;


grad_transpose = ( sum ( ( prediction - y ) .* X ) ) / m;
grad_value = grad_transpose';

grad_for_j_1 = grad_value(1);

grad_reg_value = (lambda .* ( theta(2:end) ./ m ) );
grad_for_j_1_and_above = grad_value(2:end) + grad_reg_value;

grad = [grad_for_j_1; grad_for_j_1_and_above];

% =============================================================

end
