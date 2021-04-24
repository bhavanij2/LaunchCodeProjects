function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
       
prediction = X * theta;
error = prediction - y;

X_1 = X(:, 1);
J_derivative_for_theta_1 = (1 / m) * sum(error .* X_1);
theta_1 = theta(1) - alpha * J_derivative_for_theta_1;

X_2 = X(:,2);
J_derivative_for_theta_2 = (1 / m) * sum(error .* X_2);
theta_2 = theta(2)- alpha * J_derivative_for_theta_2;

theta = [theta_1; theta_2];


    fprintf('Iteration: %f\n', iter);
    fprintf('Gradient Descent:\n');
    fprintf('%f\n', theta);
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    fprintf('J Value: %f\n', J_history(iter));

end

end
