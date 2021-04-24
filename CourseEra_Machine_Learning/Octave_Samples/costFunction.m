function J = costFunction(X, y, theta)  prediction = X * theta  sample_size = size(X,1)  squared_errors = (prediction - y) .^ 2   J = 1 / (2 * sample_size) * sum (squared_errors)
endfunction
