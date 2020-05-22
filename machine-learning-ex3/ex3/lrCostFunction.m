function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

theta2 = theta;
theta2(1) = 0;



               
J = (-1/m) * ((y'*log(sigmoid(X*theta))) + ((ones(1,m)-y')*log(ones(m,1)-sigmoid(X*theta)))) + ...
    (lambda/(2*m)) * (theta2'*theta2); 

    
grad = (1/m) * (X'*(sigmoid(X*theta) - y) + lambda*theta2);







% =============================================================

grad = grad(:);

end
