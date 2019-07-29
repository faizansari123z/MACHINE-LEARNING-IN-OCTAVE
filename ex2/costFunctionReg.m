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
z=sum(theta.^2)-theta(1,1)^2;
%J=(-1/m)*sum(y'*log(sigmoid(X*theta)-y)+(1-y)'*log(1-sigmoid(X*theta)))+(lambda/(2*m))*z;
J=(-1/m)*sum(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))+(lambda*z)/(2*m); 
 grad(1,1)=(1/m)*X(:,1)'*(sigmoid(X*theta)-y);
for i=2:length(X(1,:)),
  grad(i,1)=(1/m)*X(:,i)'*(sigmoid(X*theta)-y)+(lambda/m)*theta(i,1);
endfor




% =============================================================

end
