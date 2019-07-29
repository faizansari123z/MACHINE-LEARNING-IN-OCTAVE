function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
c=X(:,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
i=1;
if(length(theta)==2)
h=theta(1)+c.*theta(2);
endif
if(length(theta)==1)
h=theta(1)*c;
endif
p=h.-y;
p=p.^2;
J=sum(p)/(2*m);


% =========================================================================

end
