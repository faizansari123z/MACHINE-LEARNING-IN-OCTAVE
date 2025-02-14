function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
c=X(:,2);

for iter = 1:num_iters
 if(length(theta)==2)
h=theta(1)+c*theta(2);
endif
if(length(theta)==1)
h=theta(1)*c;
end
p=(h-y);
k=c.*p;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

        theta(1)=theta(1) - alpha*sum(p)/m;
         theta(2)=theta(2) - alpha*sum(k)/m;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
