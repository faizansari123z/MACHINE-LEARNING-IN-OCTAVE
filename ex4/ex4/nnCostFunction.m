function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Z=[];
for i=1:max(y)
  Z=[Z y==(size(Z,2)+1)];
endfor
a1=[ones(m,1) X]';

z2=Theta1*a1;
a2=[ones(1,size(z2,2)) ;sigmoid(z2)];

h3=sigmoid(Theta2*a2);
r=0;
r=sum(sum(Theta1.^2))+sum(sum(Theta2.^2))-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2);

J=(1/m)*sum(sum(-Z'.*log(h3)-(1-Z)'.*log(1-h3)))+(lambda*r)/(2*m);
delta3=h3-Z';
delta2=(Theta2'*delta3).*(a2.*(1-a2));

delta2=delta2(2:end,:);
D1=zeros(size(delta2,1),size(a1,1));
D2=zeros(size(delta3,1),size(a2,1));

D1=D1+delta2*a1';
D2=D2+delta3*a2';
theta1=Theta1;
theta2=Theta2;
theta1(:,1)=zeros(1,length(Theta1(:,1)));
theta2(:,1)=zeros(1,length(Theta2(:,1)));
Theta1_grad=D1/m+(lambda/m)*theta1;
Theta2_grad=D2/m+(lambda/m)*theta2;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
