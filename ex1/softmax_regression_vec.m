function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
%size theta: 785, 9
%size X: 785, 60000
%size y: 1, 60000
%num classes: 10

z = theta' * X; % size: 9, 60000
exp_z = exp(z); % size: 9, 60000
normalizer = sum(exp_z); % size: 1, 60000
p = exp_z ./ repmat(normalizer, num_classes - 1, 1); % size: 9, 60000

% This loop over classes is slower then loop over training data below
%for k=1:num_classes-1
%    f = -sum(log(p(k, y == k)));
%    g(:, k) = -X(:, y == k) * (1 - p(k, y == k))' + X(:, y ~= k) * p(k, y ~= k)';
%end

for i=1:m
    xi = X(:, i); % size 785, 1
    yi = y(i);
    for k=1:num_classes-1
        if yi == k
            f = f - log(p(k, i));
            g(:, k) = g(:, k) - xi * (1 - p(k, i));
        else
            g(:, k) = g(:, k) + xi * p(k, i);
        end
    end
end

  g=g(:); % make gradient a vector for minFunc

