function [J, grad] = lrCostFunction(theta, X, y, lambda)

    % function: Compute the loss and gradient.
    % para: theta is model parameter of size (n+1)*10, 
    % para: X is input feature data of size m*n,
    % para: y is input lable data of size m*1,
    % return: J is training loss,
    % return: grad is gradient used for GD of size (n+1)*10.

    [m,n] = size(X);
    k = 10;  % Number of classes.
    
    grad = zeros(n+1,k);
    X = [ones(m,1) X];
    
    % Compute the probability h
    h = exp(X*theta);
    h_sum = sum(h,2);
    h = h./repmat(h_sum,[1 k]);

    idx = y*m+(1:m)';
    J = -1/m*sum(log(h(idx))) + 0.5*lambda/m*sum(sum(theta(2:end,:).^2));
    
    l = (repmat(y,[1,k]) == repmat((0:k-1), [m,1]));
    grad(1, :) = -1/m*X(:,1)'*(l-h);
    grad(2:end, :) = -1/m*X(:,2:end)'*(l-h) + lambda/m*theta(2:end,:);

end
