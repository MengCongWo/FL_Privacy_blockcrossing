function [p] = predict(theta, X)

    % function: Compute the predict probability.
    % para: theta is model parameter of size (n+1)*10,
    % para: X_t is feature data of size m*n,
    % return: p is probability vector of size m*1.
    
    [m,~]=size(X);
    
    h = exp([ones(m,1) X]*theta);
    
    [~,p] = max(h, [], 2);
    p=p-1;

end
