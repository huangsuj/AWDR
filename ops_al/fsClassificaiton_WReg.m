function [results] = fsClassificaiton_WReg(X, Y, ratio, lambda, k)
    viewNum = length(X);

    % Scatter all data points and select the given ratio of labeled data
    N = length(Y); % the number of data points
    p = randperm(N); % Scatter all data points
%     p = (1:N);
    n = floor(N*ratio); % The number of sampled data points
    labelIdx = p(1:n);
    unlabelIdx = p(n+1:end);

    for v = 1:viewNum
        X_train{v} = (X{v}(labelIdx,:))';
        X_test{v} = (X{v}(unlabelIdx,:))';
    end
    Y_train = Y(labelIdx,:);
    Y_test = Y(unlabelIdx,:);
    
    results = WREG(X_train, X_test, Y_train, Y_test, lambda, k);
end