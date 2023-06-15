function results = WREG(X_train,X_test,Y_train,Y_test, lambda, k)
    view_num = size(X_train, 2);
    iters = 200;
    epsilon = 5e-3;

    %% ---------- train the model ----------
    [XX_train, XX_test, W_dv, b, ~, ~] = train_weightreg(X_train, X_test, Y_train, view_num, lambda, iters, epsilon);
%     W = sqrt(dv_dim) .\ W_dv; % in case you wanna know

    %% --- classifcation with knn ---
    Ytrain = W_dv' * XX_train + b*ones(1, length(Y_train));
    Ytest = W_dv' * XX_test + b*ones(1, length(Y_test));
    D = L2_distance(W_dv' * XX_train + b*ones(1, length(Y_train)), W_dv' * XX_test + b*ones(1, length(Y_test))); 
    y_pred = knn_classify(Y_train, Y_test, D', k, 'ascend');
    
    %% -- obtain evaluation results
    pred_lables = [Y_train;Y_test];
    pred_req = [Ytrain';Ytest'];
    save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\Youtube\WREG_req.mat', 'pred_req');
    save('C:\Users\hsj\OneDrive\my_paper\HGCNNet_for_hsj\results\scatter\Youtube\WREG_label.mat', 'pred_lables');
    results = classification_metrics(Y_test, y_pred);
end
