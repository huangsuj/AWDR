function predict_label = knn_classify(train_label, test_label, distance, K, flag)
    % distance: test_num * train_num
    % flag:'descend' or 'ascend'
    % K,the parameter of knn.
    num_test = length(test_label); 
    [~,id] = sort(distance, 2, flag);
    Label_matrix = train_label(id);
    first_k_label = Label_matrix(:,1:K);
    knn_label = zeros(num_test,1);
    for i=1:num_test 
        table = tabulate(first_k_label(i,:));
        [~,I]=max(table(:,2));
        knn_label(i)= I;
    end
    predict_label = knn_label;
