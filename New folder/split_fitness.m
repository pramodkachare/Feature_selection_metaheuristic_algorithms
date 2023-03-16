function fitness = split_fitness(f_ind, X, y, lmbda)
    if nargin < 4
        lmbda = 0.9;
    end
    
    f_ind = f_ind > 0.5;
    if sum(f_ind) > 0
        reduced_x_train=X(:,f_ind);

        model = fitcknn(reduced_x_train, y, 'NumNeighbors',5, 'kFold', 5);
        acc_train = 1-kfoldLoss(model);
        
        fitness=lmbda*(1-acc_train)+(1-lmbda)*sum(f_ind)/size(X, 1);
    else
        fitness = inf;
    end