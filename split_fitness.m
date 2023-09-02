function fitness = split_fitness(Pos, X, y, lambda)
    if nargin < 4   % Default weightage for loss in fitness calculation
        lambda = 0.99;
    end
    
    f_ind = Pos > 0.5;    % Selected features
    
    if sum(f_ind) > 0     % At least 1 feature must be selected
        reduced_x_train = X(:,f_ind);   % Reduced feature set
        
        % Test model for fitness
        model = fitcknn(reduced_x_train, y, 'NumNeighbors',5, 'kFold', 5);
        loss_train = kfoldLoss(model);     % 5-fold CV loss
        
%       fitness = lambda * (1-acc) + (1-lambda) * fraction_selected_features
        fitness = lambda * loss_train + (1-lambda) * mean(f_ind);
    else    % If no feature is selected
        fitness = 1.0;
    end