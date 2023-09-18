function fitness = split_fitness_reg(Pos, X, y, lambda)
    if nargin < 4   % Default weightage for loss in fitness calculation
        lambda = 0.99;
    end
    
    f_ind = Pos > 0.5;    % Selected features
    
    if sum(f_ind) > 0     % At least 1 feature must be selected
        reduced_x_train = X(:,f_ind);   % Reduced feature set
        
        % Test model for fitness
        model = fitrsvm(reduced_x_train, y, 'Standardize', true, 'kFold', 5);
        corr = corrcoef(kfoldPredict(model), y);
        loss_train = corr(1,2)^2; % R_squared
        
%       fitness = lambda * (1-acc) + (1-lambda) * fraction_selected_features
        fitness = lambda * loss_train + (1-lambda) * mean(f_ind);
    else    % If no feature is selected
        fitness = 1.0;
    end