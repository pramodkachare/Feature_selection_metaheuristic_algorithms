%% Main routine for feature selection using Nature-inspired Algorithms
% Revised date: 21/08/2023

%%
close all
clear  
clc

%% CONSTANTS
% Assuming data stored in the Dataset folder.
datasets = {'HeartEW.csv';...
            };

% List of NIAs        
algos ={'PSO';
        'RSA_SO';
        };

runs = 10;  % No. of independent runs
T = 100;    % Max. iterations per run
N = 20;     % No. of search agents

K_fold = 2; % No. of data folds (<=1 to use complete data)

%% Load data
for ii=1:length(datasets) 
    filename = datasets{ii};
    fprintf('Database: %s\n', filename)
    
    % Assuming data has header. Skip first row
    data = csvread(fullfile('..\dataset', filename), 1);
    target = data(:, end);   % Target
    data = data(:, 1:end-1); % Input features
    
    if K_fold > 1
        % Reduce data (stratified partitions)
        cv = cvpartition(target,'k', K_fold);
    else
        % use complete data for each iteration
        X = data;   y = target;
    end

%% Initialization
    dim=size(data, 2);   % #features
    lb = 0; % min(data, [], 1);    
    ub = 1; % max(data, [], 1);  % Upper and Lower limits
    fitfun=str2func('split_fitness');
       
    for jj =1:length(algos)
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_P_', algos{jj}, '= zeros(runs, dim);']);
        eval(['conv_curve_', algos{jj}, '= zeros(runs, T);']);
        eval(['CT_', algos{jj}, '= zeros(runs, 1);']);

        for kk=1:runs
            if K_fold > 1   % Select seperate data fold for each run
                % Generate data fold index
                ind = test(cv, mod(kk, K_fold)+K_fold*double(mod(kk, K_fold)==0));   
                X = data(ind, :); 
                y = target(ind, :);
            end
            
            % Call specifi NIA
            fprintf([algos{jj}, ' Pass: %d/%d\n'], kk, runs)
            eval(['[Best_F_', algos{jj}, '(kk),'...
                   'Best_P_', algos{jj}, '(kk, :),'...
                   'conv_curve_', algos{jj}, '(kk, :),'...
                   'CT_', algos{jj}, '(kk, 1)]='...
                  algos{jj}, '(N,T,lb,ub,dim,fitfun, X, y);']);
            
            % Save results after each run
            if isfolder('Results')
                mkdir('Results');
            end
            save(['Results\\', algos{jj},'_', filename(1:end-3),'mat'], ...
                 ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}],... 
                 ['conv_curve_' algos{jj}], ['CT_' algos{jj}]);
        end
    end  % END of algo loop
end   % END of dataset loop

%% END OF main.m