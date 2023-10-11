%Main routine for feature selection using Nature-inspired Algorithms
% 
%     Apply feature selection using set of Nature-Inspired Algorithms (NIA)
%     on set of datasets and store the results in respective MAT file.
%
%     datasets = 1-by-N cell array of dataset names in CSV format. Default
%     storage if 'Dataset' directory. Any additional path must be included
%     separately in each filename.
%     
%     algos = 1-by-N cell array of NIA names. Available NIAs are:
%     PSO : Particle Swarm Optimization (1995)
%     GWO : Grey Wolf Optimization (2014)
%     ALO : Ant Lion Optimizer (2015)
%     MVO: Multi-Verse Optimizer (2016, February)
%     WOA: Whale Optimization Algorithm (2016, May)
%     SSA : Salp Swarm Algorithm (2017)
%     SMA: Slime Mould Algorithm (2020)
%     AOA: Arithmetic Optimization Algorithm (2021, April)
%     AO: Aquila Optimizer (2021, July)
%     ROA : Remora Optimization Algorithm (2021, December)
%     DMOA: Dwarf Mongoose Optimization Algorithm (2022, March)
%     RSA : Reptile Search Algorithm (2022, April)
%     SO : Snake Optimizer (2022, April)
%     CO: Cheetah Optimizer (2022, June)
%     FLA : Fick's Law Algorithm (2023)
%
%     Example:
%  
% Author: Pramod H. Kachare
% Last revised: 30/09/2023

%%
close all
clear  
clc

%% CONSTANTS
runs = 10;      % No. of independent runs
Max_Iter = 100; % Max. iterations per run
No_P = 20;      % No. of search agents
lambda = 0.99;  % Fitness contant (multiplier for loss value)   
K_fold = 2;     % No. of data folds (<=1 to use complete data)
is_bound = false;   % 1: Calculate LB & UB based on data or 0: Use 0 & 1
verbose = 1;    % Output print level (print best fitness after fixed iters)

%% Datasets and NIAs
DATA_PATH = '..\dataset'; % Path to the dataset folder

% List of datasets to use
datasets = {'HeartEW.csv';...
            };

% List of NIAs to use        
algos ={'PSO';
        'RSA_SO';
        };

%% Load data
for ii=1:length(datasets) 
    filename = datasets{ii};
    fprintf('Database: %s\n', filename)
    
    % Assuming data has header. Skip first row
    data = csvread(fullfile(DATA_PATH, filename), 1);
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
    N_Var = size(data, 2);   % #features
    % 1: Calculate LB & UB based on data or 0: Use 0 & 1
    LB = double(is_bound) * min(data, [], 1);  % Lower limits  
    UB = 1 + double(is_bound) * max(data, [], 1);  % Upper limits
    fitfun=str2func('split_fitness');
       
    for jj =1:length(algos)
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_P_', algos{jj}, '= zeros(runs, dim);']);
        eval(['conv_curve_', algos{jj}, '= zeros(runs, T);']);
        eval(['CT_', algos{jj}, '= zeros(runs, 1);']);

        for kk=1:runs
            if K_fold > 1   % Select seperate data fold for each run
                % Generate data fold index
                % If #data folds = #runs then each run uses different fold
                % else data folds are repeated.
                ind = test(cv, mod(kk, K_fold+1)+double(mod(kk, K_fold+1)==0)); 
                X = data(ind, :); 
                y = target(ind, :);
            end
            
            % Call specifi NIA
            fprintf([algos{jj}, ' Pass: %d/%d\n'], kk, runs)
            eval(['[Best_F_', algos{jj}, '(kk),'...
                   'Best_P_', algos{jj}, '(kk, :),'...
                   'conv_curve_', algos{jj}, '(kk, :),'...
                   'CT_', algos{jj}, '(kk, 1)]='...
                  algos{jj}, '(X, y, No_P, fitfun, N_Var, Max_Iter, LB, UB, verbose);']);

            % Save results after each run
            if isfolder('Results')
                mkdir('Results');
            end
            save(['Results\\', algos{jj},'_', filename(1:end-3),'mat'], ...
                 ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}],... 
                 ['conv_curve_' algos{jj}], ...
                 ['CT_' algos{jj}], ...
                 'LB', 'UB', 'N_Var');
        end
    end  % END of algo loop
end   % END of dataset loop

%% END OF main.m