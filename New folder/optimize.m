close all
clear  
clc

rng('default') % For reproducibility
warning off

%% Data
dataset = {
            'UOT32_ResNet50.csv';
            };
          
algos ={'PSO';
%       'RSA'; 
%       'SO';   
       %'SSA';
 %      'FLA';
      % 'DMOA'
        };

T = 100; % Max. iterations
N = 30;   % Population size
runs = 10;% No. of independent runs

%% Load data
for ii=1:length(dataset) 
    filename = dataset{ii};
    fprintf('Database: %s\n', filename)
   
    data = csvread(fullfile('dataset', filename), 1);
    target = data(:, end);   % Target
    data = data(:, 1:end-1); % Input features
%% Reduce data (stratified)
    cv = cvpartition(target,'k', runs);

%% Initialization
    dim=size(data, 2);   % #features
    lb = min(data, [], 1);    ub = max(data, [], 1);  % Upper and Lower limits
    fitfun=str2func('split_fitness');
       
    for jj =1:length(algos)
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_P_', algos{jj}, '= zeros(runs, dim);']);
        eval(['conv_curve_', algos{jj}, '= zeros(runs, T);']);
        eval(['CT_', algos{jj}, '= zeros(runs, 1);']);

        for kk=1:runs
            ind = test(cv, kk);   % testing index
            X = data(ind, :); 
            y = target(ind, :);
            fprintf([algos{jj}, ' Pass: %d/%d\n'], kk, runs)
            t1 = tic();
            eval(['[Best_F_', algos{jj}, '(kk),'...
                   'Best_P_', algos{jj}, '(kk, :),'...
                   'conv_curve_', algos{jj}, '(kk, :)'...
                   ']='...
                  algos{jj}, '(N,T,lb,ub,dim,fitfun, X, y);']);
            eval(['CT_', algos{jj}, '(kk, 1) = toc(t1)']);
        end
        save(['Result\\', algos{jj},'_', filename(1:end-3),'mat'], ['Best_F_' algos{jj}],...
             ['Best_P_' algos{jj}], ['conv_curve_' algos{jj}], ['CT_' algos{jj}]);
    end
end