close all
clear  
clc

%% Data
datasets = {
%             'kddcup99_proc.csv';
            'HeartEW.csv'
            };
        
algos ={'DMOA';
%         'RSA'; % AEO
%         'SO';  % DMOA 
%         'RSA_SO';
%         'SSA';
%         'WOA';
%         'GWO';
%         'MVO';
        };
%% Load data
for ii=1:length(datasets) 
    filename = datasets{ii};
    fprintf('Database: %s\n', filename)
    
    if strcmpi(filename, 'pressure_vessel.csv')
        lb = [0 0 10 10];    
        ub = [99 99 200 200];
        dim = 4;   % #features
        fitfun=str2func('pressure_vessel');
        X = [];
        y = [];
    elseif strcmpi(filename, 'cantilever_beam.csv')
        lb = [0.01, 0.01, 0.01, 0.01, 0.01];    
        ub = [100, 100, 100, 100, 100];
        dim = 5;   % #features
        fitfun=str2func('cantilever_beam');
        X = [];
        y = [];
    else
        X = csvread(fullfile('dataset', filename), 1);
        y = X(:, end);   % Target
        X = X(:, 1:end-1); % Input features
%% Initialization
        dim=size(X, 2);   % #features
        lb = repmat(min(X(:)), 1, dim);    ub = repmat(max(X(:)), 1, dim);  % Upper and Lower limits
        fitfun=str2func('split_fitness');
    end
    T = 100; % Max. iterations
    N = 30;   % Population size
    runs = 10;% No. of independent runs
       
    for jj =1:length(algos)
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_P_', algos{jj}, '= zeros(runs, dim);']);
        eval(['conv_curve_', algos{jj}, '= zeros(runs, T);']);
        eval(['CT_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Count_rsa_', algos{jj}, '= zeros(runs, T);']);
        eval(['Count_so_', algos{jj}, '= zeros(runs, T);']);

        for kk=1:runs
            fprintf([algos{jj}, ' Pass: %d/%d\n'], kk, runs)
            if ~strcmpi(algos{jj}, 'RSA_SO')
                t1 = tic;
                eval(['[Best_F_', algos{jj}, '(kk),'...
                       'Best_P_', algos{jj}, '(kk, :),'...
                       'conv_curve_', algos{jj}, '(kk, :)]='...
                      algos{jj}, '(N,T,lb,ub,dim,fitfun, X, y);']);
                eval(['CT_' algos{jj}, '(jj) = toc(t1);']);
            else
                eval(['[Best_F_', algos{jj}, '(kk),'...
                       'Best_P_', algos{jj}, '(kk, :),'...
                       'conv_curve_', algos{jj}, '(kk, :),'...
                       'CT_', algos{jj}, '(kk, 1), '...
                       'Count_rsa_', algos{jj}, '(kk, :),'...
                       'Count_so_', algos{jj}, '(kk, :)]=',...
                      algos{jj}, '(N,T,lb,ub,dim,fitfun, X, y);']);
            end
        end
        if ~strcmpi(algos{jj}, 'RSA_SO')
            save([algos{jj},'_', filename(1:end-3),'mat'], ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}], ['conv_curve_' algos{jj}], ['CT_' algos{jj}]');
        else
            save([algos{jj},'_', filename(1:end-3),'mat'], ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}], ['conv_curve_' algos{jj}], ['CT_' algos{jj}]',...
                 ['Count_rsa_' algos{jj}], ['Count_so_' algos{jj}]);
        end
    end
end