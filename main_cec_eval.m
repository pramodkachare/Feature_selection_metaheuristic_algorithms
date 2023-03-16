close all
clear  
clc

%% Data
 funs = {'cec1';
        'cec2';
        'cec3';
        'cec4';
        'cec5';
        'cec6';
        'cec7';
        'cec8';
        'cec9';
        'cec10';
%         'cec11';
%         'cec12';
%         'cec13';
%         'cec14';
%         'cec15';
%         'cec16';
%         'cec17';
%         'cec18';
        };
        
algos ={%'RSA'; % AEO
        %'SO';  % DMOA 
%         'RSA_SO';
        'FLA';
        };
    
%% Load data
for ii=1:length(funs) 
    filename = funs{ii};
    fprintf('Database: %s\n', filename)
    
    [l, u, dim, fitfun]=CEC2019(funs{ii});
    if length(l) ==1
        lb = repmat(l, 1, dim);    
    else
        lb = l;
    end
    if length(u) == 1
        ub = repmat(u, 1, dim); 
    else
        ub = u;
    end

    X = [];
    y = [];

    T = 100; % Max. iterations
    N = 32;   % Population size
    runs = 20;% No. of independent runs
       
    for jj =1:length(algos)
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_F_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Best_P_', algos{jj}, '= zeros(runs, dim);']);
        eval(['conv_curve_', algos{jj}, '= zeros(runs, T);']);
        eval(['P_hist_', algos{jj}, '= cell(runs, 1);']);
        eval(['CT_', algos{jj}, '= zeros(runs, 1);']);
        eval(['Count_rsa_', algos{jj}, '= zeros(runs, T);']);
        eval(['Count_so_', algos{jj}, '= zeros(runs, T);']);

        for kk=1:runs
            fprintf([algos{jj}, ' Pass: %d/%d\n'], kk, runs)
            if ~strcmpi(algos{jj}, 'RSA_SO')
                t1 = tic;
                eval(['[Best_F_', algos{jj}, '(kk),'...
                       'Best_P_', algos{jj}, '(kk, :),'...
                       'conv_curve_', algos{jj}, '(kk, :),'...
                       'P_hist_', algos{jj}, '{kk}]='...
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
            save([algos{jj},'_', filename,'.mat'], ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}], ['conv_curve_' algos{jj}], ...
                 ['P_hist_' algos{jj}], ['CT_' algos{jj}]');
        else
            save([algos{jj},'_2019_', filename,'.mat'], ['Best_F_' algos{jj}],...
                 ['Best_P_' algos{jj}], ['conv_curve_' algos{jj}], ['CT_' algos{jj}]',...
                 ['Count_rsa_' algos{jj}], ['Count_so_' algos{jj}]);
        end
    end
end