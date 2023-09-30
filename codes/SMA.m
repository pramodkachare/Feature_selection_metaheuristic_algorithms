%SMA Slime Mould Algorithm
% [dest_fit,best_pos,conv_curve, CT] = SMA(data, target, No_P, fobj, 
%                                        N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Li, S., Chen, H., Wang, M., Heidari, A. A., 
%               & Mirjalili, S. (2020). 
%               Slime mould algorithm: A new method for stochastic optimization.  
%               Future Generation Computer Systems, 111, 300-323.
%               DOI: 10.1016/j.future.2020.03.055
% 
%     [dest_fit,best_pos] = WOA(data) applies feature selection on 
%     M-by-N matrix data with N examples and assuming last column as the 
%     classification target and returns the best fitness value dest_fit 
%     and 1-by-(M-1) matrix of feature positions best_pos.
%
%     [dest_fit,best_pos] = WOA(data, target) applies feature selection 
%     on M-by-N feature matrix data and 1-by-N target matrix target and 
%     returns the best fitness value dest_fit and 1-by-(M-1)matrix of 
%     feature positions best_pos.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Sep 2023)

function [dest_fit,best_pos,conv_curve, CT]=SMA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
if nargin < 1
    error('MATLAB:notEnoughInputs', 'Please provide data for feature selection.');
end

if nargin < 2  % If only data is given, assume last column as target
    target = data(:, end);
    data = data(:, 1:end-1);
end

if nargin < 3  % Default 10 search agents
    No_P = 10;
end

if nargin < 4
    fobj = str2func('split_fitness'); % Apply feature selection
end

if nargin < 5
    N_Var = size(data, 2); % Apply feature selection on columns of X
end

if nargin < 6
    Max_Iter = 100;     % Run optimization for max 100 iterations
end

if nargin < 8
    UB = 1;     % Assume upper limit for each variable is 1
end
if nargin < 7
    LB = 0;     % Assume lower limit for each variable is 0
end

if nargin < 9
    verbose = 1; % Print progress after each iteration
end
%Start timer
timer = tic();

% initialize position
best_pos=zeros(1,N_Var);
dest_fit=inf;%change this to -inf for maximization problems
AllFitness = inf*ones(No_P,1);%record the fitness of all slime mold
weight = ones(No_P,N_Var);%fitness weight of each slime mold

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the set of random solutions
X = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));

conv_curve=zeros(1,Max_Iter);

tt=1;  %Number of iterations

z=0.03; % parameter

% Main loop
while  tt <= Max_Iter
    
    %sort the fitness
    for ii=1:No_P
        % Check if solutions go outside the search space and bring them back
        Flag4ub=X(ii,:)>UB;
        Flag4lb=X(ii,:)<LB;
        X(ii,:)=(X(ii,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        AllFitness(ii) = fobj(X(ii,:), data, target);
    end
    
    [SmellOrder,SmellIndex] = sort(AllFitness);  %Eq.(2.6)
    worstFitness = SmellOrder(No_P);
    bestFitness = SmellOrder(1);

    S=bestFitness-worstFitness+eps;  % plus eps to avoid denominator zero

    %calculate the fitness weight of each slime mold
    for ii=1:No_P
        for jj=1:N_Var
            if ii<=(No_P/2)  %Eq.(2.5)
                weight(SmellIndex(ii),jj) = 1+rand()*log10((bestFitness-SmellOrder(ii))/(S)+1);
            else
                weight(SmellIndex(ii),jj) = 1-rand()*log10((bestFitness-SmellOrder(ii))/(S)+1);
            end
        end
    end
    
    %update the best fitness value and best position
    if bestFitness < dest_fit
        best_pos=X(SmellIndex(1),:);
        dest_fit = bestFitness;
    end
    
    a = atanh(-(tt/Max_Iter)+1);   %Eq.(2.4)
    b = 1-tt/Max_Iter;
    % Update the Position of search agents
    for ii=1:No_P
        if rand<z     %Eq.(2.7)
            X(ii,:) = (UB-LB)*rand+LB;
        else
            p =tanh(abs(AllFitness(ii)-dest_fit));  %Eq.(2.2)
            vb = unifrnd(-a,a,1,N_Var);  %Eq.(2.3)
            vc = unifrnd(-b,b,1,N_Var);
            for jj=1:N_Var
                r = rand();
                A = randi([1,No_P]);  % two positions randomly selected from population
                B = randi([1,No_P]);
                if r<p    %Eq.(2.1)
                    X(ii,jj) = best_pos(jj)+ vb(jj)*(weight(ii,jj)*X(A,jj)-X(B,jj));
                else
                    X(ii,jj) = vc(jj)*X(ii,jj);
                end
            end
        end
    end
    conv_curve(tt)=dest_fit;
    tt=tt+1;
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('SMA: Iteration %d    fitness: %4.3f \n', tt, dest_fit);
    end
end
CT = toc(timer);       % Total computation time in seconds
fprintf('WOA: Final fitness: %4.3f \n', dest_fit);

%% END OF SMA.m
