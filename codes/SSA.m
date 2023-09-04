%SSA Salp Swarm Algorithm
% [food_fit, food_pos, conv_curve, CT] = SSA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
%
%   Main paper: Mirjalili, S., Gandomi, A. H., Mirjalili, S. Z., Saremi, S.,
%               Faris, H., Mirjalili, S. M. (2017). 
%               Salp Swarm Algorithm: A bio-inspired optimizer for 
%               engineering design problems
%               Advances in Engineering Software, 114, 163-191.
%               DOI: 10.1016/j.advengsoft.2017.07.002
% 
%     [food_fit, food_pos] = SSA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value food_fit and 1-by-(M-1) logical matrix
%     of selected features food_pos.
%
%     [food_fit, food_pos] = SSA(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     food_fit and 1-by-(M-1) logical matrix of selected features food_pos.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Sep 2023)


function [food_fit, food_pos, conv_curve, CT]=SSA(X, y, N_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
if nargin < 1
    error('MATLAB:notEnoughInputs', 'Please provide data for feature selection.');
end

if nargin < 2  % If only data is given, assume last column as target
    y = X(:, end);
    X = X(:, 1:end-1);
end

if nargin < 3  % Default 10 search agents
    N_P = 10;
end

if nargin < 4
    fobj = str2func('split_fitness'); % Apply feature selection
end

if nargin < 5
    N_Var = size(X, 2); % Apply feature selection on columns of X
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

%Initialize the positions of search agents
if length(UB) == 1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
    LB = repmat(LB, 1, N_Var);
end

% If each variable has a different lb and ub
salp_pos = zeros(N_P, N_Var);
for ii = 1:N_P
    salp_pos(ii, :) = (UB-LB) .* rand(1,N_Var) + LB;
end

conv_curve = zeros(1,Max_Iter);

%calculate the fitness of initial salps
salp_fit = inf*ones(1, N_P);
for ii=1:N_P
    salp_fit(1,ii) = feval(fobj,salp_pos(ii,:) > (LB+UB)/2, X, y);
end

[sorted_salps_fitness,sorted_indexes]=sort(salp_fit);
sorted_salps = salp_pos(sorted_indexes,:);

food_pos = sorted_salps(1,:);
food_fit = sorted_salps_fitness(1);

fprintf('SSA: Iteration %d    fitness: %4.3f \n', 1, food_fit);
conv_curve(1)=food_fit;

%Main loop
tt=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
while tt <= Max_Iter
    c1 = 2*exp(-(4*tt/Max_Iter)^2); % Eq. (3.2) in the paper
    for ii = 1:N_P
        if ii <= N_P/2
            for jj=1:1:N_Var
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    salp_pos(ii,jj)=food_pos(jj)+c1*((UB(jj)-LB(jj))*c2+LB(jj));
                else
                    salp_pos(ii,jj)=food_pos(jj)-c1*((UB(jj)-LB(jj))*c2+LB(jj));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif ii>N_P/2 && ii<N_P+1
            point1=salp_pos(ii-1, :);
            point2=salp_pos(ii, :);
            
            salp_pos(ii, :)=(point2+point1)/2; % % Eq. (3.4) in the paper
        end
    end
    
    for ii=1:N_P
        Tp=salp_pos(ii,:)>UB;
        Tm=salp_pos(ii,:)<LB;
        salp_pos(ii,:)= salp_pos(ii,:).*~(Tp+Tm) + UB.*Tp + LB.*Tm;

        salp_fit(1,ii) = feval(fobj,salp_pos(ii,:)', X, y);
        
        if salp_fit(1,ii)<food_fit
            food_pos=salp_pos(ii,:);
            food_fit=salp_fit(1,ii);
        end
    end
    
    if mod(tt, verbose) == 0
        fprintf('SSA: Iteration %d    fitness: %4.3f \n', tt, food_fit);
    end
    conv_curve(tt)=food_fit;
    tt = tt + 1;
end

CT = toc(timer);       % Total computation time in seconds

fprintf('SSA: Final fitness: %4.3f \n', food_fit);

%% END OF SSA.m