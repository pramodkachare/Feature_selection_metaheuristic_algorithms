%SSA Salp Swarm Algorithm
% [GBEST, GPOS, cgCurve, CT] = PSO (X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Mirjalili, S., Gandomi, A. H., Mirjalili, S. Z., Saremi, S.,
%               Faris, H., Mirjalili, S. M. (2017). 
%               Salp Swarm Algorithm: A bio-inspired optimizer for 
%               engineering design problems
%               Advances in Engineering Software, 114, 163-191.
%               DOI: 10.1016/j.advengsoft.2017.07.002
% 
%     [GBEST, GPOS] = PSO(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value GBEST and 1-by-(M-1) logical matrix
%     of selected features GPOS.
%
%     [GBEST, GPOS] = PSO(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     GBEST and 1-by-(M-1) logical matrix of selected features GPOS.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Aug 2023)


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
for i=1:size(salp_pos,1)
    salp_fit(1,i) = feval(fobj,salp_pos(i,:)', X, y);
end

[sorted_salps_fitness,sorted_indexes]=sort(salp_fit);

for newindex=1:N_P
    Sorted_salps(newindex,:)=salp_pos(sorted_indexes(newindex),:);
end

food_pos=Sorted_salps(1,:);
food_fit=sorted_salps_fitness(1);

fprintf('SSA: Iteration %d    fitness: %4.3f \n', 1, food_fit);
conv_curve(1)=food_fit;

%Main loop
tt=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
while tt <= Max_Iter
    
    c1 = 2*exp(-(4*tt/Max_Iter)^2); % Eq. (3.2) in the paper
    
    for i=1:size(salp_pos,1)
        
        salp_pos= salp_pos';
        
        if i<=N_P/2
            for j=1:1:N_Var
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    salp_pos(j,i)=food_pos(j)+c1*((UB(j)-LB(j))*c2+LB(j));
                else
                    salp_pos(j,i)=food_pos(j)-c1*((UB(j)-LB(j))*c2+LB(j));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif i>N_P/2 && i<N_P+1
            point1=salp_pos(:,i-1);
            point2=salp_pos(:,i);
            
            salp_pos(:,i)=(point2+point1)/2; % % Eq. (3.4) in the paper
        end
        
        salp_pos= salp_pos';
    end
    
    for i=1:size(salp_pos,1)
        
        Tp=salp_pos(i,:)>UB;
        Tm=salp_pos(i,:)<LB;
        salp_pos(i,:)= salp_pos(i,:).*~(Tp+Tm) + UB.*Tp + LB.*Tm;
        
%         SalpFitness(1,i)=fobj(SalpPositions(i,:));
                     salp_fit(1,i) = feval(fobj,salp_pos(i,:)', X, y);

        
        if salp_fit(1,i)<food_fit
            food_pos=salp_pos(i,:);
            food_fit=salp_fit(1,i);
            
        end
    end
    
    fprintf('SSA: Iteration %d    fitness: %4.3f \n', tt, food_fit);
    conv_curve(tt)=food_fit;
    tt = tt + 1;
    
    % PUT THIS CODE INSIDE LOOP OF ITERATIONS
    % Normalize particle position values

    % Population diversity as a whole
end

CT = toc(timer);       % Total computation time in seconds

fprintf('SSA: Final fitness: %4.3f \n', food_fit);
