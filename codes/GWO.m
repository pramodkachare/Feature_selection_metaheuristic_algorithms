%GWO Grey Wolf Optimizer
% [alpha_score,alpha_pos,Convergence_curve, CT] = GWO(X, y, No_P, fobj,
% N_Var, Max_Iter, LB, UB, verbose)
%
%   Main paper: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
%               Grey wolf optimizer 
%               Advances in Engineering Software, 69, 46–61.
%               DOI: 10.1016/j.advengsoft.2013.12.007               
% 
%     [alpha_score, alpha_pos] = GWO(X) applies feature selection on M-by-N 
%     matrix X with N examples and assuming last column as the classification 
%     target and returns the best fitness value alpha_score and 1-by-(M-1) 
%     logical  matrix of selected features alpha_pos.
%
%     [alpha_score, alpha_pos] = GWO(X, y) applies feature selection on 
%     M-by-N feature matrix X and 1-by-N target matrix y and returns the 
%     best fitness value  alpha_score and 1-by-(M-1) logical matrix of 
%     selected features alpha_pos.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Aug 2023)

function [alpha_fit,alpha_pos,conv_curve, CT] = GWO(X, y, N_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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
Pos = zeros(N_P, N_Var);
for ii = 1:N_P
    Pos(ii, :) = (UB-LB) .* rand(1,N_Var) + LB;
end

conv_curve = zeros(1,Max_Iter);

% initialize alpha, beta, and delta_pos
alpha_pos = zeros(1, N_Var);
alpha_fit = inf; %change this to -inf for maximization problems

beta_pos = zeros(1,N_Var);
beta_fit = inf; %change this to -inf for maximization problems

delta_pos = zeros(1,N_Var);
delta_fit = inf; %change this to -inf for maximization problems

tt = 0;% Loop counter

% Main loop
while tt < Max_Iter
    for ii = 1:N_P  
        
       % Return the search agents that go beyond the boundaries of the search space
        Pos(ii,:) = UB(Pos(ii,:)>UB);
        Pos(ii,:) = LB(Pos(ii,:)<LB);
                
        % Calculate objective function for each search agent
        fitness = fobj(Pos(ii,:), X, y);
        
        % Update alpha, beta, and delta
        if fitness<alpha_fit 
            alpha_fit = fitness; % Update alpha
            alpha_pos = Pos(ii,:);
        end
        
        if fitness>alpha_fit && fitness<beta_fit 
            beta_fit = fitness; % Update beta
            beta_pos = Pos(ii,:);
        end
        
        if fitness>alpha_fit && fitness>beta_fit && fitness<delta_fit 
            delta_fit = fitness; % Update delta
            delta_pos = Pos(ii,:);
        end
    end
    
    a = 2-tt*((2)/Max_Iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for ii = 1:size(Pos,1)
        for j = 1:size(Pos,2)     
                       
            r1 = rand(); % r1 is a random number in [0,1]
            r2 = rand(); % r2 is a random number in [0,1]
            
            A1 = 2*a*r1-a; % Equation (3.3)
            C1 = 2*r2; % Equation (3.4)
            
            D_alpha = abs(C1*alpha_pos(j)-Pos(ii,j)); % Equation (3.5)-part 1
            X1 = alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1 = rand();
            r2 = rand();
            
            A2 = 2*a*r1-a; % Equation (3.3)
            C2 = 2*r2; % Equation (3.4)
            
            D_beta = abs(C2*beta_pos(j)-Pos(ii,j)); % Equation (3.5)-part 2
            X2 = beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1 = rand();
            r2 = rand(); 
            
            A3 = 2*a*r1-a; % Equation (3.3)
            C3 = 2*r2; % Equation (3.4)
            
            D_delta = abs(C3*delta_pos(j)-Pos(ii,j)); % Equation (3.5)-part 3
            X3 = delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Pos(ii,j) = (X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    
    tt = tt+1;    
    conv_curve(tt) = alpha_fit;
    if mod(tt, verbose) == 0
        fprintf('GWO: Iteration %d    fitness: %4.3f \n', tt, alpha_fit);
    end
end

CT = toc(timer);       % Total computation time in seconds

fprintf('GWO: Final fitness: %4.3f \n', alpha_fit);

%% END OF GWO.m