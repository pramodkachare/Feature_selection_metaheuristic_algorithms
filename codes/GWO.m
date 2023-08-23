%GWO Grey Wolf Optimizer
% [Alpha_score,Alpha_pos,Convergence_curve, CT]=GWO(X, y, No_P, fobj,
% N_Var, Max_Iter, LB, UB, verbose)
%
%   Main paper: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
%               Grey wolf optimizer 
%               Advances in Engineering Software, 69, 46–61.
%               DOI: 10.1016/j.advengsoft.2013.12.007               
% 
%     [Alpha_score, Alpha_pos] = GWO(X) applies feature selection on M-by-N 
%     matrix X with N examples and assuming last column as the classification 
%     target and returns the best fitness value Alpha_score and 1-by-(M-1) 
%     logical  matrix of selected features Alpha_pos.
%
%     [Alpha_score, Alpha_pos] = GWO(X, y) applies feature selection on 
%     M-by-N feature matrix X and 1-by-N target matrix y and returns the 
%     best fitness value  Alpha_score and 1-by-(M-1) logical matrix of 
%     selected features Alpha_pos.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Aug 2023)

function [Alpha_score,Alpha_pos,Conv_curve, CT]=GWO(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
if nargin < 1
    error('MATLAB:notEnoughInputs', 'Please provide data for feature selection.');
end

if nargin < 2  % If only data is given, assume last column as target
    y = X(:, end);
    X = X(:, 1:end-1);
end

if nargin < 3  % Default 10 search agents
    No_P = 10;
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

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1, N_Var);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,N_Var);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,N_Var);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(No_P,N_Var,UB,LB);

Conv_curve=zeros(1,Max_Iter);

tt=0;% Loop counter

% Main loop
while tt<Max_Iter
    for i=1:size(Positions,1)  
        
       % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>UB;
        Flag4lb=Positions(i,:)<LB;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;               
        
        % Calculate objective function for each search agent
        fitness=fobj(Positions(i,:), X, y);
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    
    
    a=2-tt*((2)/Max_Iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    tt=tt+1;    
    Conv_curve(tt)=Alpha_score;
    if mod(tt, verbose)==0
        fprintf('GWO: Iteration %d    fitness: %4.3f \n', tt, Alpha_score);
    end
end

CT = toc(timer);       % Total computation time in seconds

fprintf('GWO: Final fitness: %4.3f \n', Alpha_score);


% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,N_Var,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,N_Var).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:N_Var
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end

