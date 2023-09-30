%WOA Whale Optimization Algorithm
% [Leader_score,Leader_pos,conv_curve, CT] = WOA(data, target, No_P, fobj, 
%                                        N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili, A. Lewis (2016). 
%               The Whale Optimization Algorithm.  
%               Advances in Engineering Software, 95, 51-67. 
%               DOI: 10.1016/j.advengsoft.2016.01.008
% 
%     [Leader_score,Leader_pos] = WOA(data) applies feature selection on 
%     M-by-N matrix data with N examples and assuming last column as the 
%     classification target and returns the best fitness value Leader_score 
%     and 1-by-(M-1) matrix of feature positions Leader_pos.
%
%     [Leader_score,Leader_pos] = WOA(data, target) applies feature selection 
%     on M-by-N feature matrix data and 1-by-N target matrix target and 
%     returns the best fitness value Leader_score and 1-by-(M-1)matrix of 
%     feature positions Leader_pos.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Sep 2023)

function [Leader_score,Leader_pos,conv_curve, CT] = WOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

% initialize position vector and score for the leader
Leader_pos=zeros(1,N_Var);
Leader_score=inf; %change this to -inf for maximization problems

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the positions of search agents
Positions=bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));

conv_curve=zeros(1,Max_Iter);

tt=0;% Loop counter

% Main loop
while tt<Max_Iter
    for ii=1:size(Positions,1)
        
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(ii,:)>UB;
        Flag4lb=Positions(ii,:)<LB;
        Positions(ii,:)=(Positions(ii,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        
        % Calculate objective function for each search agent
        fitness=fobj(Positions(ii,:) > (LB+UB)/2, data, target);
        
        % Update the leader
        if fitness<Leader_score % Change this to > for maximization problem
            Leader_score=fitness; % Update alpha
            Leader_pos=Positions(ii,:);
        end
        
    end
    
    a=2-tt*((2)/Max_Iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+tt*((-1)/Max_Iter);
    
    % Update the Position of search agents 
    for ii=1:size(Positions,1)
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:size(Positions,2)
            
            if p<0.5   
                if abs(A)>=1
                    rand_leader_index = floor(No_P*rand()+1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-Positions(ii,j)); % Eq. (2.7)
                    Positions(ii,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                    
                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos(j)-Positions(ii,j)); % Eq. (2.1)
                    Positions(ii,j)=Leader_pos(j)-A*D_Leader;      % Eq. (2.2)
                end
                
            elseif p>=0.5
              
                distance2Leader=abs(Leader_pos(j)-Positions(ii,j));
                % Eq. (2.5)
                Positions(ii,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                
            end
            
        end
    end
    tt=tt+1;
    conv_curve(tt)=Leader_score;
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('WOA: Iteration %d    fitness: %4.3f \n', tt, Leader_score);
    end
end
CT = toc(timer);       % Total computation time in seconds
fprintf('WOA: Final fitness: %4.3f \n', Leader_score);

%% END OF WOA.m