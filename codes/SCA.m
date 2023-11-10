%SCA Sine Cosine Algorithm
% [Best_F,Best_P,conv_curve, CT] = SCA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili.
%               SCA: a sine cosine algorithm for solving optimization problems. 
%               Knowledge-based systems, 96, 120-133, 2016
%               DOI: 10.1016/j.knosys.2015.12.022 
% 
%     [Best_F,Best_P] = SCA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F,Best_P] = SCA(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Best_F and 1-by-(M-1) matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili (ali.mirjalili@gmail.com)                   
% Revised by : Pramod H. Kachare (Nov 2023)


% To run SCA: [Best_score,Best_pos,cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)

function [Best_F,Best_P,conv_curve,CT]=SCA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the set of random solutions
X = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));

Best_P=zeros(1,N_Var);
Best_F=inf;

conv_curve=zeros(1,Max_Iter);
obj_val = zeros(1,size(X,1));

% Calculate the fitness of the first set and find the best one
for ii=1:size(X,1)
    obj_val(1,ii)=fobj(X(ii,:));
    if ii==1
        Best_P=X(ii,:);
        Best_F=obj_val(1,ii);
    elseif obj_val(1,ii)<Best_F
        Best_P=X(ii,:);
        Best_F=obj_val(1,ii);
    end
    
%     All_objective_values(1,ii)=obj_val(1,ii);
end
conv_curve(1) = Best_F;

%Main loop
tt=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness
while tt<=Max_Iter+1
    % Eq. (3.4)
    a = 2;
%     Max_Iter = Max_Iter;
    r1=a-tt*((a)/Max_Iter); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for ii=1:size(X,1) % in i-th solution
        for jj=1:size(X,2) % in j-th dimension
            % Update r2, r3, and r4 for Eq. (3.3)
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(ii,jj)= X(ii,jj)+(r1*sin(r2)*abs(r3*Best_P(jj)-X(ii,jj)));
            else
                % Eq. (3.2)
                X(ii,jj)= X(ii,jj)+(r1*cos(r2)*abs(r3*Best_P(jj)-X(ii,jj)));
            end
        end
    end
    
    for ii=1:size(X,1)
        % Check if solutions go outside the search spaceand bring them back
        Flag4ub=X(ii,:)>UB;
        Flag4lb=X(ii,:)<LB;
        X(ii,:)=(X(ii,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        
        % Calculate the objective values
        obj_val(1,ii)=fobj(X(ii,:));
        
        % Update the destination if there is a better solution
        if obj_val(1,ii)<Best_F
            Best_P=X(ii,:);
            Best_F=obj_val(1,ii);
        end
    end
    
    conv_curve(tt)=Best_F;
    
    % Display the iteration and best optimum obtained so far
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('SCA: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    
    % Increase the iteration counter
    tt=tt+1;
end

CT = toc(timer);       % Total computation time in seconds
fprintf('SCA: Final fitness: %4.3f \n', Best_F);

%% END OF SCA.m