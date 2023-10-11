%MFOA Moth-Flame Optimization Algorithm
% [Best_F,Best_P,conv_curve, CT] = WOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili.
%               Moth-Flame Optimization Algorithm: A Novel Nature-inspired Heuristic Paradigm. 
%               Knowledge-Based Systems , 89, 228-249, 2015
%               DOI: 10.1016/j.knosys.2015.07.006 
% 
%     [Best_F,Best_P] = ALO(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F,Best_P] = ALO(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Best_F and 1-by-(M-1) matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili (ali.mirjalili@gmail.com)                   
% Revised by : Pramod H. Kachare (Oct 2023)


function [Best_F,Best_P,conv_curve, CT]=MFO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

%Initialize the positions of moths
Moth_pos = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));
Moth_fit = zeros(1, No_P);

conv_curve=zeros(1,Max_Iter);

tt=1;

% Main loop
while tt<Max_Iter+1
    
    % Number of flames Eq. (3.14) in the paper
    Flame_no=round(No_P-tt*((No_P-1)/Max_Iter));
    
    for i=1:No_P  
        % Check if moths go out of the search spaceand bring it back
        Flag4ub=Moth_pos(i,:)>UB;
        Flag4lb=Moth_pos(i,:)<LB;
        Moth_pos(i,:)=(Moth_pos(i,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;  
        
        % Calculate the fitness of moths
        Moth_fit(1,i)=fobj(Moth_pos(i,:) > (LB+UB)/2, data, target);          
    end
       
    if tt==1
        % Sort the first population of moths
        [fitness_sorted, I]=sort(Moth_fit);
        sorted_population=Moth_pos(I,:);
        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    else        
        % Sort the moths
        double_population=[previous_population;best_flames];
        double_fitness=[previous_fitness best_flame_fitness];
        
        [double_fit_sorted, I]=sort(double_fitness);
        double_sorted_population=double_population(I,:);
        
        fitness_sorted=double_fit_sorted(1:No_P);
        sorted_population=double_sorted_population(1:No_P,:);
        
        % Update the flames
        best_flames=sorted_population;
        best_flame_fitness=fitness_sorted;
    end
    
    % Update the position best flame obtained so far
    Best_F=fitness_sorted(1);
    Best_P=sorted_population(1,:);
      
    previous_population=Moth_pos;
    previous_fitness=Moth_fit;
    
    % a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a=-1+tt*((-1)/Max_Iter);
    
    for i=1:No_P    
        for j=1:N_Var
            if i<=Flame_no % Update the position of the moth with respect to its corresponsing flame
                
                % D in Eq. (3.13)
                dist_to_flame=abs(sorted_population(i,j)-Moth_pos(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                Moth_pos(i,j)=dist_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(i,j);
            end
            
            if i>Flame_no % Upaate the position of the moth with respct to one flame
                
                % Eq. (3.13)
                dist_to_flame=abs(sorted_population(i,j)-Moth_pos(i,j));
                b=1;
                t=(a-1)*rand+1;
                
                % Eq. (3.12)
                Moth_pos(i,j)=dist_to_flame*exp(b.*t).*cos(t.*2*pi)+sorted_population(Flame_no,j);
            end            
        end        
    end
    
    conv_curve(tt)=Best_F;
    
    % Display the iteration and best optimum obtained so far
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('MFO: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    tt=tt+1; 
end
CT = toc(timer);       % Total computation time in seconds
fprintf('MFO: Final fitness: %4.3f \n', Best_F);

%% END OF MFO.m