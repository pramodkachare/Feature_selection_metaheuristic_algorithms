%ALO Ant Lion Optimizer
% [Elt_AL_F, Elt_AL_P, conv_curve, CT] = ALO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili. 
%               The Ant Lion Optimizer. 
%               Advances in Engineering Software , 83, 80-98, 2015
%               DOI: 10.1016/j.advengsoft.2015.01.010 
% 
%     [Elt_AL_F, Elt_AL_P] = ALO(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Elt_AL_F and 1-by-(M-1) matrix of 
%     feature positions Elt_AL_P.
%
%     [Elt_AL_F, Elt_AL_P] = ALO(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Elt_AL_F and 1-by-(M-1) matrix of feature positions Elt_AL_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili                    
% Revised by : Pramod H. Kachare (Oct 2023)


function [Elt_AL_F,Elt_AL_P,conv_curve,CT]=ALO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

% Initialize the positions of antlions and ants
AL_pos=initialization(No_P,N_Var,UB,LB);
Ant_pos=initialization(No_P,N_Var,UB,LB);

% Initialize variables to save the position of elite, sorted antlions, 
% convergence curve, antlions fitness, and ants fitness
Sorted_AL=zeros(No_P,N_Var);

conv_curve=zeros(1,Max_Iter);
AL_fit=zeros(1,No_P);
Ants_fit=zeros(1,No_P);

% Calculate the fitness of initial antlions and sort them
for i=1:size(AL_pos,1)
    AL_fit(1,i)=fobj(AL_pos(i,:) > (LB+UB)/2, data, target); 
end

[sorted_antlion_fitness,sorted_indexes]=sort(AL_fit);
    
for newindex=1:No_P
     Sorted_AL(newindex,:)=AL_pos(sorted_indexes(newindex),:);
end
    
Elt_AL_P=Sorted_AL(1,:);
Elt_AL_F=sorted_antlion_fitness(1);

% Main loop start from the second iteration since the first iteration 
% was dedicated to calculating the fitness of antlions
tt=2; 
while tt<Max_Iter+1
    
    % This for loop simulate random walks
    for i=1:size(Ant_pos,1)
        % Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
        Rolette_index=RouletteWheelSelection(1./sorted_antlion_fitness);
        if Rolette_index==-1  
            Rolette_index=1;
        end
      
        % RA is the random walk around the selected antlion by rolette wheel
        RA=Random_walk_around_antlion(N_Var,Max_Iter,LB,UB, Sorted_AL(Rolette_index,:),tt);
        
        % RA is the random walk around the elite (best antlion so far)
        [RE]=Random_walk_around_antlion(N_Var,Max_Iter,LB,UB, Elt_AL_P(1,:),tt);
        
        Ant_pos(i,:)= (RA(tt,:)+RE(tt,:))/2; % Equation (2.13) in the paper          
    end
    
    for i=1:size(Ant_pos,1)  
        
        % Boundar checking (bring back the antlions of ants inside search
        % space if they go beyoud the boundaries
        Flag4ub=Ant_pos(i,:)>UB;
        Flag4lb=Ant_pos(i,:)<LB;
        Ant_pos(i,:)=(Ant_pos(i,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;  
        
        Ants_fit(1,i)=fobj(Ant_pos(i,:) > (LB+UB)/2, data, target);        
       
    end
    
    % Update antlion positions and fitnesses based of the ants (if an ant 
    % becomes fitter than an antlion we assume it was cought by the antlion  
    % and the antlion update goes to its position to build the trap)
    double_population=[Sorted_AL;Ant_pos];
    double_fitness=[sorted_antlion_fitness Ants_fit];
        
    [double_fitness_sorted, I]=sort(double_fitness);
    double_sorted_population=double_population(I,:);
        
    AL_fit=double_fitness_sorted(1:No_P);
    Sorted_AL=double_sorted_population(1:No_P,:);
        
    % Update the position of elite if any antlinons becomes fitter than it
    if AL_fit(1)<Elt_AL_F 
        Elt_AL_P=Sorted_AL(1,:);
        Elt_AL_F=AL_fit(1);
    end
      
    % Keep the elite in the population
    Sorted_AL(1,:)=Elt_AL_P;
    AL_fit(1)=Elt_AL_F;
  
    % Update the convergence curve
    conv_curve(tt)=Elt_AL_F;

    if mod(tt, verbose) == 0  %Print best fitness details at fixed iters
        fprintf('ALO: Iteration %d    fitness: %4.3f \n', tt, Elt_AL_F);
    end

    tt=tt+1; 
end
CT = toc(timer);       % Total computation time in seconds
fprintf('ALO: Final fitness: %4.3f \n', Elt_AL_F);

%% END OF ALO.m