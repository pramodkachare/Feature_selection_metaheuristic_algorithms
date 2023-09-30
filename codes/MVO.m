%MVO Multi-Verse Optimizer (MVO)
% [Best_uni_Inf_rate,Best_uni,conv_curve, CT] = MVO(data, target, No_P, fobj,
%                                       N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Hatamlou (2016). 
%               Multi-Verse Optimizer: a nature-inspired algorithm for 
%               global optimization. 
%               Neural Computing and Applications, 27, 495–513. 
%               DOI: 10.1007/s00521-015-1870-7
% 
%     [Best_uni_Inf_rate,Best_uni] = MVO(data) applies feature selection on 
%     M-by-N matrix data with N examples and assuming last column as the 
%     classification target and returns the best fitness value Best_uni_Inf_rate 
%     and 1-by-(M-1) matrix of feature positions Best_uni.
%
%     [Best_uni_Inf_rate,Best_uni] = MVO(data, target) applies feature 
%     selection on M-by-N feature matrix data and 1-by-N target matrix target 
%     and returns the best fitness value Best_uni_Inf_rate and 1-by-(M-1) 
%     matrix of feature positions Best_uni.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Sep 2023)

function [Best_uni_Inf_rate,Best_uni,conv_curve, CT] = MVO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

%Two variables for saving the position and inflation rate (fitness) of the best universe
Best_uni=zeros(1,N_Var);
Best_uni_Inf_rate=inf;

%Initialize the positions of universes
Universes=initialization(No_P,N_Var,UB,LB);

%Minimum and maximum of Wormhole Existence Probability (min and max in
% Eq.(3.3) in the paper
WEP_Max=1;
WEP_Min=0.2;

conv_curve=zeros(1,Max_Iter);

%Iteration(time) counter
Time=1;

%Main loop
while Time<Max_Iter+1
    
    %Eq. (3.3) in the paper
    WEP=WEP_Min+Time*((WEP_Max-WEP_Min)/Max_Iter);
    
    %Travelling Distance Rate (Formula): Eq. (3.4) in the paper
    TDR=1-((Time)^(1/6)/(Max_Iter)^(1/6));
    
    %Inflation rates (I) (fitness values)
    Inflation_rates=zeros(1,size(Universes,1));
    
    for i=1:size(Universes,1)
        
        %Boundary checking (to bring back the universes inside search
        % space if they go beyoud the boundaries
        Flag4ub=Universes(i,:)>UB;
        Flag4lb=Universes(i,:)<LB;
        Universes(i,:)=(Universes(i,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        
        %Calculate the inflation rate (fitness) of universes
        Inflation_rates(1,i)=fobj(Universes(i,:), data, target);
        
        %Elitism
        if Inflation_rates(1,i)<Best_uni_Inf_rate
            Best_uni_Inf_rate=Inflation_rates(1,i);
            Best_uni=Universes(i,:);
        end
        
    end
    
    [sorted_Inflation_rates,sorted_indexes]=sort(Inflation_rates);
    
    for newindex=1:No_P
        Sorted_universes(newindex,:)=Universes(sorted_indexes(newindex),:);
    end
    
    %Normaized inflation rates (NI in Eq. (3.1) in the paper)
    normalized_sorted_Inflation_rates=normr(sorted_Inflation_rates);
    
    Universes(1,:)= Sorted_universes(1,:);
    
    %Update the Position of universes
    for i=2:size(Universes,1)%Starting from 2 since the firt one is the elite
        Back_hole_index=i;
        for j=1:size(Universes,2)
            r1=rand();
            if r1<normalized_sorted_Inflation_rates(i)
                White_hole_index=RouletteWheelSelection(-sorted_Inflation_rates);% for maximization problem -sorted_Inflation_rates should be written as sorted_Inflation_rates
                if White_hole_index==-1
                    White_hole_index=1;
                end
                %Eq. (3.1) in the paper
                Universes(Back_hole_index,j)=Sorted_universes(White_hole_index,j);
            end
            
            if (size(LB,2)==1)
                %Eq. (3.2) in the paper if the boundaries are all the same
                r2=rand();
                if r2<WEP
                    r3=rand();
                    if r3<0.5
                        Universes(i,j)=Best_uni(1,j)+TDR*((UB-LB)*rand+LB);
                    end
                    if r3>0.5
                        Universes(i,j)=Best_uni(1,j)-TDR*((UB-LB)*rand+LB);
                    end
                end
            end
            
            if (size(LB,2)~=1)
                %Eq. (3.2) in the paper if the upper and lower bounds are
                %different for each variables
                r2=rand();
                if r2<WEP
                    r3=rand();
                    if r3<0.5
                        Universes(i,j)=Best_uni(1,j)+TDR*((UB(j)-LB(j))*rand+LB(j));
                    end
                    if r3>0.5
                        Universes(i,j)=Best_uni(1,j)-TDR*((UB(j)-LB(j))*rand+LB(j));
                    end
                end
            end
            
        end
    end
    
    %Update the convergence curve
    conv_curve(Time)=Best_uni_Inf_rate;
    
    %Print the best universe details after every 50 iterations
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('MVO: Iteration %d    fitness: %4.3f \n', Time, Best_uni_Inf_rate);
    end
    Time=Time+1;
end
CT = toc(timer);       % Total computation time in seconds
fprintf('MVO: Final fitness: %4.3f \n', Best_uni_Inf_rate);



% Please note that these codes have been taken from:
%http://playmedusa.com/blog/roulette-wheel-selection-algorithm-in-matlab-2/

% ---------------------------------------------------------
% Roulette Wheel Selection Algorithm. A set of weights
% represents the probability of selection of each
% individual in a group of choices. It returns the index
% of the chosen individual.
% Usage example:
% fortune_wheel ([1 5 3 15 8 1])
%    most probable result is 4 (weights 15)
% ---------------------------------------------------------

function choice = RouletteWheelSelection(weights)
  accumulation = cumsum(weights);
  p = rand() * accumulation(end);
  chosen_index = -1;
  for index = 1 : length(accumulation)
    if (accumulation(index) > p)
      chosen_index = index;
      break;
    end
  end
  choice = chosen_index;

% This function creates the first random population

function X=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    X=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        X(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end

