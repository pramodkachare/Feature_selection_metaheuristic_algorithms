%CO Cheetah Optimizaer
% [Best_F,Best_P, conv_curve, CT] = CO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Akbari, M. A., Zare, M., Azizipanah-Abarghooee, R., 
%               Mirjalili, S., & Deriche, M. (2022). 
%               The cheetah optimizer: A nature-inspired metaheuristic 
%               algorithm for large-scale optimization problems. 
%               Scientific reports, 12(1), 10953. 
%               DOI: 10.1038/s41598-022-14338-z 
% 
%     [Best_F, Best_P] = RSA(data) applies feature selection on M-by-N
%     matrix data with N examples and assuming last column as the 
%     classification target and returns the best fitness value Best_F and 
%     1-by-(M-1) matrix of feature positions Best_P.
%
%     [Best_F, Best_P] = PSO(data, target) applies feature selection on 
%     M-by-N feature matrix data and 1-by-N target matrix target and returns
%     the best fitness value GBEST and 1-by-(M-1) matrix of feature 
%     positions Best_P.
%     
%     Example:
%
%
% Original Author: M. A. Akbari and M. Zare
%  Homepages:https://www.optim-app.com    &                         
%            https://seyedalimirjalili.com/co                        
% Revised by : Pramod H. Kachare (Aug 2023)


function [Best_F,Best_P, conv_curve, CT] = CO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

%Initialize the positions of search agents
if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

m = 2;                    % Number of search agenets in a group
        
%% Generate initial population of cheetahs (Algorithm 1, L#2)
empty_individual.Position = [];
empty_individual.Cost = [];
BestSol.Cost = inf;
pop = repmat(empty_individual,No_P,1);

for ii=1:No_P
    pop(ii).Position = LB+rand(1,N_Var).*(UB-LB);
    pop(ii).Cost = fobj(pop(ii).Position > (LB+UB)/2, data, target);
    if pop(ii).Cost < BestSol.Cost
        BestSol = pop(ii); % Initial leader position
    end
end

%% Initialization (Algorithm 1, L#3)
pop1 = pop;               % Population's initial home position
conv_curve = zeros(1,Max_Iter);            % Leader fittnes value in a current hunting period
X_best = BestSol;         % Prey solution sofar
Globest = conv_curve;       % Prey fittnes value sofar

%% Initial parameters
tt = 0;                    % Hunting time counter (Algorithm 1, L#4)
it = 1;                   % Iteration counter(Algorithm 1, L#5)
%         Max_Iter = N_Var*10000;           % Maximum number of iterations (Algorithm 1, L#6)
T = ceil(N_Var/10)*60;        % Hunting time (Algorithm 1, L#7)
FEs = 0;                  % Counter for function evaluations (FEs)
%% CO Main Loop
while it <= Max_Iter % Algorithm 1, L#8
    %  m = 1+randi (ceil(n/2)); 
    i0 = randi(No_P,1,m);    % select a random member of cheetahs (Algorithm 1, L#9)
    for kk = 1 : m % Algorithm 1, L#10
        ii = i0(kk);
        % neighbor agent selection (Algorithm 1, L#11)
        if kk == length(i0)
            a = i0(kk-1);
        else
            a = i0(kk+1);
        end

        X = pop(ii).Position;    % The current position of i-th cheetah
        X1 = pop(a).Position;   % The neighbor position
        Xb = BestSol.Position;  % The leader position
        Xbest = X_best.Position;% The pery position

%         kk=0;
        % Uncomment the follwing statements, it may improve the performance of CO
        if ii<=2 && tt>2 && tt>ceil(0.2*T+1) && abs(conv_curve(tt-2)-conv_curve(tt-ceil(0.2*T+1)))<=0.0001*Globest(tt-1)
            X = X_best.Position;
            kk = 0;
        elseif ii == 3
            X = BestSol.Position;
            kk = -0.1*rand*tt/T;
        else
            kk = 0.25;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        if mod(it,100)==0 || it==1
           xd = randperm(numel(X));
        end
        Z = X;

        %% Algorithm 1, L#12
        for j = xd % select arbitrary set of arrangements
            %% Algorithm 1, L#13
            r_Hat = randn;         % Randomization paameter, Equation (1)
            r1 = rand;
            if kk == 1              % The leader's step length (it is assumed that k==1 is associated to the leade number)
                alpha = 0.0001*tt/T.*(UB(j)-LB(j)); % Step length, Equation (1) %This can be updated by desired equation 
            else                   % The members' step length
                alpha = 0.0001*tt/T*abs(Xb(j)-X(j))+0.001.*round(double(rand>0.9));%member step length, Equation (1)%This can be updated by desired equation
            end

            r = randn;
            r_Check = abs(r).^exp(r/2).*sin(2*pi*r); % Turning factor, Equation (3)%This can be updated by desired equation
            beta = X1(j)-X(j);     % Interaction factor, Equation (3)

            h0 = exp(2-2*tt/T);

            H = abs(2*r1*h0-h0);

            %% Algorithm 1, L#14

            r2 = rand;
            r3 = kk+rand;

            %% Strategy selection mechanism
            if r2 <= r3              % Algorithm 1, L#15
                r4 = 3*rand;         % Algorithm 1, L#16
                if H > r4            % Algorithm 1, L#17
                    Z(j) = X(j)+r_Hat.^-1.*alpha;    % Search, Equation(1) (Algorithm 1, L#18)
                else
                    Z(j) = Xbest(j)+r_Check.*beta;    % Attack, Equation(3) (Algorithm 1, L#20)
                end
            else
                Z(j) = X(j);         % Sit&wait, Equation(2) (Algorithm 1, L#23)
            end
        end
        %% Update the solutions of member i (Algorithm 1, L#26)
        % Check the limits
        xx1=find(Z<LB);
        Z(xx1)=LB(xx1)+rand(1,numel(xx1)).*(UB(xx1)-LB(xx1));
        xx1=find(Z>UB);
        Z(xx1)=LB(xx1)+rand(1,numel(xx1)).*(UB(xx1)-LB(xx1));

        % Evaluate the new position
        NewSol.Position = Z;
        NewSol.Cost = fobj(NewSol.Position  > (LB+UB)/2, data, target);
        if NewSol.Cost < pop(ii).Cost
            pop(ii) = NewSol;
            if pop(ii).Cost < BestSol.Cost
                BestSol = pop(ii);
            end
        end
        FEs = FEs+1;
    end

    tt = tt+1; % (Algorithm 1, L#28)

    %% Leave the prey and go back home (Algorithm 1, L#29)
    if tt>T && tt-round(T)-1>=1 && tt>2
        if  abs(conv_curve(tt-1)-conv_curve(tt-round(T)-1))<=abs(0.01*conv_curve(tt-1))

            % Change the leader position (Algorithm 1, L#30)
            best = X_best.Position;
            j0=randi(N_Var,1,ceil(N_Var/10*rand));
            best(j0) = LB(j0)+rand(1,length(j0)).*(UB(j0)-LB(j0));
            BestSol.Cost = fobj(best  > (LB+UB)/2, data, target);
            BestSol.Position = best; % Leader's new position 
            FEs = FEs+1;

            i0 = randi(No_P,1,round(1*No_P));
            % Go back home, (Algorithm 1, L#30)
            pop(i0(No_P-m+1:No_P)) = pop1(i0(1:m)); % Some members back their initial positions 

            pop(ii) = X_best; % Substitude the member i by the prey (Algorithm 1, L#31)

            tt = 1; % Reset the hunting time (Algorithm 1, L#32)
        end
    end

    it = it +1; % Algorithm 1, L#34

    %% Update the prey (global best) position (Algorithm 1, L#35)
    if BestSol.Cost<X_best.Cost
        X_best=BestSol;
    end
    conv_curve(tt)=BestSol.Cost;
    Globest(1,tt)=X_best.Cost;

    %% Display
    if mod(it, verbose) == 0  %Print best particle details at fixed iters
        fprintf('CO: Iteration %d    fitness: %4.3f \n', it, Globest(end));
    end
end
Best_F = X_best; % Global best fitness
Best_P = Globest(end);  % Global best position
CT = toc(timer);       % Total computation time in seconds

fprintf('CO: Final fitness: %4.3f \n', Best_P);

%% END OF CO.m
