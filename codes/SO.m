%SO Snake Optimizer (SO) 
% [Best_F, Best_P, conv_curve, CT] = SO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Hashim, F. A., & Hussien, A. G. (2022). 
%               Snake Optimizer: A novel meta-heuristic optimization algorithm. 
%               Knowledge-Based Systems, 242, 108320. 
%               DOI: 10.1016/j.knosys.2022.108320 
% 
%     [Best_F, Best_P] = SO(data) applies feature selection on M-by-N
%     matrix data with N examples and assuming last column as the 
%     classification target and returns the best fitness value Best_F and 
%     1-by-(M-1) matrix of feature positions Best_P.
%
%     [Best_F, Best_P] = SO(data, target) applies feature selection on 
%     M-by-N feature matrix data and 1-by-N target matrix target and returns 
%     the best fitness value GBEST and 1-by-(M-1) logical matrix of selected 
%     features GPOS.
%     
%     Example:
%
%
% Original Author: angelinbeni
% (github.com/angelinbeni/Remora_Optimization_Algorithm)
% Revised by : Pramod H. Kachare (Sep 2023)

function [Best_F, Best_P, conv_curve, CT] = SO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

%initial 
vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

% Generate initial population
X=bsxfun(@plus, bsxfun(@times, rand(No_P, N_Var), (UB-LB)), LB);

fitness = zeros(1, No_P);
for i=1:No_P
    fitness(i)=fobj(X(i,:)> (LB+UB)/2, data, target);   
end
[GYbest, gbest] = min(fitness);
Best_P = X(gbest,:);
conv_curve=zeros(1,Max_Iter);

%Diving the swarm into two equal groups males and females
Nm=round(No_P/2);%eq.(2&3)
Nf=No_P-Nm;
Xm=X(1:Nm,:);
Xf=X(Nm+1:No_P,:);
fitness_m=fitness(1:Nm);
fitness_f=fitness(Nm+1:No_P);
[fitnessBest_m, gbest1] = min(fitness_m);
Xbest_m = Xm(gbest1,:);
[fitnessBest_f, gbest2] = min(fitness_f);
Xbest_f = Xf(gbest2,:);

Xnewm = zeros(Nm, N_Var);
Xnewf = zeros(Nf, N_Var);

for tt = 1:Max_Iter
    Temp=exp(-((tt)/Max_Iter));  %eq.(4)
  Q=C1*exp(((tt-Max_Iter)/(Max_Iter)));%eq.(5)
    if Q>1
        Q=1;    
    end
    % Exploration Phase (no Food)
if Q<Threshold
    for i=1:Nm
        for j=1:1:N_Var
            rand_leader_index = floor(Nm*rand()+1);
            X_randm = Xm(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Am=exp(-fitness_m(rand_leader_index)/(fitness_m(i)+eps));%eq.(7)
            Xnewm(i,j)=X_randm(j)+Flag*C2*Am*((UB(1, j)-LB(1, j))*rand+LB(1,j));%eq.(6)
        end
    end
    for i=1:Nf
        for j=1:1:N_Var
            rand_leader_index = floor(Nf*rand()+1);
            X_randf = Xf(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Af=exp(-fitness_f(rand_leader_index)/(fitness_f(i)+eps));%eq.(9)
            Xnewf(i,j)=X_randf(j)+Flag*C2*Af*((UB(1,j)-LB(1,j))*rand+LB(1,j));%eq.(8)
        end
    end
else %Exploitation Phase (Food Exists)
    if Temp>Thresold2  %hot
        for i=1:Nm
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:N_Var
                Xnewm(i,j)=Best_P(j)+C3*Flag*Temp*rand*(Best_P(j)-Xm(i,j));%eq.(10)
            end
        end
        for i=1:Nf
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:N_Var
                Xnewf(i,j)=Best_P(j)+Flag*C3*Temp*rand*(Best_P(j)-Xf(i,j));%eq.(10)
            end
        end
    else %cold
        if rand>0.6 %fight
            for i=1:Nm
                for j=1:1:N_Var
                    FM=exp(-(fitnessBest_f)/(fitness_m(i)+eps));%eq.(13)
                    Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Q*Xbest_f(j)-Xm(i,j));%eq.(11)
                    
                end
            end
            for i=1:Nf
                for j=1:1:N_Var
                    FF=exp(-(fitnessBest_m)/(fitness_f(i)+eps));%eq.(14)
                    Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Q*Xbest_m(j)-Xf(i,j));%eq.(12)
                end
            end
        else%mating
            for i=1:Nm
                for j=1:1:N_Var
                    Mm=exp(-fitness_f(i)/(fitness_m(i)+eps));%eq.(17)
                    Xnewm(i,j)=Xm(i,j) +C3*rand*Mm*(Q*Xf(i,j)-Xm(i,j));%eq.(15
                end
            end
            for i=1:Nf
                for j=1:1:N_Var
                    Mf=exp(-fitness_m(i)/(fitness_f(i)+eps));%eq.(18)
                    Xnewf(i,j)=Xf(i,j) +C3*rand*Mf*(Q*Xm(i,j)-Xf(i,j));%eq.(16)
                end
            end
            flag_index = floor(2*rand()+1);
            egg=vec_flag(flag_index);
            if egg==1
                [~, gworst] = max(fitness_m);
                Xnewm(gworst,:)=LB+rand*(UB-LB);%eq.(19)
                [~, gworst] = max(fitness_f);
                Xnewf(gworst,:)=LB+rand*(UB-LB);%eq.(20)
            end
        end
    end
end
    for j=1:Nm
         Flag4ub=Xnewm(j,:)>UB;
         Flag4lb=Xnewm(j,:)<LB;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        y = fobj(Xnewm(j,:)> (LB+UB)/2, data, target);
        if y<fitness_m(j)
            fitness_m(j)=y;
            Xm(j,:)= Xnewm(j,:);
        end
    end
    
    [Ybest1,gbest1] = min(fitness_m);
    
    for j=1:Nf
         Flag4ub=Xnewf(j,:)>UB;
         Flag4lb=Xnewf(j,:)<LB;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        y = fobj(Xnewf(j,:)> (LB+UB)/2, data, target);
        if y<fitness_f(j)
            fitness_f(j)=y;
            Xf(j,:)= Xnewf(j,:);
        end
    end
    
    [Ybest2,gbest2] = min(fitness_f);
    
    if Ybest1<fitnessBest_m
        Xbest_m = Xm(gbest1,:);
        fitnessBest_m=Ybest1;
    end
    if Ybest2<fitnessBest_f
        Xbest_f = Xf(gbest2,:);
        fitnessBest_f=Ybest2;
        
    end
    if Ybest1<Ybest2
        conv_curve(tt)=min(Ybest1);
    else
        conv_curve(tt)=min(Ybest2);
        
    end
    if fitnessBest_m<fitnessBest_f
        GYbest=fitnessBest_m;
        Best_P=Xbest_m;
    else
        GYbest=fitnessBest_f;
        Best_P=Xbest_f;
    end
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
      fprintf('SO: Iteration %d    fitness: %4.3f \n', tt,  min(fitnessBest_m, fitnessBest_f));
    end
end
CT = toc(timer);       % Total computation time in seconds
Best_F = GYbest;
fprintf('SO: Final fitness: %4.3f \n', Best_F);

%% END OF ROA.m



