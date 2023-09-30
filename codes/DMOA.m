%DMOA Dwarf Mongoose Optimization Algorithm
% [BestF, BestP, conv_curve, CT] = DMOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022).
%               Dwarf mongoose optimization algorithm.  
%               Computer methods in applied mechanics & engineering, 391, 114570. 
%               DOI: 10.1016/j.cma.2022.114570

% 
%     [BestF, BestP] = DMOA(data) applies feature selection on M-by-N matrix
%     data with N examples and assuming last column as the classification target 
%     and returns the best fitness value BestF and 1-by-(M-1) matrix of 
%     feature positions BestP.
%
%     [BestF, BestP] = DMOA(data, target) applies feature selection on M-by-N 
%     feature matrix data and 1-by-N target matrix target and returns the 
%     best fitness value BestF and 1-by-(M-1)matrix of feature positions BestP.
%     
%     Example:
%
%
% Original Author: Jeffrey O. Agushaka, Absalom E. Ezugwu & Laith Abualigah 
%                  (EzugwuA@ukzn.ac.za)
% Revised by : Pramod H. Kachare (Sep 2023)


function [BestF,BestP,conv_curve,CT] = DMOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

VarSize=[1 N_Var];   % Decision Variables Matrix Size

% ABC Settings
nBabysitter= 3;         % Number of babysitters

nAlphaGroup=No_P-nBabysitter;         % Number of Alpha group

nScout=nAlphaGroup;         % Number of Scouts

L=round(0.6*N_Var*nBabysitter); % Babysitter Exchange Parameter 

peep=2;             % Alpha female痴 vocalization 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Empty Mongoose Structure
empty_mongoose.Position=[];
empty_mongoose.Cost=[];

% Initialize Population Array
pop=repmat(empty_mongoose,nAlphaGroup,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;
tau=inf;
Iter=1;
sm=inf(nAlphaGroup,1);

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end


% Create Initial Population
for ii=1:nAlphaGroup
    
    pop(ii).Position=bsxfun(@plus, LB, bsxfun(@times, rand(1,N_Var), (UB-LB)));
    pop(ii).Cost=fobj(pop(ii).Position > (LB+UB)/2, data, target);
    if pop(ii).Cost<=BestSol.Cost
        BestSol=pop(ii);
    end
end

% Abandonment Counter
C=zeros(nAlphaGroup,1);
CF=(1-Iter/Max_Iter)^(2*Iter/Max_Iter);

% Array to Hold Best Cost Values
conv_curve=zeros(Max_Iter,1);

%% DMOA Main Loop

for tt=1:Max_Iter
    
    % Alpha group
     F=zeros(nAlphaGroup,1);
     MeanCost = mean([pop.Cost]);
    for ii=1:nAlphaGroup
        
        % Calculate Fitness Values and Selection of Alpha
        F(ii) = exp(-pop(ii).Cost/MeanCost); % Convert Cost to Fitness
    end
        P=F/sum(F);
      % Foraging led by Alpha female
    for mm = 1:nAlphaGroup
        % Select Alpha female
        ii=RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to Alpha
        K=[1:ii-1 ii+1:nAlphaGroup];
        k=K(randi([1 numel(K)]));
        
        % Define Vocalization Coeff.
        phi=(peep/2)*unifrnd(-1,+1,VarSize);
        
        % New Mongoose Position
        newpop.Position=pop(ii).Position+phi.*(pop(ii).Position-pop(k).Position);
        
        % Evaluation
        newpop.Cost=fobj(newpop.Position > (LB+UB)/2, data, target);
        
        % Comparision
        if newpop.Cost<=pop(ii).Cost
            pop(ii)=newpop;
        else
            C(ii)=C(ii)+1;
        end
        
    end   
    
    % Scout group
    for ii=1:nScout
        
        % Choose k randomly, not equal to i
        K=[1:ii-1 ii+1:nAlphaGroup];
        k=K(randi([1 numel(K)]));
        
        % Define Vocalization Coeff.
        phi=(peep/2)*unifrnd(-1,+1,VarSize);
        
        % New Mongoose Position
        newpop.Position=pop(ii).Position+phi.*(pop(ii).Position-pop(k).Position);
        
        % Evaluation
        newpop.Cost=fobj(newpop.Position > (LB+UB)/2, data, target);
        
        % Sleeping mould
        sm(ii)=(newpop.Cost-pop(ii).Cost)/max(newpop.Cost,pop(ii).Cost);
        
        % Comparision
        if newpop.Cost<=pop(ii).Cost
            pop(ii)=newpop;
        else
            C(ii)=C(ii)+1;
        end
        
    end    
    % Babysitters
    for ii=1:nBabysitter
        if C(ii)>=L
            pop(ii).Position=unifrnd(LB,UB,VarSize);
            pop(ii).Cost=fobj(pop(ii).Position > (LB+UB)/2, data, target);
            C(ii)=0;
        end
    end    
     % Update Best Solution Ever Found
    for ii=1:nAlphaGroup
        if pop(ii).Cost<=BestSol.Cost
            BestSol=pop(ii);
        end
    end    
        
   % Next Mongoose Position
   newtau=mean(sm);
   for ii=1:nScout
        M=(pop(ii).Position.*sm(ii))/pop(ii).Position;
        if newtau>tau
           newpop.Position=pop(ii).Position-CF*phi*rand.*(pop(ii).Position-M);
        else
           newpop.Position=pop(ii).Position+CF*phi*rand.*(pop(ii).Position-M);
        end
        tau=newtau;
   end
       
   % Update Best Solution Ever Found
    for ii=1:nAlphaGroup
        if pop(ii).Cost<=BestSol.Cost
            BestSol=pop(ii);
        end
    end
    
    % Store Best Cost Ever Found
    conv_curve(tt)=BestSol.Cost;
    BestF=BestSol.Cost;
    BestP=BestSol.Position;
    
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('DMOA: Iteration %d    fitness: %4.3f \n', tt, BestSol.Cost);
    end
end
CT = toc(timer);       % Total computation time in seconds
fprintf('DMOA: Final fitness: %4.3f \n', BestSol.Cost);

%% END OF FLA.m