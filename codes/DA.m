% DA Dragonfly Algorithm
% [Best_F,Best_P,conv_curve, CT] = WOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili.
%               Dragonfly algorithm: a new meta-heuristic optimization 
%               technique for solving single-objective, discrete, and 
%               multi-objective problems. 
%               Neural Computing and Applications, 27, 1053–1073, 2016
%               DOI: 10.1007/s00521-015-1920-1 
% 
%     [Best_F,Best_P] = DA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F,Best_P] = DA(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Best_F and 1-by-(M-1) matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili (ali.mirjalili@gmail.com)                   
% Revised by : Pramod H. Kachare (Oct 2023)


function [Best_F,Best_P,conv_curve,CT]=DA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

conv_curve=zeros(1,Max_Iter);

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the positions
X = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));
DeltaX = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));

Fitness=zeros(1,No_P);

%The initial radius of gragonflies' neighbourhoods
Delta_max=(UB-LB)/10;

Food_fit = inf;
Food_pos = zeros(N_Var,1);

Enemy_fit = -inf;
Enemy_pos = zeros(N_Var,1);

for tt=1:Max_Iter
    r=(UB-LB)/4+((UB-LB)*(tt/Max_Iter)*2);
    
    w=0.9-tt*((0.9-0.4)/Max_Iter);
       
    my_c=0.1-tt*((0.1-0)/(Max_Iter/2));
    if my_c<0
        my_c=0;
    end
    
    s=2*rand*my_c; % Seperation weight
    a=2*rand*my_c; % Alignment weight
    c=2*rand*my_c; % Cohesion weight
    f=2*rand;      % Food attraction weight
    e=my_c;        % Enemy distraction weight
    
    for ii=1:No_P %Calculate all the objective values first
        Fitness(1,ii)=fobj(X(:,ii)' > (LB+UB)/2, data, target);
        if Fitness(1,ii)<Food_fit
            Food_fit=Fitness(1,ii);
            Food_pos=X(:,ii);
        end
        
        if Fitness(1,ii)>Enemy_fit
            if all(X(:,ii)<UB') && all( X(:,ii)>LB')
                Enemy_fit=Fitness(1,ii);
                Enemy_pos=X(:,ii);
            end
        end
    end
    
    for ii=1:No_P
        index=0;
        neighbours_no=0;
        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %find the neighbouring solutions
        for j=1:No_P
            Dist2Enemy=distance(X(:,ii),X(:,j));
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;
                neighbours_no=neighbours_no+1;
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end
        
        % Seperation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.1)
        S=zeros(N_Var,1);
        if neighbours_no>1
            for k=1:neighbours_no
                S=S+(Neighbours_X(:,k)-X(:,ii));
            end
            S=-S;
        else
            S=zeros(N_Var,1);
        end
        
        % Alignment%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.2)
        if neighbours_no>1
            A=(sum(Neighbours_DeltaX')')/neighbours_no;
        else
            A=DeltaX(:,ii);
        end
        
        % Cohesion%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.3)
        if neighbours_no>1
            C_temp=(sum(Neighbours_X')')/neighbours_no;
        else
            C_temp=X(:,ii);
        end
        
        C=C_temp-X(:,ii);
        
        % Attraction to food%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.4)
        Dist2Food=distance(X(:,ii),Food_pos(:,1));
        if all(Dist2Food<=r)
            F=Food_pos-X(:,ii);
        else
            F=0;
        end
        
        % Distraction from enemy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.5)
        Dist2Enemy=distance(X(:,ii),Enemy_pos(:,1));
        if all(Dist2Enemy<=r)
            Enemy=Enemy_pos+X(:,ii);
        else
            Enemy=zeros(N_Var,1);
        end
        
        for jj=1:N_Var
            if X(jj,ii)>UB(jj)
                X(jj,ii)=LB(jj);
                DeltaX(jj,ii)=rand;
            end
            if X(jj,ii)<LB(jj)
                X(jj,ii)=UB(jj);
                DeltaX(jj,ii)=rand;
            end
        end
        
        if any(Dist2Food>r)
            if neighbours_no>1
                for j=1:N_Var
                    DeltaX(j,ii)=w*DeltaX(j,ii)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,ii)>Delta_max(j)
                        DeltaX(j,ii)=Delta_max(j);
                    end
                    if DeltaX(j,ii)<-Delta_max(j)
                        DeltaX(j,ii)=-Delta_max(j);
                    end
                    X(j,ii)=X(j,ii)+DeltaX(j,ii);
                end
            else
                % Eq. (3.8)
                X(:,ii)=X(:,ii)+Levy(N_Var)'.*X(:,ii);
                DeltaX(:,ii)=0;
            end
        else
            for j=1:N_Var
                % Eq. (3.6)
                DeltaX(j,ii)=(a*A(j,1)+c*C(j,1)+s*S(j,1)+f*F(j,1)+e*Enemy(j,1)) + w*DeltaX(j,ii);
                if DeltaX(j,ii)>Delta_max(j)
                    DeltaX(j,ii)=Delta_max(j);
                end
                if DeltaX(j,ii)<-Delta_max(j)
                    DeltaX(j,ii)=-Delta_max(j);
                end
                X(j,ii)=X(j,ii)+DeltaX(j,ii);
            end 
        end
        
        Flag4ub=X(:,ii)>UB';
        Flag4lb=X(:,ii)<LB';
        X(:,ii)=(X(:,ii).*(~(Flag4ub+Flag4lb)))+UB'.*Flag4ub+LB'.*Flag4lb;
        
    end
    Best_F = Food_fit;
    Best_P = Food_pos;
    
    conv_curve(tt)=Best_F;
    % Display the iteration and best optimum obtained so far
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('DA: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
end
CT = toc(timer);       % Total computation time in seconds
fprintf('DA: Final fitness: %4.3f \n', Best_F);

%% END OF DA.m