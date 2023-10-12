%GOA Grasshopper Optimization Algorithm
% [Best_F,Best_P,conv_curve, CT] = GOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%  Main paper: S. Saremi, S. Mirjalili, & A. Lewis                          
%              Grasshopper Optimisation Algorithm: Theory and Application.
%              Advances in Engineering Software , 105, 30-47, 2017              
%              DOI: 10.1016/j.advengsoft.2017.01.004   
%                                                                         
%     [Best_F,Best_P] = GOA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F,Best_P] = GOA(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Best_F and 1-by-(M-1) matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili (ali.mirjalili@gmail.com)                   
% Revised by : Pramod H. Kachare (Oct 2023)


function [Best_F, Best_P, conv_curve, CT] = GOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% Trajectories,fit_history,pos_history
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

flag=0;   % Flag for even number of variables
if (rem(N_Var,2)~=0)
    fprintf('GOA should be run with a even number of variables.\n');
    fprintf('Appending a variable ...\n')
    N_Var = N_Var+1;
    UB = [UB; 100];
    LB = [LB; -100];
    flag=1;  % Flag for odd number of variables
end

%Start timer
timer = tic();

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the population of grasshoppers
GH_pos = bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));
GH_fit = zeros(1,No_P);

fit_history  = zeros(No_P,Max_Iter);
pos_history  = zeros(No_P,Max_Iter,N_Var);
conv_curve   = zeros(1,Max_Iter);
Trajectories = zeros(No_P,Max_Iter);

cMax=1;
cMin=0.00004;
%Calculate the fitness of initial grasshoppers

for ii=1:size(GH_pos,1)
    if flag == 1
        GH_fit(1,ii)=fobj(GH_pos(ii,1:end-1));
    else
        GH_fit(1,ii)=fobj(GH_pos(ii,:));
    end
    fit_history(ii,1)=GH_fit(1,ii);
    pos_history(ii,1,:)=GH_pos(ii,:);
    Trajectories(:,1)=GH_pos(:,1);
end

[sorted_fit,sorted_ind]=sort(GH_fit);

% Find the best grasshopper (target) in the first population 
for newindex=1:No_P
    Sorted_GH(newindex,:)=GH_pos(sorted_ind(newindex),:);
end

Best_P=Sorted_GH(1,:);
Best_F=sorted_fit(1);

% Main loop
tt=2; % Start from the second iteration since the first iteration was dedicated to calculating the fitness of antlions
while tt<Max_Iter+1
    
    c=cMax-tt*((cMax-cMin)/Max_Iter); % Eq. (2.8) in the paper
    
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      for ii=1:size(GH_pos,1)
        temp= GH_pos';
       % for k=1:2:dim  
            S_i=zeros(N_Var,1);
            for jj=1:No_P
                if ii~=jj
                    Dist=distance(temp(:,jj), temp(:,ii)); % Calculate the distance between two grasshoppers
                    
                    r_ij_vec=(temp(:,jj)-temp(:,ii))/(Dist+eps); % xj-xi/dij in Eq. (2.7)
                    xj_xi=2+rem(Dist,2); % |xjd - xid| in Eq. (2.7) 
                    
                    s_ij=((UB - LB)*c/2)*S_func(xj_xi).*r_ij_vec; % The first part inside the big bracket in Eq. (2.7)
                    S_i=S_i+s_ij;
                end
            end
            S_i_total = S_i;
            
      %  end
        
        X_new = c * S_i_total'+ (Best_P); % Eq. (2.7) in the paper      
        GH_pos_temp(ii,:)=X_new'; 
      end
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GrassHopperPositions
    GH_pos=GH_pos_temp;
    
    for ii=1:size(GH_pos,1)
        % Relocate grasshoppers that go outside the search space 
        Tp=GH_pos(ii,:)>UB';Tm=GH_pos(ii,:)<LB';GH_pos(ii,:)=(GH_pos(ii,:).*(~(Tp+Tm)))+UB'.*Tp+LB'.*Tm;
        
        % Calculating the objective values for all grasshoppers
        if flag == 1
            GH_fit(1,ii)=fobj(GH_pos(ii,1:end-1));
        else
            GH_fit(1,ii)=fobj(GH_pos(ii,:));
        end
        fit_history(ii,tt)=GH_fit(1,ii);
        pos_history(ii,tt,:)=GH_pos(ii,:);
        
        Trajectories(:,tt)=GH_pos(:,1);
        
        % Update the target
        if GH_fit(1,ii)<Best_F
            Best_P=GH_pos(ii,:);
            Best_F=GH_fit(1,ii);
        end
    end
        
    conv_curve(tt)=Best_F;
    % Display the iteration and best optimum obtained so far
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('GOA: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    tt = tt + 1;
end

if (flag==1)
    Best_P = Best_P(1:N_Var-1);
end

CT = toc(timer);       % Total computation time in seconds
fprintf('GOA: Final fitness: %4.3f \n', Best_F);

%% END OF GOA.m