%RSA Reptile Search Algorithm (RSA)
% [Best_F,Best_P, conv_curve, CT] = RSA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Abualigah, L., Abd Elaziz, M., Sumari, P., Geem, Z. W., 
%               & Gandomi, A. H. (2022). 
%               Reptile Search Algorithm (RSA): A nature-inspired 
%               meta-heuristic optimizer
%               Expert Systems with Applications, 191, 116158.
%               DOI: 10.1016/j.eswa.2021.116158 
% 
%     [Best_F, Best_P] = RSA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) logical matrix
%     of selected features Best_P.
%
%     [Best_F, Best_P] = PSO(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     GBEST and 1-by-(M-1) logical matrix of selected features GPOS.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Aug 2023)

function [Best_F, Best_P, conv_curve, CT]=RSA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
if nargin < 1
    error('MATLAB:notEnoughInputs', 'Please provide data for feature selection.');
end

if nargin < 2  % If only data is given, assume last column as target
    y = X(:, end);
    X = X(:, 1:end-1);
end

if nargin < 3  % Default 10 search agents
    No_P = 10;
end

if nargin < 4
    fobj = str2func('split_fitness'); % Apply feature selection
end

if nargin < 5
    N_Var = size(X, 2); % Apply feature selection on columns of X
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

Best_P=zeros(1,N_Var);           % best positions
Best_F=inf;                    % best fitness

%Initialize the positions of search agents
if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

% If each variable has a different lb and ub
Pos = zeros(No_P, N_Var);
for ii=1:No_P
    Pos(ii, :)= (UB(ii)-LB(ii)) .* rand(1, N_Var) + LB(ii);
end

Pos_new=zeros(No_P, N_Var);
conv_curve=zeros(1,Max_Iter);               % Convergance array


tt=1;                         % starting iteration
Alpha=0.1;                   % the best value 0.1
Beta=0.005;                  % the best value 0.005
Ffun=zeros(1,size(Pos,1));     % (old fitness values)
Ffun_new=zeros(1,size(Pos,1)); % (new fitness values)

for ii=1:No_P 
    Ffun(1,ii)=fobj(Pos(ii,:), X, y);   %Calculate the fitness values of solutions
    if Ffun(1,ii)<Best_F
        Best_F=Ffun(1,ii);
        Best_P=Pos(ii,:)>0.5;
    end
end


while tt<Max_Iter+1  %Main loop %Update the Position of solutions
    ES=2*randn*(1-(tt/Max_Iter));  % Probability Ratio
    for ii=2:No_P 
        for jj=1:N_Var
            R=Best_P(1,jj)-Pos(randi([1 size(Pos,1)]),jj)/((Best_P(1,jj))+eps);
            P=Alpha+(Pos(ii,jj)-mean(Pos(ii,:)))./(Best_P(1,jj).*(UB(1, jj)-LB(1, jj))+eps);
            Eta=Best_P(1,jj)*P;
            if (tt<Max_Iter/4)
                Pos_new(ii,jj)=Best_P(1,jj)-Eta*Beta-R*rand;    
            elseif (tt<2*Max_Iter/4 && tt>=Max_Iter/4)
                Pos_new(ii,jj)=Best_P(1,jj)*Pos(randi([1 size(Pos,1)]),jj)*ES*rand;
            elseif (tt<3*Max_Iter/4 && tt>=2*Max_Iter/4)
                Pos_new(ii,jj)=Best_P(1,jj)*P*rand;
            else
                Pos_new(ii,jj)=Best_P(1,jj)-Eta*eps-R*rand;
            end
        end
        Flag_UB=Pos_new(ii,:)>UB; % check if they exceed (up) the boundaries
        Flag_LB=Pos_new(ii,:)<LB; % check if they exceed (down) the boundaries
        Pos_new(ii,:)=(Pos_new(ii,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;

        if sum(Pos_new(ii,:) > 0.5) >1  % must have at least 1 feature
            Ffun_new(1,ii)=fobj(Pos_new(ii,:), X, y);
        else
            Ffun_new(1,ii) = Ffun(1,ii-1);
        end

        if Ffun_new(1,ii)<Ffun(1,ii)
            Pos(ii,:)=Pos_new(ii,:);
            Ffun(1,ii)=Ffun_new(1,ii);
        end
        if Ffun(1,ii)<Best_F
            Best_F=Ffun(1,ii);
            Best_P=Pos(ii,:)>0.5;
        end
    end
    conv_curve(tt)=Best_F;  %Update the convergence curve

    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('RSA: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    tt=tt+1;
end

if Ffun(1,ii)<Best_F
    Best_F=Ffun(1,ii);
    Best_P=Pos(ii,:)>0.5;
end


CT = toc(timer);       % Total computation time in seconds
fprintf('RSA: Final fitness: %4.3f \n', Best_F);

%% END OF RSA.m