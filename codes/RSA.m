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

function [Best_F, Best_P, Conv]=RSA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

    Best_P=zeros(1,N_Var);           % best positions
    Best_F=inf;                    % best fitness
    
    %Initialize the positions of search agents
    if length(UB)==1    % If same limit is applied on all variables
        UB = repmat(UB, 1, N_Var);
        LB = repmat(LB, 1, N_Var);
    end

    % If each variable has a different lb and ub
    X = zeros(No_P, N_Var);
    for i=1:No_P
        X(i, :)= (UB-LB) .* rand(1,N_Var) + LB;
    end
   
    Xnew=zeros(No_P, N_Var);
    Conv=zeros(1,Max_Iter);               % Convergance array


    t=1;                         % starting iteration
    Alpha=0.1;                   % the best value 0.1
    Beta=0.005;                  % the best value 0.005
    Ffun=zeros(1,size(X,1));     % (old fitness values)
    Ffun_new=zeros(1,size(X,1)); % (new fitness values)

    for i=1:size(X,1) 
        Ffun(1,i)=fobj(X(i,:), X, y);   %Calculate the fitness values of solutions
            if Ffun(1,i)<Best_F
                Best_F=Ffun(1,i);
                Best_P=X(i,:)>0.5;
            end
    end


    while t<Max_Iter+1  %Main loop %Update the Position of solutions
        ES=2*randn*(1-(t/Max_Iter));  % Probability Ratio
        for i=2:size(X,1) 
            for j=1:size(X,2)  
                    R=Best_P(1,j)-X(randi([1 size(X,1)]),j)/((Best_P(1,j))+eps);
                    P=Alpha+(X(i,j)-mean(X(i,:)))./(Best_P(1,j).*(UB(1, j)-LB(1, j))+eps);
                    Eta=Best_P(1,j)*P;
                    if (t<Max_Iter/4)
                        Xnew(i,j)=Best_P(1,j)-Eta*Beta-R*rand;    
                    elseif (t<2*Max_Iter/4 && t>=Max_Iter/4)
                        Xnew(i,j)=Best_P(1,j)*X(randi([1 size(X,1)]),j)*ES*rand;
                    elseif (t<3*Max_Iter/4 && t>=2*Max_Iter/4)
                        Xnew(i,j)=Best_P(1,j)*P*rand;
                    else
                        Xnew(i,j)=Best_P(1,j)-Eta*eps-R*rand;
                    end
            end

                Flag_UB=Xnew(i,:)>UB; % check if they exceed (up) the boundaries
                Flag_LB=Xnew(i,:)<LB; % check if they exceed (down) the boundaries
                Xnew(i,:)=(Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;

                if sum(Xnew(i,:) > 0.5) >1  % must have at least 1 feature
                    Ffun_new(1,i)=fobj(Xnew(i,:), X, y);
                else
                    Ffun_new(1,i) = Ffun(1,i-1);
                end

                if Ffun_new(1,i)<Ffun(1,i)
                    X(i,:)=Xnew(i,:);
                    Ffun(1,i)=Ffun_new(1,i);
                end
                if Ffun(1,i)<Best_F
                    Best_F=Ffun(1,i);
                    Best_P=X(i,:)>0.5;
                end
        end

        Conv(t)=Best_F;  %Update the convergence curve

        if mod(t,1)==0  %Print the best universe details after every t iterations
            fprintf('RSA: Iteration %d    fitness: %4.3f \n', t, Best_F);
        end
        t=t+1;
    end
    fprintf('RSA: Final fitness: %4.3f \n', Best_F);
end

function X=initialization(N,Dim,UB,LB)
    B_no= size(UB,2); % numnber of boundaries
    if B_no==1
        X=rand(N,Dim).*(UB-LB)+LB;
    end
    % If each variable has a different lb and ub
    if B_no>1
        for i=1:Dim
            Ub_i=UB(i);
            Lb_i=LB(i);
            X(:,i)=rand(N,1).*(Ub_i-Lb_i)+Lb_i;
        end
    end
end