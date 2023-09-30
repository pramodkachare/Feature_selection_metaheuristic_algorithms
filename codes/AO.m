%AO Aquila Optimizer
% [Best_F,Best_P,conv_curve,CT] = AO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Abualigah, L, Yousri, D, Abd Elaziz, M, Ewees, A, Al-qaness, M,
%               & Gandomi, A. (2021). 
%               Aquila Optimizer: A novel meta-heuristic optimization algorithm.  
%               Computers & Industrial Engineering, 157, 107250. 
%               DOI: 10.1016/j.cie.2021.107250

% 
%     [Best_F, Best_P] = AO(data) applies feature selection on M-by-N matrix
%     data with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F, Best_P] = AO(data, target) applies feature selection on M-by-N 
%     feature matrix data and 1-by-N target matrix target and returns the 
%     best fitness value Best_F and 1-by-(M-1)matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Abualigah, L, Yousri, D, Abd Elaziz, M, Ewees, A, 
%                  Al-qaness, M, Gandomi, A.(Aligah.2020@gmail.com)
% Revised by : Pramod H. Kachare (Sep 2023)


function [Best_F,Best_P,conv_curve,CT] = AO(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

Best_P=zeros(1,N_Var);
Best_F=inf;

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

% Initialize postions
X=bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));
Xnew=X;

Ffun=zeros(1,size(X,1));
Ffun_new=zeros(1,size(Xnew,1));

% Array to Hold Best Cost Values
conv_curve=zeros(Max_Iter,1);

tt=1;

alpha=0.1;      delta=0.1;

while tt<Max_Iter+1
    for ii=1:size(X,1)
        F_UB=X(ii,:)>UB;
        F_LB=X(ii,:)<LB;
        X(ii,:)=(X(ii,:).*(~(F_UB+F_LB)))+UB.*F_UB+LB.*F_LB;
        Ffun(1,ii)=fobj(X(ii,:) > (LB+UB)/2, data, target);
        if Ffun(1,ii)<Best_F
            Best_F=Ffun(1,ii);
            Best_P=X(ii,:);
        end
    end
    
    
    G2=2*rand()-1; % Eq. (16)
    G1=2*(1-(tt/Max_Iter));  % Eq. (17)
    to = 1:N_Var;
    u = .0265;
    r0 = 10;
    r = r0 +u*to;
    omega = .005;
    phi0 = 3*pi/2;
    phi = -omega*to+phi0;
    x = r .* sin(phi);  % Eq. (9)
    y = r .* cos(phi); % Eq. (10)
    QF=tt^((2*rand()-1)/(1-Max_Iter)^2); % Eq. (15)
        %-------------------------------------------------------------------------------------
    for ii=1:size(X,1)
        %-------------------------------------------------------------------------------------
        if tt<=(2/3)*Max_Iter
            if rand <0.5
                Xnew(ii,:)=Best_P(1,:)*(1-tt/Max_Iter)+(mean(X(ii,:))-Best_P(1,:))*rand(); % Eq. (3) and Eq. (4)
                Ffun_new(1,ii)=fobj(Xnew(ii,:) > (LB+UB)/2, data, target);
                if Ffun_new(1,ii)<Ffun(1,ii)
                    X(ii,:)=Xnew(ii,:);
                    Ffun(1,ii)=Ffun_new(1,ii);
                end
            else
                %-------------------------------------------------------------------------------------
                Xnew(ii,:)=Best_P(1,:).*Levy(N_Var)+X((floor(N*rand()+1)),:)+(y-x)*rand;       % Eq. (5)
                Ffun_new(1,ii)=fobj(Xnew(ii,:) > (LB+UB)/2, data, target);
                if Ffun_new(1,ii)<Ffun(1,ii)
                    X(ii,:)=Xnew(ii,:);
                    Ffun(1,ii)=Ffun_new(1,ii);
                end
            end
            %-------------------------------------------------------------------------------------
        else
            if rand<0.5
                Xnew(ii,:)=(Best_P(1,:)-mean(X))*alpha-rand+((UB-LB)*rand+LB)*delta;   % Eq. (13)
                Ffun_new(1,ii)=fobj(Xnew(ii,:) > (LB+UB)/2, data, target);
                if Ffun_new(1,ii)<Ffun(1,ii)
                    X(ii,:)=Xnew(ii,:);
                    Ffun(1,ii)=Ffun_new(1,ii);
                end
            else
                %-------------------------------------------------------------------------------------
                Xnew(ii,:)=QF*Best_P(1,:)-(G2*X(ii,:)*rand)-G1.*Levy(N_Var)+rand*G2; % Eq. (14)
                Ffun_new(1,ii)=fobj(Xnew(ii,:) > (LB+UB)/2, data, target);
                if Ffun_new(1,ii)<Ffun(1,ii)
                    X(ii,:)=Xnew(ii,:);
                    Ffun(1,ii)=Ffun_new(1,ii);
                end
            end
        end
    end
    %-------------------------------------------------------------------------------------
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('AO: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    conv_curve(tt)=Best_F;
    tt=tt+1;
end
CT = toc(timer);       % Total computation time in seconds
fprintf('AO: Final fitness: %4.3f \n', Best_F);
end


function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end


%% END OF FLA.m