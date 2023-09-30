%AOA Arithmetic Optimization Algorithm
% [Best_F,Best_P,conv_curve,CT] = AOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M.,
%               & Gandomi, A. H. (2021). 
%               The Arithmetic Optimization Algorithm. 
%               Computer Methods in Applied Mechanics & Engineering, 376, 113609. 
%               DOI: 10.1016/j.cma.2020.113609

% 
%     [Best_F, Best_P] = AOA(data) applies feature selection on M-by-N matrix
%     data with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F, Best_P] = AOA(data, target) applies feature selection on M-by-N 
%     feature matrix data and 1-by-N target matrix target and returns the 
%     best fitness value Best_F and 1-by-(M-1)matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Laith Abualigah, Ali Diabat, Seyedali Mirjalili, 
%                  Mohamed Abd Elaziz, & Amir H. Gandomi (Aligah.2020@gmail.com)
% Revised by : Pramod H. Kachare (Sep 2023)


function [Best_F,Best_P,conv_curve,CT]=AOA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

%Two variables to keep the positions and the fitness value of the best-obtained solution
Best_P=zeros(1,N_Var);
Best_F=inf;

conv_curve = zeros(Max_Iter, 1);

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

%Initialize the positions of solution
X=bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));
Xnew=X;

Ffun=zeros(1,size(X,1));% (fitness values)
Ffun_new=zeros(1,size(Xnew,1));% (fitness values)

MOP_Max=1;      MOP_Min=0.2;        Alpha=5;        Mu=0.499;

for ii=1:size(X,1)
    Ffun(1,ii)=fobj(X(ii,:) > (LB+UB)/2, data, target);  %Calculate the fitness values of solutions
    if Ffun(1,ii)<Best_F
        Best_F=Ffun(1,ii);
        Best_P=X(ii,:);
    end
end
    
tt=1;
while tt<Max_Iter+1  %Main loop
    MOP=1-((tt)^(1/Alpha)/(Max_Iter)^(1/Alpha));   % Probability Ratio 
    MOA=MOP_Min+tt*((MOP_Max-MOP_Min)/Max_Iter); %Accelerated function
   
    %Update the Position of solutions
    for ii=1:size(X,1)   % if each of the UB and LB has a just value 
        for jj=1:size(X,2)
           r1=rand();
            if (size(LB,2)==1)
                if r1<MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(ii,jj)=Best_P(1,jj)/(MOP+eps)*((UB-LB)*Mu+LB);
                    else
                        Xnew(ii,jj)=Best_P(1,jj)*MOP*((UB-LB)*Mu+LB);
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(ii,jj)=Best_P(1,jj)-MOP*((UB-LB)*Mu+LB);
                    else
                        Xnew(ii,jj)=Best_P(1,jj)+MOP*((UB-LB)*Mu+LB);
                    end
                end               
            end
            
           
            if (size(LB,2)~=1)   % if each of the UB and LB has more than one value 
                r1=rand();
                if r1<MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(ii,jj)=Best_P(1,jj)/(MOP+eps)*((UB(jj)-LB(jj))*Mu+LB(jj));
                    else
                        Xnew(ii,jj)=Best_P(1,jj)*MOP*((UB(jj)-LB(jj))*Mu+LB(jj));
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(ii,jj)=Best_P(1,jj)-MOP*((UB(jj)-LB(jj))*Mu+LB(jj));
                    else
                        Xnew(ii,jj)=Best_P(1,jj)+MOP*((UB(jj)-LB(jj))*Mu+LB(jj));
                    end
                end               
            end
            
        end
        
        Flag_UB=Xnew(ii,:)>UB; % check if they exceed (up) the boundaries
        Flag_LB=Xnew(ii,:)<LB; % check if they exceed (down) the boundaries
        Xnew(ii,:)=(Xnew(ii,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
 
        Ffun_new(1,ii)=fobj(Xnew(ii,:) > (LB+UB)/2, data, target);  % calculate Fitness function 
        if Ffun_new(1,ii)<Ffun(1,ii)
            X(ii,:)=Xnew(ii,:);
            Ffun(1,ii)=Ffun_new(1,ii);
        end
        if Ffun(1,ii)<Best_F
            Best_F=Ffun(1,ii);
            Best_P=X(ii,:);
        end  
    end
    
    %Update the convergence curve
    conv_curve(tt)=Best_F;
    
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('AOA: Iteration %d    fitness: %4.3f \n', tt, Best_F);
    end
    tt=tt+1;  % incremental iteration 
end
CT = toc(timer);       % Total computation time in seconds
fprintf('AOA: Final fitness: %4.3f \n', Best_F);

%% END OF AOA.m