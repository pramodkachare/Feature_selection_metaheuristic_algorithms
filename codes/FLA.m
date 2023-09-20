%FLA Fick's Law Algorithm (FLA)
% [BestF, BestP, conv_curve, CT] = FLA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Hashim, F. A., Mostafa, R. R., Hussien, A. G., Mirjalili, S.
%               & Sallam, K. M. (2023). 
%               Fick’s Law Algorithm: A physical law-based algorithm for 
%               numerical optimization.  
%               Knowledge-Based Systems, 260, 110146. 
%               DOI: 10.1016/j.knosys.2022.110146

% 
%     [BestF, BestP] = PSO(data) applies feature selection on M-by-N matrix
%     data with N examples and assuming last column as the classification target 
%     and returns the best fitness value BestF and 1-by-(M-1) matrix of 
%     feature positions BestP.
%
%     [BestF, BestP] = PSO(data, target) applies feature selection on M-by-N 
%     feature matrix data and 1-by-N target matrix target and returns the 
%     best fitness value BestF and 1-by-(M-1)matrix of feature positions BestP.
%     
%     Example:
%
%
% Original Author: Abdelazim G. Hussien (abdelazim.hussien@liu.se,
% aga08@fayoum.edu.eg)
% Revised by : Pramod H. Kachare (Aug 2023)

function [BestF, BestP, conv_curve, CT] = FLA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

% Hyper-parameters
C1=0.5;     C2=2;       c3=.1;      c4=.2;      c5=2;       D=.01;

if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end

% Initialize postions
X=bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));

FS = zeros(1, No_P);
for ii=1:No_P
    FS(ii) = fobj(X(ii,:) > (LB+UB)/2, data, target);
end
[BestF, IndexBestF] = min(FS);
BestP = X(IndexBestF,:);

% Divide population in two equal halfs
n1=round(No_P/2);
n2=No_P-n1;
X1=X(1:n1,:);
X2=X(n1+1:No_P,:);

FS1 = zeros(1, No_P);
for ii=1:n1
    FS1(ii) = fobj(X1(ii,:) > (LB+UB)/2, data, target);
end
FS2 = zeros(1, No_P);
for ii=1:n2
    FS2(ii) = fobj(X2(ii,:) > (LB+UB)/2, data, target);
end

[FSeo1, IndexFSeo1] = min(FS1);
[FSeo2, IndexFSeo2] = min(FS2);
Xeo1 = X1(IndexFSeo1,:);
Xeo2 = X2(IndexFSeo2,:);
vec_flag=[1,-1];

if FSeo1 < FSeo2
    FSss = FSeo1;
%     YSol = Xeo1;
else
    FSss = FSeo2;
%     YSol = Xeo2;
end

TF = zeros(1, Max_Iter);
conv_curve = zeros(1, Max_Iter);
for tt = 1:Max_Iter
    TF(tt)=sinh(tt/Max_Iter)^C1;
    X=[X1; X2];
    %             DO
    if TF(tt)<0.9
        DOF=exp(-(C2*TF(tt)-rand))^C2;
        TDO=c5*TF(tt)-rand;%   direction of flow
        if (TDO)<rand
%         select no of molecules
            M1N = c3*n1;
            M2N = c4*n1;
            NT12 =round((M2N-M1N).*rand(1,1) + M1N);
            X1new = zeros(NT12, N_Var);
            for uu=1:NT12
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                Xm2=mean(X2);
                Xm1=mean(X1);
                J=-D*(Xm2-Xm1)/norm(Xeo2- X1(uu,:)+eps);
                X1new(uu,:)= Xeo2+ DFg*DOF.*rand(1,N_Var).*(J.*Xeo2-X1(uu,:));
            end
            for uu = NT12+1:n1
                for jj = 1:N_Var
                    p=rand;
                    if p<0.8
                        X1new(uu,jj) = Xeo1(jj);
                    elseif p<.9
                        r3=rand;
                        X1new(uu,jj)=X1(uu,jj)+DOF.*((UB(1,jj)-LB(1,jj))*r3+LB(1,jj));
                    else
                        X1new(uu,jj) =X1(uu,jj);
                    end
                end
            end
            X2new = zeros(n2, N_Var);
            for uu=1:n2
                r4=rand;
                X2new(uu,:)= Xeo2+DOF.*((UB-LB)*r4+LB);
            end
        else
            M1N = .1*n2;
            M2N = .2*n2;
            Ntransfer =round((M2N-M1N).*rand(1,1) + M1N);
            for uu=1:Ntransfer
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                R1=randi(n1);
                Xm1=mean(X1);
                Xm2=mean(X2);
                J=-D*(Xm1-Xm2)/norm(Xeo1- X2(uu,:)+eps);
                X2new(uu,:)=  Xeo1+DFg*DOF.*rand(1,N_Var).*(J.*Xeo1-1*X2(uu,:));
            end
            for uu=Ntransfer+1:n2
                for jj=1:N_Var
                    p=rand;
                    if p<0.8
                        X2new(uu,jj) = Xeo2(jj);
                    elseif p<.9
                        r3=rand;
                        X2new(uu,jj)=X2(uu,jj)+DOF.*((UB(1,jj)-LB(1,jj))*r3+LB(1,jj));
                    else
                        X2new(uu,jj) =X2(uu,jj);
                    end

                end
            end
            for uu=1:n1
                r4=rand;
                X1new(uu,:)= Xeo1+DOF.*((UB-LB)*r4+LB);
            end
        end
    else
%         Equilibrium operator (EO)
        if TF(tt)<=1
            for uu=1:n1
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                Xm1=mean(X1);
                Xmeo1=Xeo1;
                J=-D*(Xmeo1-Xm1)/norm(Xeo1- X1(uu,:)+eps);
                DRF= exp(-J/TF(tt));
                MS=exp(-FSeo1/(FS1(uu)+eps));
                R1=rand(1,N_Var);
                Qeo=DFg*DRF.*R1;
                X1new(uu,:)= Xeo1+Qeo.*X1(uu,:)+Qeo.*(MS*Xeo1-X1(uu,:));
            end
            for uu=1:n2
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                Xm2=mean(X2);
                Xmeo2=Xeo2;
                J=-D*(Xmeo2-Xm2)/norm(Xeo2- X2(uu,:)+eps);
                DRF= exp(-J/TF(tt));
                MS=exp(-FSeo2/(FS2(uu)+eps));
                R1=rand(1,N_Var);
                Qeo=DFg*DRF.*R1;
                X2new(uu,:)=  Xeo2+Qeo.*X2(uu,:)+Qeo.*(MS*Xeo1-X2(uu,:));
            end
        else
            %     Steady state operator (SSO):
        for uu=1:n1
            flag_index = floor(2*rand()+1);
            DFg=vec_flag(flag_index);
            Xm1=mean(X1);
            Xm=mean(X);
            J=-D*(Xm-Xm1)/norm(BestP- X1(uu,:)+eps);
            DRF= exp(-J/TF(tt));
            MS=exp(-FSss/(FS1(uu)+eps));
            R1=rand(1,N_Var);
            Qg=DFg*DRF.*R1;
            X1new(uu,:)=  BestP+Qg.*X1(uu,:)+Qg.*(MS*BestP-X1(uu,:));
        end
        for uu=1:n2
            Xm1=mean(X1);
            Xm=mean(X);
            J=-D*(Xm1-Xm)/norm(BestP- X2(uu,:)+eps);
            DRF= exp(-J/TF(tt));
            MS=exp(-FSss/(FS2(uu)+eps));
            flag_index = floor(2*rand()+1);
            DFg=vec_flag(flag_index);
                        Qg=DFg*DRF.*R1;
            X2new(uu,:)= BestP+ Qg.*X2(uu,:)+Qg.*(MS*BestP-X2(uu,:));
        end
        end
    end
    for j=1:n1
        FU=X1new(j,:)>UB;FL=X1new(j,:)<LB;X1new(j,:)=(X1new(j,:).*(~(FU+FL)))+UB.*FU+LB.*FL;
        v = fobj(X1new(j,:) > (LB+UB)/2, data, target);
        if v<FS1(j)
            FS1(j)=v;
            X1(j,:)= X1new(j,:);
        end
    end
    for j=1:n2
        FU=X2new(j,:)>UB;FL=X2new(j,:)<LB;X2new(j,:)=(X2new(j,:).*(~(FU+FL)))+UB.*FU+LB.*FL;
        v = fobj(X2new(j,:) > (LB+UB)/2, data, target);
        if v<FS2(j)
            FS2(j)=v;
            X2(j,:)= X2new(j,:);
        end
    end
    
    [FSeo1, IndexFSeo1] = min(FS1);
    [FSeo2, IndexFSeo2] = min(FS2);
    
    Xeo1 = X1(IndexFSeo1,:);
    Xeo2 = X2(IndexFSeo2,:);
    if FSeo1<FSeo2
        FSss=FSeo1;
        YSol=Xeo1;
    else
        FSss=FSeo2;
        YSol=Xeo2;
    end
    
    conv_curve(tt)=FSss;
    if FSss<BestF
        BestF=FSss;
        BestP =YSol;
        
    end

    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('FLA: Iteration %d    fitness: %4.3f \n', tt, BestF);
    end  
end
CT = toc(timer);       % Total computation time in seconds

%% END OF FLA.m