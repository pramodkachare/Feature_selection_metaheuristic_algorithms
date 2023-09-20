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

C1=0.5;C2=2;c3=.1;c4=.2;c5=2;D=.01;
X=bsxfun(@plus, LB, bsxfun(@times, rand(No_P,N_Var), (UB-LB)));%intial postions
for i=1:No_P
    FS(i) = feval(fobj,X(i,:), data, target);
end
[BestF, IndexBestF] = min(FS);
BestP = X(IndexBestF,:);
n1=round(No_P/2);
n2=No_P-n1;
X1=X(1:n1,:);
X2=X(n1+1:No_P,:);
for i=1:n1
    FS1(i) = feval(fobj,X1(i,:), data, target);
end
for i=1:n2
    FS2(i) = feval(fobj,X2(i,:), data, target);
end

[FSeo1, IndexFSeo1] = min(FS1);
[FSeo2, IndexFSeo2] = min(FS2);
Xeo1 = X1(IndexFSeo1,:);
Xeo2 = X2(IndexFSeo2,:);
vec_flag=[1,-1];
  if FSeo1<FSeo2
        FSss=FSeo1;
        YSol=Xeo1;
    else
        FSss=FSeo2;
        YSol=Xeo2;
    end
for t = 1:Max_Iter
    TF(t)=sinh(t/Max_Iter)^C1;
    X=[X1;X2];
    %             DO
    if TF(t)<0.9
           DOF=exp(-(C2*TF(t)-rand))^C2;
            TDO=c5*TF(t)-rand;%   direction of flow
            if (TDO)<rand
                %         select no of molecules
                M1N = c3*n1;
                M2N = c4*n1;
                NT12 =round((M2N-M1N).*rand(1,1) + M1N);
                for u=1:NT12
                    flag_index = floor(2*rand()+1);
                    DFg=vec_flag(flag_index);
                    Xm2=mean(X2);
                    Xm1=mean(X1);
                    J=-D*(Xm2-Xm1)/norm(Xeo2- X1(u,:)+eps);
                    X1new(u,:)= Xeo2+ DFg*DOF.*rand(1,N_Var).*(J.*Xeo2-X1(u,:));
                end
                for u=NT12+1:n1
                    for tt=1:N_Var
                        p=rand;
                        if p<0.8
                            X1new(u,tt) = Xeo1(tt);
                        elseif p<.9
                            r3=rand;
                            X1new(u,tt)=X1(u,tt)+DOF.*((UB(1,tt)-LB(1,tt))*r3+LB(1,tt));
                        else
                            X1new(u,tt) =X1(u,tt);
                        end
                        
                    end
                end
                for u=1:n2
                    r4=rand;
                    X2new(u,:)= Xeo2+DOF.*((UB-LB)*r4+LB);
                end
            else
                M1N = .1*n2;
                M2N = .2*n2;
                Ntransfer =round((M2N-M1N).*rand(1,1) + M1N);
                for u=1:Ntransfer
                    flag_index = floor(2*rand()+1);
                    DFg=vec_flag(flag_index);
                    R1=randi(n1);
                    Xm1=mean(X1);
                    Xm2=mean(X2);
                    J=-D*(Xm1-Xm2)/norm(Xeo1- X2(u,:)+eps);
                    X2new(u,:)=  Xeo1+DFg*DOF.*rand(1,N_Var).*(J.*Xeo1-1*X2(u,:));
                end
                for u=Ntransfer+1:n2
                    for tt=1:N_Var
                        p=rand;
                        if p<0.8
                            X2new(u,tt) = Xeo2(tt);
                        elseif p<.9
                            r3=rand;
                            X2new(u,tt)=X2(u,tt)+DOF.*((UB(1,tt)-LB(1,tt))*r3+LB(1,tt));
                        else
                            X2new(u,tt) =X2(u,tt);
                        end
                        
                    end
                end
                for u=1:n1
                    r4=rand;
                    X1new(u,:)= Xeo1+DOF.*((UB-LB)*r4+LB);
                end
            end
 
    else
%         Equilibrium operator (EO)
        if TF(t)<=1
            for u=1:n1
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                Xm1=mean(X1);
                Xmeo1=Xeo1;
                J=-D*(Xmeo1-Xm1)/norm(Xeo1- X1(u,:)+eps);
                DRF= exp(-J/TF(t));
                MS=exp(-FSeo1/(FS1(u)+eps));
                R1=rand(1,N_Var);
                Qeo=DFg*DRF.*R1;
                X1new(u,:)= Xeo1+Qeo.*X1(u,:)+Qeo.*(MS*Xeo1-X1(u,:));
            end
            for u=1:n2
                flag_index = floor(2*rand()+1);
                DFg=vec_flag(flag_index);
                Xm2=mean(X2);
                Xmeo2=Xeo2;
                J=-D*(Xmeo2-Xm2)/norm(Xeo2- X2(u,:)+eps);
                DRF= exp(-J/TF(t));
                MS=exp(-FSeo2/(FS2(u)+eps));
                R1=rand(1,N_Var);
                Qeo=DFg*DRF.*R1;
                X2new(u,:)=  Xeo2+Qeo.*X2(u,:)+Qeo.*(MS*Xeo1-X2(u,:));
            end
        else
            %     Steady state operator (SSO):
                for u=1:n1
            flag_index = floor(2*rand()+1);
            DFg=vec_flag(flag_index);
            Xm1=mean(X1);
            Xm=mean(X);
            J=-D*(Xm-Xm1)/norm(BestP- X1(u,:)+eps);
            DRF= exp(-J/TF(t));
            MS=exp(-FSss/(FS1(u)+eps));
            R1=rand(1,N_Var);
            Qg=DFg*DRF.*R1;
            X1new(u,:)=  BestP+Qg.*X1(u,:)+Qg.*(MS*BestP-X1(u,:));
        end
        for u=1:n2
            Xm1=mean(X1);
            Xm=mean(X);
            J=-D*(Xm1-Xm)/norm(BestP- X2(u,:)+eps);
            DRF= exp(-J/TF(t));
            MS=exp(-FSss/(FS2(u)+eps));
            flag_index = floor(2*rand()+1);
            DFg=vec_flag(flag_index);
                        Qg=DFg*DRF.*R1;
            X2new(u,:)= BestP+ Qg.*X2(u,:)+Qg.*(MS*BestP-X2(u,:));
        end
        end
    end
    for j=1:n1
        FU=X1new(j,:)>UB;FL=X1new(j,:)<LB;X1new(j,:)=(X1new(j,:).*(~(FU+FL)))+UB.*FU+LB.*FL;
        v = feval(fobj,X1new(j,:), data, target);
        if v<FS1(j)
            FS1(j)=v;
            X1(j,:)= X1new(j,:);
        end
    end
    for j=1:n2
        FU=X2new(j,:)>UB;FL=X2new(j,:)<LB;X2new(j,:)=(X2new(j,:).*(~(FU+FL)))+UB.*FU+LB.*FL;
        v = feval(fobj,X2new(j,:), data, target);
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
    conv_curve(t)=FSss;
    if FSss<BestF
        BestF=FSss;
        BestP =YSol;
        
    end
    if mod(t,1)==0  %Print the best universe details after every t iterations
        fprintf('FLA: Iteration %d    fitness: %4.3f \n', t, BestF);
    end  
end

end