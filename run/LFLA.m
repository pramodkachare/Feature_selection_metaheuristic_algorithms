%_________________________________________________________________________%
%  Fick's Law Algorithm (FLA) with Levy flight                            %
%                                                                         %
%  Developed in MATLAB R2021b                                             %
%                                                                         %
%  Coresponding Author:  Abdelazim G. Hussien                             %
%                                                                         %
%                                                                         %
%         e-Mail: abdelazim.hussien@liu.se                                %
%                 aga08@fayoum.edu.eg                                     %
%                                                                         %
%                                                                         %
%   Main paper: Fatma Hashim, Reham R Mostafa, Abdelazim G. Hussien,      %
%                     Seyedali Mirjalili, & Karam M. Sallam               %
%               Knowledge-based Systems                                   %
%                                                                         %
%_________________________________________________________________________%
function [BestF, Xss, CNVG] = LFLA(NoMolecules, T, lb, ub, dim, objfunc, data, target)
C1=0.5;C2=2;c3=.1;c4=.2;c5=2;
D=.01;
if numel(ub)>1
    ub = repmat(ub, NoMolecules, 1);
end
if numel(lb)>1
    lb = repmat(lb, NoMolecules, 1);
end

X=lb+rand(NoMolecules,dim).*(ub-lb);%intial postions
for i=1:NoMolecules
FS(i) = feval(objfunc,X(i,:), data, target);
end
[BestF, IndexBestF] = min(FS);
Xss = X(IndexBestF,:);
n1=round(NoMolecules/2);
n2=NoMolecules-n1;
X1=X(1:n1,:);
X2=X(n1+1:NoMolecules,:);
for i=1:n1
FS1(i) = feval(objfunc,X1(i,:), data, target);
end
for i=1:n2
FS2(i) = feval(objfunc,X2(i,:), data, target);
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
for t = 1:T
    TF(t)=sinh(t/T)^C1;
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
                    X1new(u,:)= Xeo2+ DFg*DOF.*rand(1,dim).*(J.*Xeo2-X1(u,:));
                end
                for u=NT12+1:n1
                    for tt=1:dim
                        p=rand;
                        if p<0.8
                            X1new(u,tt) = Xeo1(tt);
                        elseif p<.9
                            r3=rand;
                            if numel(ub) > 1
                                 X1new(u,tt)=X1(u,tt)+DOF.*((ub(1,tt)-lb(1, tt))*r3+lb(1,tt));
                            else
                                X1new(u,tt)=X1(u,tt)+DOF.*((ub-lb)*r3+lb);
                            end
                        else
                            X1new(u,tt) =X1(u,tt);
                        end
                        
                    end
                end
                for u=1:n2
                    r4=rand;
%                     if numel(ub)>1
                        X2new(u,:)= Xeo2+DOF.*((ub(1, :)-lb(1,:))*r4+lb(1,:));
%                     else
%                         X2new(u,:)= Xeo2+DOF.*((ub-lb)*r4+lb);
%                     end
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
                    X2new(u,:)=  Xeo1+DFg*DOF.*rand(1,dim).*(J.*Xeo1-1*X2(u,:));
                end
                for u=Ntransfer+1:n2
                    for tt=1:dim
                        p=rand;
                        if p<0.8
                            X2new(u,tt) = Xeo2(tt);
                        elseif p<.9
                            r3=rand;
                            if numel(ub) > 1
                                 X2new(u,tt)=X2(u,tt)+DOF.*((ub(1,tt)-lb(1, tt))*r3+lb(1,tt));
                            else
                                X2new(u,tt)=X2(u,tt)+DOF.*((ub-lb)*r3+lb);
                            end
                        else
                            X2new(u,tt) =X2(u,tt);
                        end
                        
                    end
                end
                for u=1:n1
                    r4=rand;
%                     if numel(ub)>1
                        X1new(u,:)= Xeo1+DOF.*((ub(1, :)-lb(1,:))*r4+lb(1,:));
%                     else
%                         X1new(u,:)= Xeo1+DOF.*((ub-lb)*r4+lb);
%                     end
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
                R1=rand(1,dim);
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
                R1=rand(1,dim);
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
            J=-D*(Xm-Xm1)/norm(Xss- X1(u,:)+eps);
            DRF= exp(-J/TF(t));
            MS=exp(-FSss/(FS1(u)+eps));
            R1=rand(1,dim);
            Qg=DFg*DRF.*R1;
            X1new(u,:)=  Xss+Qg.*X1(u,:)+Qg.*(MS*Xss-X1(u,:));
        end
        for u=1:n2
            Xm1=mean(X1);
            Xm=mean(X);
            J=-D*(Xm1-Xm)/norm(Xss- X2(u,:)+eps);
            DRF= exp(-J/TF(t));
            MS=exp(-FSss/(FS2(u)+eps));
            flag_index = floor(2*rand()+1);
            DFg=vec_flag(flag_index);
                        Qg=DFg*DRF.*R1;
            X2new(u,:)= Xss+ Qg.*X2(u,:)+Qg.*(MS*Xss-X2(u,:));
        end
        end
    end
    for j=1:n1
        FU=X1new(j,:)>ub(1,:);
        FL=X1new(j,:)<lb(1,:);
        X1new(j,:)=(X1new(j,:).*(~(FU+FL)))+ub(1,:).*FU+lb(1,:).*FL;
        v = feval(objfunc,X1new(j,:), data, target);
        if v<FS1(j)
            FS1(j)=v;
            X1(j,:)= X1new(j,:);
        end
    end
    for j=1:n2
        FU=X2new(j,:)>ub(1,:);
        FL=X2new(j,:)<lb(1,:);
        X2new(j,:)=(X2new(j,:).*(~(FU+FL)))+ub(1,:).*FU+lb(1,:).*FL;
        v = feval(objfunc,X2new(j,:), data, target);
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
    CNVG(t)=FSss;
    if FSss<BestF
        BestF=FSss;
        Xss =YSol;
    end
    
%%%%%%%%%%% Modified with Levy flight %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    for i=1:NoMolecules
        if i <= n1
            for j=1:dim
                X1(i,j) = Xss(1,j)*Levy(1);  
            end
        else
            for j=1:dim
                X2(i-n1,j) = Xss(1,j)*Levy(1);  
            end
        end
    end       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp(['iteration ' num2str(t) ':   ' ...
        'Best Cost Bw = ' num2str(BestF)]);  
end
end


%% Levy Flight 
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end