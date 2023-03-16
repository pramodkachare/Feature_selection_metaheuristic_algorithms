%_______________________________________________________________________________________%
%  Reptile Search Algroithm (RSA) source codes demo version 1.0                         %
%                                                                                       %
%  Developed in MATLAB R2015a (7.13)                                                    %
%                                                                                       %
%  Author and programmer: Laith Abualigah                                               %
%                                                                                       %
%         e-Mail: Aligah.2020@gmail.com                                                 %
%       Homepage:                                                                       %
%         1- https://scholar.google.com/citations?user=39g8fyoAAAAJ&hl=en               %
%         2- https://www.researchgate.net/profile/Laith_Abualigah                       %
%_______________________________________________________________________________________%
%  Main paper:            Reptile Search Algorithm (RSA):                               %
%                  A novel nature-inspired metaheuristic algorithm                      %                                                                       %
%_______________________________________________________________________________________%

function [Best_F,Best_P,Conv, CT]= RSA_SO(N,T,LB,UB,Dim,F_obj, data, target)
    Best_P = zeros(1,Dim);           % best positions
    Best_F = inf;  
    Conv=zeros(1,T);               % Convergance array
    CT=zeros(1,T);                 % computation time (s)
    
    X=initialization(N,Dim,UB,LB); % Initialize the positions of solution
    Ffun=zeros(1,size(X,1));     % (old fitness values)
    for i=1:size(X,1) 
        Ffun(1,i)= F_obj(X(i,:), data, target);   %Calculate the fitness values of solutions
        if Ffun(1,i)<Best_F
            Best_F=Ffun(1,i);
            Best_P=X(i,:)>0.5;
        end
    end
        
    for jj = 1: T
        t1 = tic;
        [~, ~, ~, Xnew_rsa]=RSA(N/2, T, LB, UB, Dim, F_obj, data, target, X(1:N/2, :), jj);
        rsa_t = toc(t1);
        
        t1 = tic;
        [~, ~, ~, Xnewm_so, Xnewf_so] = SO(N/2, T, F_obj, Dim, LB, UB, data, target, X(N/2+1:end, :), jj);
        so_t = toc(t1);
        
        t1 = tic;
        Xnext = [Xnew_rsa; Xnewm_so; Xnewf_so];
        Ffun_next=zeros(1,size(Xnext,1)); % (new fitness values)

        for i=1:size(X,1) 
            if sum(Xnext(i,:)>0.5) > 0
                Ffun_next(1,i)= F_obj(Xnext(i,:), data, target);   %Calculate the fitness values of solutions
            else
                Ffun_next(1,i)= Inf;
            end
            if Ffun_next(1,i)<Best_F
                Best_F=Ffun_next(1,i);
                Best_P=Xnext(i,:)>0.5;
            end
        end
        
        for i = 1:N/2
            if Ffun_next(1,i) < Ffun_next(1,N/2+i)
                Xnext(N/2+i, :) = Xnext(i, :);
            else
                Xnext(i, :) = Xnext(N/2+i, :);
            end
        end
    
        if mod(jj,1)==0  %Print the best universe details after every t iterations
            fprintf('RSA-SO: Iteration %d    fitness: %4.3f \n', jj, Best_F);
        end
        Conv(jj) = Best_F;
        X = Xnext;
        CT(1, jj) = max(rsa_t, so_t)+ toc(t1);
    end
    CT = sum(CT);
end


function [Best_F,Best_P,Conv, Xnew]=RSA(N,T,LB,UB,Dim,F_obj, data, target, X, t)
    Best_P=zeros(1,Dim);           % best positions
    Best_F=inf;                    % best fitness
%     X=initialization(N,Dim,UB,LB); % Initialize the positions of solution
    Xnew=zeros(N,Dim);
    Conv=zeros(1,T);               % Convergance array


%     t=1;                         % starting iteration
    Alpha=0.1;                   % the best value 0.1
    Beta=0.005;                  % the best value 0.005
    Ffun=zeros(1,size(X,1));     % (old fitness values)
    Ffun_new=zeros(1,size(X,1)); % (new fitness values)

    for i=1:size(X,1) 
        Ffun(1,i)=F_obj(X(i,:), data, target);   %Calculate the fitness values of solutions
            if Ffun(1,i)<Best_F
                Best_F=Ffun(1,i);
                Best_P=X(i,:)>0.5;
            end
    end


%     while t<T+1  %Main loop %Update the Position of solutions
        ES=2*randn*(1-(t/T));  % Probability Ratio
        for i=2:size(X,1) 
            for j=1:size(X,2)  
                    R=Best_P(1,j)-X(randi([1 size(X,1)]),j)/((Best_P(1,j))+eps);
                    P=Alpha+(X(i,j)-mean(X(i,:)))/(Best_P(1,j)*(UB-LB)+eps);
                    Eta=Best_P(1,j)*P;
                    if (t<T/4)
                        Xnew(i,j)=Best_P(1,j)-Eta*Beta-R*rand;    
                    elseif (t<2*T/4 && t>=T/4)
                        Xnew(i,j)=Best_P(1,j)*X(randi([1 size(X,1)]),j)*ES*rand;
                    elseif (t<3*T/4 && t>=2*T/4)
                        Xnew(i,j)=Best_P(1,j)*P*rand;
                    else
                        Xnew(i,j)=Best_P(1,j)-Eta*eps-R*rand;
                    end
            end

                Flag_UB=Xnew(i,:)>UB; % check if they exceed (up) the boundaries
                Flag_LB=Xnew(i,:)<LB; % check if they exceed (down) the boundaries
                Xnew(i,:)=(Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;

                if sum(Xnew(i,:) > 0.5) >1  % must have at least 1 feature
                    Ffun_new(1,i)=F_obj(Xnew(i,:), data, target);
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

%         if mod(t,1)==0  %Print the best universe details after every t iterations
%             fprintf('RSA: Iteration %d    fitness: %4.3f \n', t, Best_F);
%         end
%         t=t+1;
%     end
%     fprintf('RSA: Final fitness: %4.3f \n', Best_F);
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


%___________________________________________________________________%
%  Snake Optimizer (SO) source codes version 1.0                    %
%                                                                   %
%  Developed in MATLAB R2021b                                       %
%                                                                   %
%  Author and programmer:  Fatma Hashim & Abdelazim G. Hussien      %
%                                                                   %
%         e-Mail: fatma_hashim@h-eng.helwan.edu.eg                  %
%                 abdelazim.hussien@liu.se                          %
%                 aga08@fayoum.edu.eg                               %
%                                                                   %
%                                                                   %
%   Main paper: Fatma Hashim & Abdelazim G. Hussien                 %
%               Knowledge-based Systems                             %
%               in press,                                           %
%               DOI: 10.1016/j.knosys.2022.108320                   %
%                                                                   %
%___________________________________________________________________%
function [Xfood, fval, gbest_t, Xnewm, Xnewf] = SO(N,T,fobj, dim,lb,ub, data, target, X, t)
%initial 
vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;
% X=lb+rand(N,dim)*(ub-lb);

for i=1:N
    fitness(i)=feval(fobj,X(i,:), data, target);   
end
[GYbest, gbest] = min(fitness);
Xfood = X(gbest,:);
%Diving the swarm into two equal groups males and females
Nm=round(N/2);%eq.(2&3)
Nf=N-Nm;
Xm=X(1:Nm,:);
Xf=X(Nm+1:N,:);
fitness_m=fitness(1:Nm);
fitness_f=fitness(Nm+1:N);
[fitnessBest_m, gbest1] = min(fitness_m);
Xbest_m = Xm(gbest1,:);
[fitnessBest_f, gbest2] = min(fitness_f);
Xbest_f = Xf(gbest2,:);

% for t = 1:T
    Temp=exp(-((t)/T));  %eq.(4)
  Q=C1*exp(((t-T)/(T)));%eq.(5)
    if Q>1        Q=1;    end
    % Exploration Phase (no Food)
if Q<Threshold
    for i=1:Nm
        for j=1:1:dim
            rand_leader_index = floor(Nm*rand()+1);
            X_randm = Xm(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Am=exp(-fitness_m(rand_leader_index)/(fitness_m(i)+eps));%eq.(7)
            Xnewm(i,j)=X_randm(j)+Flag*C2*Am*((ub-lb)*rand+lb);%eq.(6)
        end
    end
    for i=1:Nf
        for j=1:1:dim
            rand_leader_index = floor(Nf*rand()+1);
            X_randf = Xf(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Af=exp(-fitness_f(rand_leader_index)/(fitness_f(i)+eps));%eq.(9)
            Xnewf(i,j)=X_randf(j)+Flag*C2*Af*((ub-lb)*rand+lb);%eq.(8)
        end
    end
else %Exploitation Phase (Food Exists)
    if Temp>Thresold2  %hot
        for i=1:Nm
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewm(i,j)=Xfood(j)+C3*Flag*Temp*rand*(Xfood(j)-Xm(i,j));%eq.(10)
            end
        end
        for i=1:Nf
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewf(i,j)=Xfood(j)+Flag*C3*Temp*rand*(Xfood(j)-Xf(i,j));%eq.(10)
            end
        end
    else %cold
        if rand>0.6 %fight
            for i=1:Nm
                for j=1:1:dim
                    FM=exp(-(fitnessBest_f)/(fitness_m(i)+eps));%eq.(13)
                    Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Q*Xbest_f(j)-Xm(i,j));%eq.(11)
                    
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    FF=exp(-(fitnessBest_m)/(fitness_f(i)+eps));%eq.(14)
                    Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Q*Xbest_m(j)-Xf(i,j));%eq.(12)
                end
            end
        else%mating
            for i=1:Nm
                for j=1:1:dim
                    Mm=exp(-fitness_f(i)/(fitness_m(i)+eps));%eq.(17)
                    Xnewm(i,j)=Xm(i,j) +C3*rand*Mm*(Q*Xf(i,j)-Xm(i,j));%eq.(15
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    Mf=exp(-fitness_m(i)/(fitness_f(i)+eps));%eq.(18)
                    Xnewf(i,j)=Xf(i,j) +C3*rand*Mf*(Q*Xm(i,j)-Xf(i,j));%eq.(16)
                end
            end
            flag_index = floor(2*rand()+1);
            egg=vec_flag(flag_index);
            if egg==1;
                [GYworst, gworst] = max(fitness_m);
                Xnewm(gworst,:)=lb+rand*(ub-lb);%eq.(19)
                [GYworst, gworst] = max(fitness_f);
                Xnewf(gworst,:)=lb+rand*(ub-lb);%eq.(20)
            end
        end
    end
end
    for j=1:Nm
         Flag4ub=Xnewm(j,:)>ub;
         Flag4lb=Xnewm(j,:)<lb;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = feval(fobj,Xnewm(j,:), data, target);
        if y<fitness_m(j)
            fitness_m(j)=y;
            Xm(j,:)= Xnewm(j,:);
        end
    end
    
    [Ybest1,gbest1] = min(fitness_m);
    
    for j=1:Nf
         Flag4ub=Xnewf(j,:)>ub;
         Flag4lb=Xnewf(j,:)<lb;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = feval(fobj,Xnewf(j,:), data, target);
        if y<fitness_f(j)
            fitness_f(j)=y;
            Xf(j,:)= Xnewf(j,:);
        end
    end
    
    [Ybest2,gbest2] = min(fitness_f);
    
    if Ybest1<fitnessBest_m
        Xbest_m = Xm(gbest1,:);
        fitnessBest_m=Ybest1;
    end
    if Ybest2<fitnessBest_f
        Xbest_f = Xf(gbest2,:);
        fitnessBest_f=Ybest2;
        
    end
    if Ybest1<Ybest2
        gbest_t(t)=min(Ybest1);
    else
        gbest_t(t)=min(Ybest2);
        
    end
    if fitnessBest_m<fitnessBest_f
        GYbest=fitnessBest_m;
        Xfood=Xbest_m;
    else
        GYbest=fitnessBest_f;
        Xfood=Xbest_f;
    end
%     if mod(t,1)==0  %Print the best universe details after every t iterations
%         fprintf('SO: Iteration %d    fitness: %4.3f \n', t, min(fitnessBest_m, fitnessBest_f));
%     end
% end
fval = GYbest;
% fprintf('SO: Final fitness: %4.3f \n', gbest_t(end));
end




