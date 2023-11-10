%SCA Sine Cosine Algorithm
% [Best_F,Best_P,conv_curve, CT] = SCA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: S. Mirjalili.
%               SCA: a sine cosine algorithm for solving optimization problems. 
%               Knowledge-based systems, 96, 120-133, 2016
%               DOI: 10.1016/j.knosys.2015.12.022 
% 
%     [Best_F,Best_P] = SCA(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
%
%     [Best_F,Best_P] = SCA(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     Best_F and 1-by-(M-1) matrix of feature positions Best_P.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili (ali.mirjalili@gmail.com)                   
% Revised by : Pramod H. Kachare (Nov 2023)


% To run SCA: [Best_score,Best_pos,cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)

function [Best_F,Best_P,conv_curve]=SCASCA(data, target, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
%Initialize the set of random solutions
X=initialization(No_P,N_Var,UB,LB);

Best_P=zeros(1,N_Var);
Best_F=inf;

conv_curve=zeros(1,Max_Iter);
Objective_values = zeros(1,size(X,1));

% Calculate the fitness of the first set and find the best one
for i=1:size(X,1)
    Objective_values(1,i)=fobj(X(i,:));
    if i==1
        Best_P=X(i,:);
        Best_F=Objective_values(1,i);
    elseif Objective_values(1,i)<Best_F
        Best_P=X(i,:);
        Best_F=Objective_values(1,i);
    end
    
    All_objective_values(1,i)=Objective_values(1,i);
end

%Main loop
t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness
while t<=Max_Iter
    
    % Eq. (3.4)
    a = 2;
    Max_Iter = Max_Iter;
    r1=a-t*((a)/Max_Iter); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for i=1:size(X,1) % in i-th solution
        for j=1:size(X,2) % in j-th dimension
            
            % Update r2, r3, and r4 for Eq. (3.3)
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Best_P(j)-X(i,j)));
            else
                % Eq. (3.2)
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Best_P(j)-X(i,j)));
            end
            
        end
    end
    
    for i=1:size(X,1)
         
        % Check if solutions go outside the search spaceand bring them back
        Flag4ub=X(i,:)>UB;
        Flag4lb=X(i,:)<LB;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+UB.*Flag4ub+LB.*Flag4lb;
        
        % Calculate the objective values
        Objective_values(1,i)=fobj(X(i,:));
        
        % Update the destination if there is a better solution
        if Objective_values(1,i)<Best_F
            Best_P=X(i,:);
            Best_F=Objective_values(1,i);
        end
    end
    
    conv_curve(t)=Best_F;
    
    % Display the iteration and best optimum obtained so far
    if mod(t,50)==0
        display(['At iteration ', num2str(t), ' the optimum is ', num2str(Best_F)]);
    end
    
    % Increase the iteration counter
    t=t+1;
end