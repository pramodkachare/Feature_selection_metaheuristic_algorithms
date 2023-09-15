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
%     and returns the best fitness value Best_F and 1-by-(M-1) matrix of 
%     feature positions Best_P.
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

function [Score,BestRemora,Convergence]=ROA(Search_Agents,Max_iterations,...
    Lowerbound,Upperbound,dimensions,objective)
tic;       
BestRemora=zeros(1,dimensions);
Score=inf; 
Remora=init(Search_Agents,dimensions,Upperbound,Lowerbound); % Generate initial remora population
Prevgen{1}=Remora; 
Convergence=zeros(1,Max_iterations);
t=0;
while t<Max_iterations  
    
%% Memory of previous generation

    if t<=1
        PreviousRemora = Prevgen{1};
    else
        PreviousRemora = Prevgen{t-1};
    end
    
% Boundary check

    for i=1:size(Remora,1)
        Flag4Upperbound=Remora(i,:)>Upperbound;
        Flag4Lowerbound=Remora(i,:)<Lowerbound;
        Remora(i,:)=(Remora(i,:).*(~(Flag4Upperbound+Flag4Lowerbound)))+Upperbound.*Flag4Upperbound+Lowerbound.*Flag4Lowerbound;
        fitness=objective(Remora(i,:));
        
 % Evaluate fitness function of search agents   
 
		if fitness<Score 
            Score=fitness; 
            BestRemora=Remora(i,:);
        end
        
    end
    
  % Make a experience attempt through equation (2)
  
    for j=1:size(Remora,1)
        RemoraAtt = Remora(j,:)+(Remora(j,:)-PreviousRemora(j,:))*randn;                    %Equation(2)
        
  % Calculate the fitness function value of the attempted solution (fitnessAtt)
  
        fitnessAtt=objective(RemoraAtt);
        
  % Calculate the fitness function value of the current solution (fitnessI)
  
        fitnessI=objective(Remora(j,:));
        
  % Check if the current fitness (fitnessI) is better than the attempted fitness(fitnessAtt)
  % if No, Perform host feeding by equation (9)
  
        if fitnessI>fitnessAtt
            V = 2*(1-t/Max_iterations);                                                     % Equation (12)
            B = 2*V*rand-V;                                                                 % Equation (11)
            C = 0.1;
            A = B*(Remora(j,:)-C*BestRemora);                                               % Equation (10)
            Remora(j,:)= Remora(j,:)+A;                                                     % Equation (9)
            
  % If yes perform host conversion using equation (1) and (5)       
  
        elseif randi([0 1],1)==0
            a=-(1+t/Max_iterations);                                                        % Equation (7)
            alpha = rand*(a-1)+1;                                                           % Equation (6)
            D = abs(BestRemora-Remora(j,:));                                                % Equation (8)
            Remora(j,:) = D*exp(alpha)*cos(2*pi*a)+Remora(j,:);                             % Equation (5)
        else 
            m=randperm(size(Remora,1));
            Remora(j,:)=BestRemora-((rand*(BestRemora+Remora(m(1),:))/2)-Remora(m(1),:));   % Equation (1)
        end
    end
    
  t=t+1 ;
  fprintf('Iteration : %d, best score : %d\n',t,Score)
  Prevgen{t+1}=Remora; 
  Convergence(t)=Score;
    end   
end


