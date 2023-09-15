%ROA Reptile Search Algorithm
% [Best_F,Best_P, conv_curve, CT] = ROA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Jia, H., Peng, X., & Lang, C. (2021). 
%               Remora optimization algorithm
%               Expert Systems with Applications, 185, 115665.
%               DOI: 10.1016/j.eswa.2021.115665 
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

function [Best_F,Best_Pos,conv_curve, CT]=ROA(X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
tic;       
Best_Pos=zeros(1,N_Var);
Best_F=inf; 
Remora=init(Search_Agents,N_Var,UB,LB); % Generate initial remora population
Prevgen{1}=Remora; 
conv_curve=zeros(1,Max_Iter);
tt=0;
while tt<Max_Iter  
    
%% Memory of previous generation

    if tt<=1
        PreviousRemora = Prevgen{1};
    else
        PreviousRemora = Prevgen{tt-1};
    end
    
% Boundary check

    for i=1:size(Remora,1)
        Flag4Upperbound=Remora(i,:)>UB;
        Flag4Lowerbound=Remora(i,:)<LB;
        Remora(i,:)=(Remora(i,:).*(~(Flag4Upperbound+Flag4Lowerbound)))+UB.*Flag4Upperbound+LB.*Flag4Lowerbound;
        fitness=fobj(Remora(i,:));
        
 % Evaluate fitness function of search agents   
 
		if fitness<Best_F 
            Best_F=fitness; 
            Best_Pos=Remora(i,:);
        end
        
    end
    
  % Make a experience attempt through equation (2)
  
    for j=1:size(Remora,1)
        RemoraAtt = Remora(j,:)+(Remora(j,:)-PreviousRemora(j,:))*randn;                    %Equation(2)
        
  % Calculate the fitness function value of the attempted solution (fitnessAtt)
  
        fitnessAtt=fobj(RemoraAtt);
        
  % Calculate the fitness function value of the current solution (fitnessI)
  
        fitnessI=fobj(Remora(j,:));
        
  % Check if the current fitness (fitnessI) is better than the attempted fitness(fitnessAtt)
  % if No, Perform host feeding by equation (9)
  
        if fitnessI>fitnessAtt
            V = 2*(1-tt/Max_Iter);                                                     % Equation (12)
            B = 2*V*rand-V;                                                                 % Equation (11)
            C = 0.1;
            A = B*(Remora(j,:)-C*Best_Pos);                                               % Equation (10)
            Remora(j,:)= Remora(j,:)+A;                                                     % Equation (9)
            
  % If yes perform host conversion using equation (1) and (5)       
  
        elseif randi([0 1],1)==0
            a=-(1+tt/Max_Iter);                                                        % Equation (7)
            alpha = rand*(a-1)+1;                                                           % Equation (6)
            D = abs(Best_Pos-Remora(j,:));                                                % Equation (8)
            Remora(j,:) = D*exp(alpha)*cos(2*pi*a)+Remora(j,:);                             % Equation (5)
        else 
            m=randperm(size(Remora,1));
            Remora(j,:)=Best_Pos-((rand*(Best_Pos+Remora(m(1),:))/2)-Remora(m(1),:));   % Equation (1)
        end
    end
    
  tt=tt+1 ;
  fprintf('Iteration : %d, best score : %d\n',tt,Best_F)
  Prevgen{tt+1}=Remora; 
  conv_curve(tt)=Best_F;
    end   
end


