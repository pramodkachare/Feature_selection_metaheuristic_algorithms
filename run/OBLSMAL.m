% Source codes demo version 1.0
% ------------------------------------------------------------------------------------------------------------
% Main paper (Please refer to the main paper):
% Slime Mould Algorithm: A New Method for Stochastic Optimization
% Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili
% Future Generation Computer Systems,2020
% DOI: https://doi.org/10.1016/j.future.2020.03.055
% https://www.sciencedirect.com/science/article/pii/S0167739X19320941
% ------------------------------------------------------------------------------------------------------------
% Website of SMA: http://www.alimirjalili.com/SMA.html
% You can find and run the SMA code online at http://www.alimirjalili.com/SMA.html

% You can find the SMA paper at https://doi.org/10.1016/j.future.2020.03.055
% Please follow the paper for related updates in researchgate: https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization
% ------------------------------------------------------------------------------------------------------------
%  Main idea: Shimin Li
%  Author and programmer: Shimin Li,Ali Asghar Heidari,Huiling Chen
%  e-Mail: simonlishimin@foxmail.com
% ------------------------------------------------------------------------------------------------------------
%  Co-author:
%             Huiling Chen(chenhuiling.jlu@gmail.com)
%             Mingjing Wang(wangmingjing.style@gmail.com)
%             Ali Asghar Heidari(aliasghar68@gmail.com, as_heidari@ut.ac.ir)
%             Seyedali Mirjalili(ali.mirjalili@gmail.com)
%             
%             Researchgate: Ali Asghar Heidari https://www.researchgate.net/profile/Ali_Asghar_Heidari
%             Researchgate: Seyedali Mirjalili https://www.researchgate.net/profile/Seyedali_Mirjalili
%             Researchgate: Huiling Chen https://www.researchgate.net/profile/Huiling_Chen
% ------------------------------------------------------------------------------------------------------------
% _____________________________________________________
%  Co-author and Advisor: Seyedali Mirjalili
%
%         e-Mail: ali.mirjalili@gmail.com
%
%       Homepage: http://www.alimirjalili.com
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Max_iter: maximum iterations, N: populatoin size, Convergence_curve: Convergence curve
% To run SMA: [Destination_fitness,bestPositions,Convergence_curve]=SMA(N,Max_iter,lb,ub,dim,fobj)
function [Destination_fitness,bestPositions,Convergence_curve]=OBLSMAL(N,Max_iter,lb,ub,dim,fobj,data, target)
% disp('SMA is now tackling your problem')

% initialize position
bestPositions=zeros(1,dim);
Destination_fitness=inf;%change this to -inf for maximization problems
AllFitness = inf*ones(N,1);%record the fitness of all slime mold
weight = ones(N,dim);%fitness weight of each slime mold
%Initialize the set of random solutions





%initializing boundary for opposition
Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    for i=1:dim
        upper(1,i)=ub;
        lower(1,i)=lb;
    end
    % If each variable has a different lb and ub
else
    for i=1:dim
        upper(1,i)=ub(i);
        lower(1,i)=lb(i);
    end
end





X=initialization(N,dim,ub,lb);
Convergence_curve=zeros(1,Max_iter);
it=1;  %Number of iterations
lb=ones(1,dim).*lb; % lower boundary 
ub=ones(1,dim).*ub; % upper boundary
z=0.03; % parameter

% Main loop
while  it <= Max_iter
    
    %sort the fitness
    
    if rand <0.9
        
        
        
    for i=1:N
        % Check if solutions go outside the search space and bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        AllFitness(i) = fobj(X(i,:), data, target);
    end
    
    [SmellOrder,SmellIndex] = sort(AllFitness);  %Eq.(2.6)
    worstFitness = SmellOrder(N);
    bestFitness = SmellOrder(1);

    S=bestFitness-worstFitness+eps;  % plus eps to avoid denominator zero

    %calculate the fitness weight of each slime mold
    for i=1:N
        for j=1:dim
            if i<=(N/2)  %Eq.(2.5)
                weight(SmellIndex(i),j) = 1+rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            else
                weight(SmellIndex(i),j) = 1-rand()*log10((bestFitness-SmellOrder(i))/(S)+1);
            end
        end
    end
    
    %update the best fitness value and best position
    if bestFitness < Destination_fitness
        bestPositions=X(SmellIndex(1),:);
        Destination_fitness = bestFitness;
    end
    
    a = atanh(-(it/Max_iter)+1);   %Eq.(2.4)
    b = 1-it/Max_iter;
    
% % %         for x=1:size(X,1)
% % %             for y=1:size(X,2)
% % %                 if upper(1,y)<X(x,y)
% % %                     upper(1,y)=X(x,y);
% % %                 end
% % %                 if lower(1,y)>X(x,y)
% % %                     lower(1,y)=X(x,y);
% % %                 end
% % %             end
% % %         end
    
    
    
        aa=2-it*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Oppose the least fitness elements
    threshold=aa;
    X=corOppose(X,AllFitness,ub,lb,upper,lower,dim,threshold);
    
    
    % Update the Position of search agents
    for i=1:N
        if rand<z     %Eq.(2.7)
            X(i,:) = (ub-lb)*rand+lb;
        else
            p =tanh(abs(AllFitness(i)-Destination_fitness));  %Eq.(2.2)
            vb = unifrnd(-a,a,1,dim);  %Eq.(2.3)
            vc = unifrnd(-b,b,1,dim);
            for j=1:dim
                r = rand();
                A = randi([1,N]);  % two positions randomly selected from population
                B = randi([1,N]);
                if r<p    %Eq.(2.1)
                    X(i,j) = bestPositions(j)+ vb(j)*(weight(i,j)*X(A,j)-X(B,j));
                else
                    X(i,j) = vc(j)*X(i,j);
                end
            end
        end
    end
    
    
    
    
    
    
    
    
       else
       for i=1:N
           for j=1:dim
   X(i,j) = bestPositions(1,j)*Levy(1);  
           end
       end
       
    end
   
    
    Convergence_curve(it)=Destination_fitness;
    it=it+1;
    
%         if mod(it,1)==0
        display(['At iteration ', num2str(it), ' the fitness value is ', num2str(Destination_fitness)]);
%     end
end

end



function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end