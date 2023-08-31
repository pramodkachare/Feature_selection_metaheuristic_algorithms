%SSA Salp Swarm Algorithm
% [GBEST, GPOS, cgCurve, CT] = PSO (X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
% 
%   Main paper: Mirjalili, S., Gandomi, A. H., Mirjalili, S. Z., Saremi, S.,
%               Faris, H., Mirjalili, S. M. (2017). 
%               Salp Swarm Algorithm: A bio-inspired optimizer for 
%               engineering design problems
%               Advances in Engineering Software, 114, 163-191.
%               DOI: 10.1016/j.advengsoft.2017.07.002
% 
%     [GBEST, GPOS] = PSO(X) applies feature selection on M-by-N matrix X
%     with N examples and assuming last column as the classification target 
%     and returns the best fitness value GBEST and 1-by-(M-1) logical matrix
%     of selected features GPOS.
%
%     [GBEST, GPOS] = PSO(X, y) applies feature selection on M-by-N feature 
%     matrix X and 1-by-N target matrix y and returns the best fitness value
%     GBEST and 1-by-(M-1) logical matrix of selected features GPOS.
%     
%     Example:
%
%
% Original Author: Dr. Seyedali Mirjalili
% Revised by : Pramod H. Kachare (Aug 2023)


function [FoodFitness,FoodPosition,Convergence_curve]=SSA(N,Max_iter,lb,ub,dim,fobj, X, y)

if size(ub,2)==1
    ub=ones(1, dim)*ub;
    lb=ones(1, dim)*lb;
end

Convergence_curve = zeros(1,Max_iter);

%Initialize the positions of salps
SalpPositions=initializations(N,dim,ub,lb);


FoodPosition=zeros(1,dim);
FoodFitness=inf;


%calculate the fitness of initial salps

for i=1:size(SalpPositions,1)
%     SalpFitness(1,i)=fobj(SalpPositions(i,:));
             SalpFitness(1,i) = feval(fobj,SalpPositions(i,:)', X, y);

end

[sorted_salps_fitness,sorted_indexes]=sort(SalpFitness);

for newindex=1:N
    Sorted_salps(newindex,:)=SalpPositions(sorted_indexes(newindex),:);
end

FoodPosition=Sorted_salps(1,:);
FoodFitness=sorted_salps_fitness(1);

fprintf('SSA: Iteration %d    fitness: %4.3f \n', 1, FoodFitness);
Convergence_curve(1)=FoodFitness;

%Main loop
l=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
while l<Max_iter+1
    
    c1 = 2*exp(-(4*l/Max_iter)^2); % Eq. (3.2) in the paper
    
    for i=1:size(SalpPositions,1)
        
        SalpPositions= SalpPositions';
        
        if i<=N/2
            for j=1:1:dim
                c2=rand();
                c3=rand();
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if c3<0.5 
                    SalpPositions(j,i)=FoodPosition(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                else
                    SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif i>N/2 && i<N+1
            point1=SalpPositions(:,i-1);
            point2=SalpPositions(:,i);
            
            SalpPositions(:,i)=(point2+point1)/2; % % Eq. (3.4) in the paper
        end
        
        SalpPositions= SalpPositions';
    end
    
    for i=1:size(SalpPositions,1)
        
        Tp=SalpPositions(i,:)>ub;
        Tm=SalpPositions(i,:)<lb;
        SalpPositions(i,:)= SalpPositions(i,:).*~(Tp+Tm) + ub.*Tp + lb.*Tm;
        
%         SalpFitness(1,i)=fobj(SalpPositions(i,:));
                     SalpFitness(1,i) = feval(fobj,SalpPositions(i,:)', X, y);

        
        if SalpFitness(1,i)<FoodFitness
            FoodPosition=SalpPositions(i,:);
            FoodFitness=SalpFitness(1,i);
            
        end
    end
    
    fprintf('SSA: Iteration %d    fitness: %4.3f \n', l, FoodFitness);
    Convergence_curve(l)=FoodFitness;
    l = l + 1;
    
    % PUT THIS CODE INSIDE LOOP OF ITERATIONS
    % Normalize particle position values

    % Population diversity as a whole
end
fprintf('SSA: Final fitness: %4.3f \n', FoodFitness);


function Positions=initializations(SearchAgents_no,dim,ub,lb)
    Boundary_no= length(ub); % numnber of boundaries

    % If the boundaries of all variables are equal and user enter a signle
    % number for both ub and lb
    if Boundary_no==1
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end

    % If each variable has a different lb and ub
    if Boundary_no>1
        for i=1:dim
            ub_i=ub(i);
            lb_i=lb(i);
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end



