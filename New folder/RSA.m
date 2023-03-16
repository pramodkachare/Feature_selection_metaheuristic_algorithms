function [Best_F,Best_P,Conv]=RSA(N,T,LB,UB,Dim,F_obj, data, target)
    Best_P=zeros(1,Dim);           % best positions
    Best_F=inf;                    % best fitness
    X=initialization(N,Dim,UB,LB); % Initialize the positions of solution
    Xnew=zeros(N,Dim);
    Conv=zeros(1,T);               % Convergance array


    t=1;                         % starting iteration
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


    while t<T+1  %Main loop %Update the Position of solutions
        ES=2*randn*(1-(t/T));  % Probability Ratio
        for i=2:size(X,1) 
            for j=1:size(X,2)  
                    R=Best_P(1,j)-X(randi([1 size(X,1)]),j)/((Best_P(1,j))+eps);
                    P=Alpha+(X(i,j)-mean(X(i,:)))./(Best_P(1,j).*(UB-LB)+eps);
                    Eta=Best_P(1,j)*P;
                    if (t<T/4)
                        Xnew(i,j)=Best_P(1,j)-Eta(1,j)*Beta-R*rand;    
                    elseif (t<2*T/4 && t>=T/4)
                        Xnew(i,j)=Best_P(1,j)*X(randi([1 size(X,1)]),j)*ES*rand;
                    elseif (t<3*T/4 && t>=2*T/4)
                        Xnew(i,j)=Best_P(1,j)*P(1,j)*rand;
                    else
                        Xnew(i,j)=Best_P(1,j)-Eta(1,j)*eps-R*rand;
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

        if mod(t,1)==0  %Print the best universe details after every t iterations
            fprintf('RSA: Iteration %d    fitness: %4.3f \n', t, Best_F);
        end
        t=t+1;
    end
    fprintf('RSA: Final fitness: %4.3f \n', Best_F);
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