%PSO Particle Swarm Optimization
% [GBEST, GPOS, cgCurve, CT] = PSO (X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
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

function [GBEST,  GPOS, cgCurve, CT] = PSO (X, y, No_P, fobj, N_Var, Max_Iter, LB, UB, verbose)
if nargin < 1
    error('MATLAB:notEnoughInputs', 'Please provide data for feature selection.');
end

if nargin < 2  % If only data is given, assume last column as target
    y = X(:, end);
    X = X(:, 1:end-1);
end

if nargin < 3  % Default 10 search agents
    No_P = 10;
end

if nargin < 4
    fobj = str2func('split_fitness'); % Apply feature selection
end

if nargin < 5
    N_Var = size(X, 2); % Apply feature selection on columns of X
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

% Extra variables for data visualization
% average_objective = zeros(1, maxIter);
cgCurve = zeros(1, Max_Iter);
FirstP_D1 = zeros(1 , Max_Iter);
position_history = zeros(No_P , Max_Iter , N_Var );

% Define the PSO's paramters
wMax = 0.9;
wMin = 0.2;
c1 = 2;
c2 = 2;
if length(UB)==1    % If same limit is applied on all variables
    UB = repmat(UB, 1, N_Var);
end
if length(LB)==1    % If same limit is applied on all variables
    LB = repmat(LB, 1, N_Var);
end
vMax = (UB - LB) .* 0.2;
vMin  = -vMax;

% The PSO algorithm

% Initialize the particles
for k = 1 : No_P
    % Particle position
    Swarm.Particles(k).X = (UB-LB) .* rand(1,N_Var) + LB;
    % Particle velocity
    Swarm.Particles(k).V = zeros(1, N_Var);
    
    % Best position of a particle
    Swarm.Particles(k).PBEST.X = zeros(1,N_Var);
    % Best fitness of a particle
    Swarm.Particles(k).PBEST.O = inf;
    
    % Global best particle position
    Swarm.GBEST.X = zeros(1,N_Var);
    % Global best particle fitness
    Swarm.GBEST.O = inf;
end


% Main loop
for tt = 1 : Max_Iter  
    % Calcualte the objective value
    for k = 1 : No_P      
        currentX = Swarm.Particles(k).X;
        position_history(k , tt , : ) = currentX;      
        
        Swarm.Particles(k).O = fobj(currentX  > (LB+UB)/2, X, y);
%         average_objective(t) =  average_objective(t)  + Swarm.Particles(k).O;
        
        % Update the PBEST
        if Swarm.Particles(k).O < Swarm.Particles(k).PBEST.O
            Swarm.Particles(k).PBEST.X = currentX;
            Swarm.Particles(k).PBEST.O = Swarm.Particles(k).O;
        end
        
        % Update the GBEST
        if Swarm.Particles(k).O < Swarm.GBEST.O
            Swarm.GBEST.X = currentX;
            Swarm.GBEST.O = Swarm.Particles(k).O;
        end
    end
    
    % Update the X and V vectors
    w = wMax - tt .* ((wMax - wMin) / Max_Iter);
    
    FirstP_D1(tt) = Swarm.Particles(1).X(1);
    
    for k = 1 : No_P
        Swarm.Particles(k).V = w .* Swarm.Particles(k).V ...
            + c1 .* rand(1,N_Var) .* (Swarm.Particles(k).PBEST.X - Swarm.Particles(k).X) ...
            + c2 .* rand(1,N_Var) .* (Swarm.GBEST.X - Swarm.Particles(k).X);
        
        % Check velocities are within limits
        ind = find(Swarm.Particles(k).V > vMax);
        Swarm.Particles(k).V(ind) = vMax(ind);
        
        ind = find(Swarm.Particles(k).V < vMin);      
        Swarm.Particles(k).V(ind) = vMin(ind);
        
        Swarm.Particles(k).X = Swarm.Particles(k).X + Swarm.Particles(k).V;
        
        % Check positions are within limits
        ind = find(Swarm.Particles(k).X > UB);
        Swarm.Particles(k).X(ind) = UB(ind);

        ind = find(Swarm.Particles(k).X < LB);
        Swarm.Particles(k).X(ind) = LB(ind);
    end
            
    cgCurve(tt) = Swarm.GBEST.O;
    if mod(tt, verbose) == 0  %Print best particle details at fixed iters
        fprintf('PSO: Iteration %d    fitness: %4.3f \n', tt, Swarm.GBEST.O);
    end
%     average_objective(t) = average_objective(t) / noP;
    
end
GBEST = Swarm.GBEST.O; % Global best fitness
GPOS = Swarm.GBEST.X;  % Global best position
CT = toc(timer);       % Total computation time in seconds

fprintf('PSO: Final fitness: %4.3f \n', Swarm.GBEST.O);

%% END OF PSO.m