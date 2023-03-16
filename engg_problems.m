%% Cantilever Beam Design (CBD)

f = @(x) 0.0624*(x(1)+x(2)+x(3)+x(4)+x(5));
g = @(x) (61/x(1)^3+37/x(2)^3+19/x(3)^3+7/x(4)^3+1/x(5)^3)-1;

%% Three-bar truss design (TBD)
f = @(x) (2*sqrt(2*x(1))+x(1))*100;
g1 = @(x) ((sqrt(2)*x(1)+x(2))/(sqrt(2)*x(1)^2+2*x(1)*x(2)))*2 -2;
g2 = @(x) (x(2)/(sqrt(2)*x(1)^2+2*x(1)*x(2)))*2 -2;
g3 = @(x) (1/(sqrt(2)*x(2)+x(1)))*2 -2;

%% Pressure Vessel Design (PVD)
 f = @(x) 0.6224*x(1)*x(2)*x(3) + 1.7781*x(2)*x(3)^2 +...
     3.1661*x(1)^2*x(4) + 19.84*x(1)^2*x(3);
 
g1 = @(x) -x(1)+0.0193*x(3);
g2 = @(x) -x(3)+0.00954*x(3);
g3 = @(x) -pi*x(3)^2*x(4)-(4/3)*pi*x(3)^3+1296000;
g4 = @(x) x(4)-240;
