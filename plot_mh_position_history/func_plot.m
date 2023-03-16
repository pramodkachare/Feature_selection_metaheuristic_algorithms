% This function draw the benchmark functions

function h = func_plot(K, ver)
func_name = ['cec', num2str(K)];
if ver == 17
    [lb,ub,dim,fobj]=CEC2017(func_name);
    
    switch func_name 
        case 'cec1' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec2' 
            x=-100:2:100; y=x; %[-10,10]
        case 'cec3' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec4' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec5' 
            x=-200:2:200; y=x; %[-5,5]
        case 'cec6' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec7' 
            x=-1:0.03:1;  y=x;  %[-1,1]
        case 'cec8' 
            x=-500:10:500;y=x; %[-500,500]
        case 'cec9' 
            x=-5:0.1:5;   y=x; %[-5,5]    
        case 'cec10' 
            x=-20:0.5:20; y=x;%[-500,500]
        case 'cec11' 
            x=-500:10:500; y=x;%[-0.5,0.5]
        case 'cec12' 
            x=-10:0.1:10; y=x;%[-pi,pi]
        case 'cec13' 
            x=-5:0.08:5; y=x;%[-3,1]
        case 'cec14' 
            x=-100:2:100; y=x;%[-100,100]
        case 'cec15' 
            x=-5:0.1:5; y=x;%[-5,5]
        case 'cec16' 
            x=-1:0.01:1; y=x;%[-5,5]
        case 'cec17' 
            x=-5:0.1:5; y=x;%[-5,5]
        case 'cec18' 
            x=-5:0.06:5; y=x;%[-5,5]
        case 'cec19' 
            x=-5:0.1:5; y=x;%[-5,5]
        case 'cec20' 
            x=-5:0.1:5; y=x;%[-5,5]        
        case 'cec21' 
            x=-5:0.1:5; y=x;%[-5,5]
        case 'cec22' 
            x=-5:0.1:5; y=x;%[-5,5]     
        case 'cec23' 
            x=-5:0.1:5; y=x;%[-5,5]  
    end 
else
    [lb,ub,dim,fobj]=CEC2019(func_name);
    
    switch func_name 
        case 'cec1' 
            x=-8192:20:8192; y=x; %[-100,100]
        case 'cec2' 
            x=-100:2:100; y=x; %[-10,10]
        case 'cec3' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec4' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec5' 
            x=-200:2:200; y=x; %[-5,5]
        case 'cec6' 
            x=-100:2:100; y=x; %[-100,100]
        case 'cec7' 
            x=-1:0.03:1;  y=x;  %[-1,1]
        case 'cec8' 
            x=-500:10:500;y=x; %[-500,500]
        case 'cec9' 
            x=-5:0.1:5;   y=x; %[-5,5]    
        case 'cec10' 
            x=-100:2:100; y=x;%[-100,100]
    end 
end

   

    

L=length(x);
f=[];

for i=1:L
    for j=1:L
        if strcmp(func_name,'F15')==0 && strcmp(func_name,'cec19')==0 && strcmp(func_name,'cec20')==0 && strcmp(func_name,'cec21')==0 && strcmp(func_name,'cec22')==0 && strcmp(func_name,'cec23')==0
            f(i,j)=fobj([x(i),y(j)]);
        end
        if strcmp(func_name,'cec15')==1
            f(i,j)=fobj([x(i),y(j),0,0]);
        end
        if strcmp(func_name,'cec19')==1
            f(i,j)=fobj([x(i),y(j),0]);
        end
        if strcmp(func_name,'cec20')==1
            f(i,j)=fobj([x(i),y(j),0,0,0,0]);
        end       
        if strcmp(func_name,'cec21')==1 || strcmp(func_name,'cec22')==1 ||strcmp(func_name,'cec23')==1
            f(i,j)=fobj([x(i),y(j),0,0]);
        end          
    end
end

[~, h] = contour(x,y,f);

end

