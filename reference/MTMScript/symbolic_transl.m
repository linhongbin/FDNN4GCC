function T = symbolic_transl(d,ax)
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/transl.m

T = sym(zeros(4));
R = sym(eye(3));
p = sym(zeros(3,1));

if lower(ax) == 'x'
    p(1) = d;
elseif lower(ax) == 'y'
    p(2) = d;
elseif lower(ax) == 'z'
    p(3) = d;
elseif strcmp(ax,'all')
    p = reshape(d,3,1);
else 
    error('not a standard axis')
    
end
T = [R p;...
    zeros(1,3) 1];
end
