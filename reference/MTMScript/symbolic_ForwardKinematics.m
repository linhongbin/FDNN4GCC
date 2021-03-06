function p = symbolic_ForwardKinematics (i,dh_table,cm)
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/ForwardKinematics.m
% return position of center of mass of ith link
T = sym(eye(4));
for j=1:i
    theta = dh_table(j,1);
    d = dh_table(j,2);
    a = dh_table(j,3);
    alpha = dh_table(j,4);
    T = T*symbolic_DHtransform(theta,d,a,alpha);
    %disp(T);
end
T = T*symbolic_transl(cm(:,i),'all');
p = simplify(T(1:3,4));
end

