function T = symbolic_DHtransform(theta,d,a,alpha)
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/DHtransform.m
%take in four standard DH parameters between two consecutive frames and 
%return 4x4 homogeneous intermediate transformation matrix between
%the links
T = symbolic_transl(d,'z')*symbolic_rot(theta,'z')*symbolic_transl(a,'x')*symbolic_rot(alpha,'x');
    for r=1:4
        for c=1:4
            if isa(T(r,c), 'sym')
            elseif abs(T(r,c))<1e-4
               T(r,c)=0;
            end
        end
    end
end
