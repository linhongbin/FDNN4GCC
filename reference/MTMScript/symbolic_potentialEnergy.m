function P = symbolic_potentialEnergy(dh_table,m,cm,gravity_vector, start_link_No, stop_link_No)
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/potentialEnergy.m

    P = sym(0);
    for i = start_link_No:stop_link_No
        rc = symbolic_ForwardKinematics(i,dh_table,cm);
        P = P + m(i)*gravity_vector'*rc;
    end 
end