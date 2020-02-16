function [rho,alphaList,aList,dList,thetaList] = symbolic_MTM_DH_table()
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/dvrk501mtm_getTorqueEq.m
    syms L4_z0 L2 L3 real
    syms q1 q2 q3 q4 q5 q6 q7 real;
    
     %types of joints: 1 for revolute, 0 for prismatic
    rho = [1 1 1 1 1 1 1];
    % alpha_i
    alphaList=[pi/2 0 -pi/2 pi/2 -pi/2 pi/2 0];
    % A_i
    aList=[0 L2 L3 0 0 0 0];
    % d_i
    dList=[0 0 0 L4_z0 0 0 0];
    % theta_i
    thetaList=[(q1-pi/2) (q2-pi/2) (q3+pi/2) q4 q5 (q6-pi/2) q7+pi/2];
end