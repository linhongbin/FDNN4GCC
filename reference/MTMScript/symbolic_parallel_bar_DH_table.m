function [rho,alphaList,aList,dList,thetaList] = symbolic_parallel_bar_DH_table()
%  Reference: This is a modified version of WPI Code
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/DHtransform.m
    syms L1 L2 L3 L4_y0 L4_z0 L3_parallel real;
    syms q1 q2 q3 q4 q5 q6 q7 q4_parallel q5_parallel real;
     
    % parrallel mechanism constaint
    q4_parallel = -q3;
    q5_parallel = q3;
    
     %types of joints: 1 for revolute, 0 for prismatic
    rho = [1 1 1 1 1];
    % alpha_i
    alphaList=[pi/2 0 0 0 0];
    % A_i
    aList=[0 L2 L3_parallel L2 0];
    % d_i
    dList=[0 0 0 0 0];
    % theta_i
    thetaList=[(q1-pi/2) (q2-pi/2) (q3+pi/2) (q4_parallel+pi/2)   q5_parallel];
end