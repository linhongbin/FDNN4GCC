function Torque = symbolic_mtm_gravity_torque()
%  Reference: This is code from WPI
%               link:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Symbolic_Torques/dvrk501mtm_getTorqueEq.m
    syms g real;
    syms m1 m2 m3 m4 m5 m6 m7 m4_parallel m5_parallel real;
    syms cm1_x cm2_x cm3_x cm4_x cm5_x cm6_x cm7_x cm4_parallel_x cm5_parallel_x real;
    syms cm1_y cm2_y cm3_y cm4_y cm5_y cm6_y cm7_y cm4_parallel_y cm5_parallel_y real;
    syms cm1_z cm2_z cm3_z cm4_z cm5_z cm6_z cm7_z cm4_parallel_z cm5_parallel_z real;
    syms L1 L2 L3 L4_y0 L4_z0 L5_y0 L5_z0 L6_z0 L6_x0 L3_parallel real;
    syms q1 q2 q3 q4 q5 q6 q7 q3_parallel q4_parallel q5_parallel real;
    q = [q1, q2, q3, q4, q5, q6, q7];

    [rho,alphaList,aList,dList,thetaList] = symbolic_MTM_DH_table();
    DH_table = [thetaList' dList' aList' alphaList'];

    gravity_vector = [0 0 -g]'; %negative direction in y-axiss        
    
    m = [m1 m2 m3 m4 m5 m6 m7];
    cm1 = [cm1_x cm1_y cm1_z]';
    cm2 = [cm2_x cm2_y cm2_z]';
    cm3 = [cm3_x cm3_y cm3_z]';
    cm4 = [cm4_x cm4_y cm4_z]';
    cm5 = [cm5_x cm5_y cm5_z]';
    cm6 = [cm6_x cm6_y cm6_z]';
    cm7 = [cm7_x cm7_y cm7_z]';
    cm = [cm1 cm2 cm3 cm4 cm5 cm6 cm7]; %respect to generalized (local) coordinate
    pE_MTM = symbolic_potentialEnergy(DH_table, m, cm, gravity_vector, 1,6);
    
    
    [rho_parallel,alpha_parallel_List,a_parallel_List,d_parallel_List,theta_parallel_List] = symbolic_parallel_bar_DH_table();
    cm4_parallel = [cm4_parallel_x cm4_parallel_y cm4_parallel_z]';
    cm5_parallel = [cm5_parallel_x cm5_parallel_y cm5_parallel_z]';
    cm_parallel = [cm1 cm2 cm3 cm4_parallel cm5_parallel];
    m_parallel = [m1 m2 m3 m4_parallel m5_parallel];
    DH_table_parallel = [theta_parallel_List' d_parallel_List' a_parallel_List' alpha_parallel_List'];
    pE_parallel = symbolic_potentialEnergy(DH_table_parallel, m_parallel, cm_parallel, gravity_vector, 4,5);
    
    pE = pE_MTM+pE_parallel;
    
    torque = sym('t_%d', [7,1]);
    for i = 1:length(q)
        torque(i) = -diff(pE, q(i));
    end
    Torque = simplify(torque);
end
