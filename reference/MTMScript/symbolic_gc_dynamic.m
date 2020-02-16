function [Regressor_Matrix_Pos,Regressor_Matrix_Neg, Parameter_matrix, bool_Regressor_Mat] = symbolic_gc_dynamic(is_external_compensate, pol_order_arr)
%  Institute: The Chinese University of Hong Kong
%  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
%  Created on: 2018-10-05
%  Reference: This is a modified version of WPI Code
%               link1:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Regressor_and_Parameter_matrix/Torque_Script.m
%               link2:https://github.com/WPI-AIM/dvrk_gravity_comp/blob/master/Final_Submission/MATLAB%20code/Regressor_and_Parameter_matrix/Matrices_formation_script.m

% torques symbolic
syms g real;
syms m1 m2 m3 m4 m5 m6 m7 m4_parallel m5_parallel real;
syms cm1_x cm2_x cm3_x cm4_x cm5_x cm6_x cm7_x cm4_parallel_x cm5_parallel_x real;
syms cm1_y cm2_y cm3_y cm4_y cm5_y cm6_y cm7_y cm4_parallel_y cm5_parallel_y real;
syms cm1_z cm2_z cm3_z cm4_z cm5_z cm6_z cm7_z cm4_parallel_z cm5_parallel_z real;
syms L1 L2 L3 L4_y0 L4_z0 L5_y0 L5_z0 L6_z0 L6_x0 L3_parallel real;
syms q1 q2 q3 q4 q5 q6 q7 q3_parallel q4_parallel q5_parallel real;

if ~is_external_compensate
    Torque = symbolic_mtm_gravity_torque();
    % First of all Populate the parametetric matrix
    Parameter_matrix(1,1) = L2*m2+L2*m3+L2*m4+L2*m5+L2*m6+cm2_x*m2-cm4_parallel_x*m4_parallel;
    m_coeffs(1,1) = L2; m_coeffs(1,2) = m2;

    Parameter_matrix(2,1) = cm2_y*m2-cm4_parallel_y*m4_parallel;
    m_coeffs(2,1) = cm2_y; m_coeffs(2,2) = m2;

    Parameter_matrix(3,1) = L3*m3+L3*m4+L3*m5+L3*m6+cm3_x*m3+ L3_parallel*m4_parallel+ L3_parallel*m5_parallel-cm5_parallel_y*m5_parallel;
    m_coeffs(3,1) = L3;   m_coeffs(3,2) = m3;

    Parameter_matrix(4,1) = cm4_y*m4 +cm3_z*m3 +L4_z0*m4 +L4_z0*m5 +L4_z0*m6+ cm5_parallel_x*m5_parallel ;
    m_coeffs(4,1) =cm4_y; m_coeffs(4,2) = m4;

    Parameter_matrix(5,1) = cm4_x*m4;
    m_coeffs(5,1) =cm4_x; m_coeffs(5,2) = m4;

    Parameter_matrix(6,1) = -cm4_z*m4 + cm5_y*m5;
    m_coeffs(6,1) =cm5_y; m_coeffs(6,2) = m5;

    Parameter_matrix(7,1) = cm5_z*m5 +cm6_y*m6;
    m_coeffs(7,1) =cm5_z; m_coeffs(7,2) = m5;

    Parameter_matrix(8,1) = cm5_x*m5;
    m_coeffs(8,1) =cm5_x; m_coeffs(8,2) = m5;

    Parameter_matrix(9,1) = cm6_z*m6;
    m_coeffs(9,1) =cm6_z; m_coeffs(9,2) = m6;

    Parameter_matrix(10,1) = cm6_x*m6;
    m_coeffs(10,1) =cm6_x; m_coeffs(10,2) = m6;
    
    param_num = size(Parameter_matrix,1);
    Regressor_Matrix = sym(zeros(7,param_num));

    % bool_Regressor_Mat, if Regressor(i,j)~=Null, bool_Regressor_Mat=1; else bool_Regressor_Mat=0;
    bool_Regressor_Mat = sym(zeros(7,param_num));
    bool_Regressor_Mat(2,1:2) = sym(1);
    bool_Regressor_Mat(2:3,3:4) = sym(1);
    bool_Regressor_Mat(2:4,5:6) = sym(1);
    bool_Regressor_Mat(2:5,7:8) = sym(1);
    bool_Regressor_Mat(2:6,9:10) = sym(1);
    
        for i=1:7
        for j=1:size(bool_Regressor_Mat,2)
            if(bool_Regressor_Mat(i,j) == 1)
                if(m_coeffs(j,1) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,2));
                elseif(m_coeffs(j,2) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,1));
                else
                    Regressor_Matrix (i,j)  = Get_Cof(Torque(i),m_coeffs(j,2),m_coeffs(j,1));
                end
            else
                Regressor_Matrix (i,j)  = 0;
            end
        end
    end

    % Testing if T=Regressor_Matrix*Parameter_matrix
     T = Regressor_Matrix*Parameter_matrix;
     Diff  = simplify(T - Torque);
         for i=1:7
        for j=1:size(bool_Regressor_Mat,2)
            if(bool_Regressor_Mat(i,j) == 1)
                if(m_coeffs(j,1) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,2));
                elseif(m_coeffs(j,2) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,1));
                else
                    Regressor_Matrix (i,j)  = Get_Cof(Torque(i),m_coeffs(j,2),m_coeffs(j,1));
                end
            else
                Regressor_Matrix (i,j)  = 0;
            end
        end
    end

    % Testing if T=Regressor_Matrix*Parameter_matrix
     T = Regressor_Matrix*Parameter_matrix;
     Diff  = simplify(T - Torque)

     Regressor_Matrix_Pos = Regressor_Matrix;
     Regressor_Matrix_Neg = Regressor_Matrix;
else
    sym_pol_mat_pos = sym('sym_pol_mat_pos',[6 max(pol_order_arr)+1]);
    sym_pol_mat_neg = sym('sym_pol_mat_neg',[6 max(pol_order_arr)+1]);
    % polynomial symbolic
    for i=1:6
        for j=1:pol_order_arr(i)
            temp = sprintf('a%d_%d_pos', i, j);
            sym_pol_mat_pos(i,j+1) = sym(temp);
            temp = sprintf('a%d_%d_neg', i, j);
            sym_pol_mat_neg(i,j+1) = sym(temp);
        end
    end

    for i=1:6
        temp = sprintf('drift%d_pos', i);
        sym_pol_mat_pos(i,1) = sym(temp);
        temp = sprintf('drift%d_neg', i);
        sym_pol_mat_neg(i,1) = sym(temp);
    end

    % syms a6_1_pos a6_2_pos a6_3_pos a6_4_pos drift6_pos;
    % syms a5_1_pos a5_2_pos a5_3_pos a5_4_pos drift5_pos;
    % syms a4_1_pos a4_2_pos a4_3_pos a4_4_pos drift4_pos;
    % syms a3_1_pos a3_2_pos a3_3_pos a3_4_pos drift3_pos;
    % syms a2_1_pos a2_2_pos a2_3_pos a2_4_pos drift2_pos;
    % syms a1_1_pos a1_2_pos a1_3_pos a1_4_pos drift1_pos;
    % 
    % syms a6_1_neg a6_2_neg a6_3_neg a6_4_neg drift6_neg;
    % syms a5_1_neg a5_2_neg a5_3_neg a5_4_neg drift5_neg;
    % syms a4_1_neg a4_2_neg a4_3_neg a4_4_neg drift4_neg;
    % syms a3_1_neg a3_2_neg a3_3_neg a3_4_neg drift3_neg;
    % syms a2_1_neg a2_2_neg a2_3_neg a2_4_neg drift2_neg;
    % syms a1_1_neg a1_2_neg a1_3_neg a1_4_neg drift1_neg;


    Torque = symbolic_mtm_gravity_torque();
    Torque_1 = Torque;
    q_arr = [q1,q2,q3,q4,q5,q6];
    for i=1:size(sym_pol_mat_pos,1)
        for j=1:pol_order_arr(i)+1
            Torque(i) = Torque(i) + (sym_pol_mat_pos(i,j) + sym_pol_mat_neg(i,j))*q_arr(i)^(j-1);
        end
    end


    % Torque_1(1) = Torque_1(1) + (drift1_pos+drift1_neg) + (a1_1_pos+a1_1_neg)*q1 + (a1_2_pos+a1_2_neg)*q1^2 + (a1_3_pos+a1_3_neg)*q1^3 + (a1_4_pos+a1_4_neg)*q1^4;
    % Torque_1(2) = Torque_1(2) + (drift2_pos+drift2_neg) + (a2_1_pos+a2_1_neg)*q2 + (a2_2_pos+a2_2_neg)*q2^2 + (a2_3_pos+a2_3_neg)*q2^3 + (a2_4_pos+a2_4_neg)*q2^4;
    % Torque_1(3) = Torque_1(3) + (drift3_pos+drift3_neg) + (a3_1_pos+a3_1_neg)*q3 + (a3_2_pos+a3_2_neg)*q3^2 + (a3_3_pos+a3_3_neg)*q3^3 + (a3_4_pos+a3_4_neg)*q3^4;
    % Torque_1(4) = Torque_1(4) + (drift4_pos+drift4_neg) + (a4_1_pos+a4_1_neg)*q4 + (a4_2_pos+a4_2_neg)*q4^2 + (a4_3_pos+a4_3_neg)*q4^3 + (a4_4_pos+a4_4_neg)*q4^4;
    % Torque_1(5) = Torque_1(5) + (drift5_pos+drift5_neg) + (a5_1_pos+a5_1_neg)*q5 + (a5_2_pos+a5_2_neg)*q5^2 + (a5_3_pos+a5_3_neg)*q5^3 + (a5_4_pos+a5_4_neg)*q5^4;
    % Torque_1(6) = Torque_1(6) + (drift6_pos+drift6_neg) + (a6_1_pos+a6_1_neg)*q6 + (a6_2_pos+a6_2_neg)*q6^2 + (a6_3_pos+a6_3_neg)*q6^3 + (a6_4_pos+a6_4_neg)*q6^4;
    % diff2 = Torque - Torque_1;
    %%
    % First of all Populate the parametetric matrix
    Parameter_matrix(1,1) = L2*m2+L2*m3+L2*m4+L2*m5+L2*m6+cm2_x*m2-cm4_parallel_x*m4_parallel;
    m_coeffs(1,1) = L2; m_coeffs(1,2) = m2;

    Parameter_matrix(2,1) = cm2_y*m2-cm4_parallel_y*m4_parallel;
    m_coeffs(2,1) = cm2_y; m_coeffs(2,2) = m2;

    Parameter_matrix(3,1) = L3*m3+L3*m4+L3*m5+L3*m6+cm3_x*m3+ L3_parallel*m4_parallel+ L3_parallel*m5_parallel-cm5_parallel_y*m5_parallel;
    m_coeffs(3,1) = L3;   m_coeffs(3,2) = m3;

    Parameter_matrix(4,1) = cm4_y*m4 +cm3_z*m3 +L4_z0*m4 +L4_z0*m5 +L4_z0*m6+ cm5_parallel_x*m5_parallel ;
    m_coeffs(4,1) =cm4_y; m_coeffs(4,2) = m4;

    Parameter_matrix(5,1) = cm4_x*m4;
    m_coeffs(5,1) =cm4_x; m_coeffs(5,2) = m4;

    Parameter_matrix(6,1) = -cm4_z*m4 + cm5_y*m5;
    m_coeffs(6,1) =cm5_y; m_coeffs(6,2) = m5;

    Parameter_matrix(7,1) = cm5_z*m5 +cm6_y*m6;
    m_coeffs(7,1) =cm5_z; m_coeffs(7,2) = m5;

    Parameter_matrix(8,1) = cm5_x*m5;
    m_coeffs(8,1) =cm5_x; m_coeffs(8,2) = m5;

    Parameter_matrix(9,1) = cm6_z*m6;
    m_coeffs(9,1) =cm6_z; m_coeffs(9,2) = m6;

    Parameter_matrix(10,1) = cm6_x*m6;
    m_coeffs(10,1) =cm6_x; m_coeffs(10,2) = m6;

    for i=1:size(sym_pol_mat_pos,1)
        for j=1:pol_order_arr(i)+1
            Parameter_matrix(end+1,1) = sym_pol_mat_pos(i,j);
        end
    end

    for i=1:size(sym_pol_mat_neg,1)
        for j=1:pol_order_arr(i)+1
            Parameter_matrix(end+1,1) = sym_pol_mat_neg(i,j);
        end
    end 
    m_coeffs(11:size(Parameter_matrix,1),1) = Parameter_matrix(11:size(Parameter_matrix,1));
    m_coeffs(11:size(Parameter_matrix,1),2) = 1;


    % Parameter_matrix(11,1) = drift1_pos;
    % 
    % Parameter_matrix(12,1) = a1_1_pos;
    % 
    % Parameter_matrix(13,1) = a1_2_pos;
    % 
    % Parameter_matrix(14,1) = a1_3_pos;
    % 
    % Parameter_matrix(15,1) = a1_4_pos;
    % 
    % Parameter_matrix(16,1) = drift2_pos;
    % 
    % Parameter_matrix(17,1) = a2_1_pos;
    % 
    % Parameter_matrix(18,1) = a2_2_pos;
    % 
    % Parameter_matrix(19,1) = a2_3_pos;
    % 
    % Parameter_matrix(20,1) = a2_4_pos;
    % 
    % Parameter_matrix(21,1) = drift3_pos;
    % 
    % Parameter_matrix(22,1) = a3_1_pos;
    % 
    % Parameter_matrix(23,1) = a3_2_pos;
    % 
    % Parameter_matrix(24,1) = a3_3_pos;
    % 
    % Parameter_matrix(25,1) = a3_4_pos;
    % 
    % Parameter_matrix(26,1) = drift4_pos;
    % 
    % Parameter_matrix(27,1) = a4_1_pos;
    % 
    % Parameter_matrix(28,1) = a4_2_pos;
    % 
    % Parameter_matrix(29,1) = a4_3_pos;
    % 
    % Parameter_matrix(30,1) = a4_4_pos;
    % 
    % Parameter_matrix(31,1) = drift5_pos;
    % 
    % Parameter_matrix(32,1) = a5_1_pos;
    % 
    % Parameter_matrix(33,1) = a5_2_pos;
    % 
    % Parameter_matrix(34,1) = a5_3_pos;
    % 
    % Parameter_matrix(35,1) = a5_4_pos;
    % 
    % Parameter_matrix(36,1) = drift6_pos;
    % 
    % Parameter_matrix(37,1) = a6_1_pos;
    % 
    % Parameter_matrix(38,1) = a6_2_pos;
    % 
    % Parameter_matrix(39,1) = a6_3_pos;
    % 
    % Parameter_matrix(40,1) = a6_4_pos;
    % 
    % Parameter_matrix(41,1) = drift1_neg;
    % 
    % Parameter_matrix(42,1) = a1_1_neg;
    % 
    % Parameter_matrix(43,1) = a1_2_neg;
    % 
    % Parameter_matrix(44,1) = a1_3_neg;
    % 
    % Parameter_matrix(45,1) = a1_4_neg;
    % 
    % Parameter_matrix(46,1) = drift2_neg;
    % 
    % Parameter_matrix(47,1) = a2_1_neg;
    % 
    % Parameter_matrix(48,1) = a2_2_neg;
    % 
    % Parameter_matrix(49,1) = a2_3_neg;
    % 
    % Parameter_matrix(50,1) = a2_4_neg;
    % 
    % Parameter_matrix(51,1) = drift3_neg;
    % 
    % Parameter_matrix(52,1) = a3_1_neg;
    % 
    % Parameter_matrix(53,1) = a3_2_neg;
    % 
    % Parameter_matrix(54,1) = a3_3_neg;
    % 
    % Parameter_matrix(55,1) = a3_4_neg;
    % 
    % Parameter_matrix(56,1) = drift4_neg;
    % 
    % Parameter_matrix(57,1) = a4_1_neg;
    % 
    % Parameter_matrix(58,1) = a4_2_neg;
    % 
    % Parameter_matrix(59,1) = a4_3_neg;
    % 
    % Parameter_matrix(60,1) = a4_4_neg;
    % 
    % Parameter_matrix(61,1) = drift5_neg;
    % 
    % Parameter_matrix(62,1) = a5_1_neg;
    % 
    % Parameter_matrix(63,1) = a5_2_neg;
    % 
    % Parameter_matrix(64,1) = a5_3_neg;
    % 
    % Parameter_matrix(65,1) = a5_4_neg;
    % 
    % Parameter_matrix(66,1) = drift6_neg;
    % 
    % Parameter_matrix(67,1) = a6_1_neg;
    % 
    % Parameter_matrix(68,1) = a6_2_neg;
    % 
    % Parameter_matrix(69,1) = a6_3_neg;
    % 
    % Parameter_matrix(70,1) = a6_4_neg;

    % m_coeffs(11:70,1) = Parameter_matrix(11:70);
    % m_coeffs(11:70,2) = 1;


    param_num = size(Parameter_matrix,1);
    Regressor_Matrix = sym(zeros(7,param_num));

    % bool_Regressor_Mat, if Regressor(i,j)~=Null, bool_Regressor_Mat=1; else bool_Regressor_Mat=0;
    bool_Regressor_Mat = sym(zeros(7,param_num));
    bool_Regressor_Mat(2,1:2) = sym(1);
    bool_Regressor_Mat(2:3,3:4) = sym(1);
    bool_Regressor_Mat(2:4,5:6) = sym(1);
    bool_Regressor_Mat(2:5,7:8) = sym(1);
    bool_Regressor_Mat(2:6,9:10) = sym(1);



    start_index = 10;
    for i=1:6
        choose_index = start_index+1:start_index+pol_order_arr(i)+1;
        bool_Regressor_Mat(i,choose_index) = sym(1);
        start_index = start_index+pol_order_arr(i)+1;
    end

    start_index = (sum(pol_order_arr)+6)+10;
    for i=1:6
        choose_index = start_index+1:start_index+pol_order_arr(i)+1;
        bool_Regressor_Mat(i,choose_index) = sym(1);
        start_index = start_index+pol_order_arr(i)+1;
    end

    % bool_Regressor_Mat(1,11:15) = 1;
    % bool_Regressor_Mat(2,16:20) = 1;
    % bool_Regressor_Mat(3,21:25) = 1;
    % bool_Regressor_Mat(4,26:30) = 1;
    % bool_Regressor_Mat(5,31:35) = 1;
    % bool_Regressor_Mat(6,36:40) = 1;
    % 
    % bool_Regressor_Mat(1,41:45) = 1;
    % bool_Regressor_Mat(2,46:50) = 1;
    % bool_Regressor_Mat(3,51:55) = 1;
    % bool_Regressor_Mat(4,56:60) = 1;
    % bool_Regressor_Mat(5,61:65) = 1;
    % bool_Regressor_Mat(6,66:70) = 1;

    %%
    % 
    for i=1:7
        for j=1:size(bool_Regressor_Mat,2)
            if(bool_Regressor_Mat(i,j) == 1)
                if(m_coeffs(j,1) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,2));
                elseif(m_coeffs(j,2) == 1)
                     Regressor_Matrix (i,j)  = Get_One_Cof(Torque(i),m_coeffs(j,1));
                else
                    Regressor_Matrix (i,j)  = Get_Cof(Torque(i),m_coeffs(j,2),m_coeffs(j,1));
                end
            else
                Regressor_Matrix (i,j)  = 0;
            end
        end
    end

    % Testing if T=Regressor_Matrix*Parameter_matrix
     T = Regressor_Matrix*Parameter_matrix;
     Diff  = simplify(T - Torque)

    Regressor_Matrix_Pos = Regressor_Matrix;
    % Regressor_Matrix_Pos(1,41:45) = 0;
    % Regressor_Matrix_Pos(2,46:50) = 0;
    % Regressor_Matrix_Pos(3,51:55) = 0;
    % Regressor_Matrix_Pos(4,56:60) = 0;
    % Regressor_Matrix_Pos(5,61:65) = 0;
    % Regressor_Matrix_Pos(6,66:70) = 0;

    Regressor_Matrix_Neg = Regressor_Matrix;
    % Regressor_Matrix_Neg(1,11:15) = 0;
    % Regressor_Matrix_Neg(2,16:20) = 0;
    % Regressor_Matrix_Neg(3,21:25) = 0;
    % Regressor_Matrix_Neg(4,26:30) = 0;
    % Regressor_Matrix_Neg(5,31:35) = 0;
    % Regressor_Matrix_Neg(6,36:40) = 0;

    % 
    % for i=1:6
    %     Regressor_Matrix_Neg(i,(i-1)*pol_param_num+11: i*pol_param_num+10) = 0;
    %     Regressor_Matrix_Pos(i, (i+5)*pol_param_num+11: (i+6)*pol_param_num+10) = 0;
    % end

    start_index = 10;
    for i=1:6
        choose_index = start_index+1:start_index+pol_order_arr(i)+1;
        Regressor_Matrix_Neg(i,choose_index) = 0;
        start_index = start_index+pol_order_arr(i)+1;
    end

    start_index = (sum(pol_order_arr)+6)+10;
    for i=1:6
        choose_index = start_index+1:start_index+pol_order_arr(i)+1;
        Regressor_Matrix_Pos(i,choose_index) = 0;
        start_index = start_index+pol_order_arr(i)+1;
    end


end




















