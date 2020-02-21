% addpath('./model/CAD')

% joint limits
upper_limit = deg2rad([30,45,34,190,175,40]);
lower_limit = deg2rad([-30,-14,-34,-80,-85,-40]);
% coupling_index_list = {[2,3]};
% coupling_upper_limit = [41];
% coupling_lower_limit = [-11];

sample_num_each_joint = 4;
X1 = linspace(lower_limit(1), upper_limit(1), sample_num_each_joint);
X2 = linspace(lower_limit(2), upper_limit(2), sample_num_each_joint);
X3 = linspace(lower_limit(3), upper_limit(3), sample_num_each_joint);
X4 = linspace(lower_limit(4), upper_limit(4), sample_num_each_joint);
X5 = linspace(lower_limit(5), upper_limit(5), sample_num_each_joint);
X6 = linspace(lower_limit(6), upper_limit(6), sample_num_each_joint);
    
input_mat = [];
output_mat = [];
count =0;
for idx1 = 1:size(X1,2)
    for idx2 = 1:size(X2,2)
        for idx3 = 1:size(X3,2)
            for idx4 = 1:size(X4,2)
                for idx5 = 1:size(X5,2)
                    for idx6 = 1:size(X6,2)
                        input = [X1(idx1);X2(idx2);X3(idx3);X4(idx4);X5(idx5);X6(idx6)];
                        output = MTMR28002Gravity(input(1),input(2),input(3),input(4),input(5),input(6));
                        input_mat = [input_mat; input.'];
                        output_mat = [output_mat; output.'];
                    end
                end
            end
            count = count +1;
            file_name = sprintf('MTMR_CAD_sim_%d.mat', count);
            save(file_name, 'input_mat', 'output_mat')
            input_mat = [];
            output_mat = [];
        end
        fprintf('Progress = %d %%\n',int32(double(idx2+(idx1-1)*size(X2,2))*100/double((size(X1,2)* size(X2,2)))))
    end
end



function torques = MTMR28002Gravity(q1,q2,q3,q4,q5,q6)
    dynamic_vec = CAD_dynamic_vec();
    torques = CAD_analytical_regressor(9.81,q2,q3,q4,q5,q6)*dynamic_vec;
end



function Regressor_Matrix = CAD_analytical_regressor(g,q2,q3,q4,q5,q6)
    Regressor_Matrix = ...
    [         0,         0,                                     0,                                       0,                                                                                                                                      0,                                                                                                                                                                                                                                             0,                                                     0,                                                             0,                                                                                                                                      0,                                                                                                                                                                                                                                              0,             0, 0,  0, 0
     g*sin(q2), g*cos(q2), g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2),          g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3),         g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3),         g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4),          g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3),          g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6),             0, 1,  0, 0
             0,         0,                        g*cos(q2 + q3),                         -g*sin(q2 + q3), -(g*(2*cos(q2)*cos(q5)*sin(q3) + 2*cos(q3)*cos(q5)*sin(q2) + 2*cos(q2)*cos(q3)*cos(q4)*sin(q5) - 2*cos(q4)*sin(q2)*sin(q3)*sin(q5)))/2, (g*(2*cos(q2)*cos(q3)*sin(q4)*sin(q6) + 2*cos(q2)*cos(q6)*sin(q3)*sin(q5) + 2*cos(q3)*cos(q6)*sin(q2)*sin(q5) - 2*sin(q2)*sin(q3)*sin(q4)*sin(q6) - 2*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6) + 2*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3)))/2,         (g*(cos(q2 + q3 + q4) + cos(q2 + q3 - q4)))/2, (g*(2*sin(q2)*sin(q3)*sin(q4) - 2*cos(q2)*cos(q3)*sin(q4)))/2, -(g*(2*cos(q2)*sin(q3)*sin(q5) + 2*cos(q3)*sin(q2)*sin(q5) - 2*cos(q2)*cos(q3)*cos(q4)*cos(q5) + 2*cos(q4)*cos(q5)*sin(q2)*sin(q3)))/2, -(g*(2*cos(q6)*sin(q2)*sin(q3)*sin(q4) - 2*cos(q2)*cos(q3)*cos(q6)*sin(q4) + 2*cos(q2)*sin(q3)*sin(q5)*sin(q6) + 2*cos(q3)*sin(q2)*sin(q5)*sin(q6) - 2*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6) + 2*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6)))/2, -cos(q2 + q3), 0,  0, 0
             0,         0,                                     0,                                       0,                                                                                                         g*sin(q2 + q3)*sin(q4)*sin(q5),                                                                                                                                                                                    g*sin(q2 + q3)*(cos(q4)*sin(q6) + cos(q5)*cos(q6)*sin(q4)),                               -g*sin(q2 + q3)*sin(q4),                                       -g*sin(q2 + q3)*cos(q4),                                                                                                        -g*sin(q2 + q3)*cos(q5)*sin(q4),                                                                                                                                                                                     g*sin(q2 + q3)*(cos(q4)*cos(q6) - cos(q5)*sin(q4)*sin(q6)),             0, 0,  0, 0
             0,         0,                                     0,                                       0,             -g*(cos(q2)*cos(q3)*sin(q5) - sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q2)),                                                                                     g*(cos(q5)*cos(q6)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5)*cos(q6) + cos(q2)*cos(q4)*cos(q6)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*cos(q6)*sin(q2)*sin(q5)),                                                     0,                                                             0,             -g*(cos(q5)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5) + cos(q2)*cos(q4)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*sin(q2)*sin(q5)),                                                                                     -g*(cos(q5)*sin(q2)*sin(q3)*sin(q6) - cos(q2)*cos(q3)*cos(q5)*sin(q6) + cos(q2)*cos(q4)*sin(q3)*sin(q5)*sin(q6) + cos(q3)*cos(q4)*sin(q2)*sin(q5)*sin(q6)),             0, 0, q5, 1
             0,         0,                                     0,                                       0,                                                                                                                                      0,                 g*(cos(q2)*cos(q6)*sin(q3)*sin(q4) + cos(q3)*cos(q6)*sin(q2)*sin(q4) + cos(q2)*cos(q3)*sin(q5)*sin(q6) - sin(q2)*sin(q3)*sin(q5)*sin(q6) + cos(q2)*cos(q4)*cos(q5)*sin(q3)*sin(q6) + cos(q3)*cos(q4)*cos(q5)*sin(q2)*sin(q6)),                                                     0,                                                             0,                                                                                                                                      0,                  g*(cos(q2)*cos(q3)*cos(q6)*sin(q5) - cos(q2)*sin(q3)*sin(q4)*sin(q6) - cos(q3)*sin(q2)*sin(q4)*sin(q6) - cos(q6)*sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*cos(q6)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*cos(q6)*sin(q2)),             0, 0,  0, 0];

end
function  dynamic_vec= CAD_dynamic_vec()
%Contribute by Luke and Web from Anton MTMR
cm2_x =-0.38;  
cm2_y = 0.00;
cm2_z =0.00;
m2 = 0.65;

cm3_x = -0.25; 
cm3_y = 0.00 ;
cm3_z = 0.00;
m3 = 0.04;

cm4_x = 0.0; 
cm4_y = -0.084;
cm4_z = -0.12;
m4 = 0.14;

cm5_x = 0.0; 
cm5_y = 0.036; 
cm5_z = -0.065;
m5 = 0.04;

cm6_x = 0.0; 
cm6_y = -0.025; 
cm6_z = 0.05;
m6 = 0.05;

L2 = 0.2794;
L3 = 0.3645;
L4_z0 =  0.1506;

counter_balance = 0.54;
cable_offset = 0.33;
drift2 = -cable_offset;
E5 = 0.007321;
drift5 = - 0.0065;


Parameter_matrix(1,1)  = L2*m2+L2*m3+L2*m4+L2*m5+L2*m6+cm2_x*m2;
Parameter_matrix(2,1)  = cm2_y*m2;
Parameter_matrix(3,1)  = L3*m3+L3*m4+L3*m5+L3*m6+cm3_x*m3;
Parameter_matrix(4,1)  = cm4_y*m4 +cm3_z*m3 +L4_z0*m4 +L4_z0*m5 +L4_z0*m6 ;
Parameter_matrix(5,1)  = cm5_z*m5 +cm6_y*m6;
Parameter_matrix(6,1)  = cm6_z*m6 ;
Parameter_matrix(7,1)  = cm4_x*m4;
Parameter_matrix(8,1)  = - cm4_z*m4 + cm5_y*m5;
Parameter_matrix(9,1) = cm5_x*m5;
Parameter_matrix(10,1) = cm6_x*m6;
Parameter_matrix(11,1) = counter_balance;
Parameter_matrix(12,1) = drift2;
Parameter_matrix(13,1) = E5;
Parameter_matrix(14,1) = drift5;

double(Parameter_matrix);

dynamic_vec =  Parameter_matrix;
end


