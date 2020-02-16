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