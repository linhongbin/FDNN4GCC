addpath('./gen_code')


% joint limits
joint_pos_upper_limit = [30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
coupling_index_list = {[2,3]};
coupling_upper_limit = [41];
coupling_lower_limit = [-11];

traj_pivot_points_num = 100;

s = rng;
gen_mat = [];
bool_gen_mat = [];
while(size(gen_mat,2) ~= traj_pivot_points_num)
    alpha = rand(1,6);
    data_tmp = diag(alpha)*joint_pos_upper_limit.'+(diag([1,1,1,1,1,1]) -diag(alpha))*joint_pos_lower_limit.';
    if hw_joint_space_check(data_tmp.',joint_pos_upper_limit,joint_pos_lower_limit,...
        coupling_index_list,coupling_upper_limit,coupling_lower_limit);
     data_tmp = [data_tmp;0.0];
     gen_mat = cat(2,gen_mat,data_tmp);
    end
    if mod(size(gen_mat,2),1000) == 0
        fprintf('Progress: %.2f\n',size(gen_mat,2)*100/traj_pivot_points_num)
    end
end

input_mat = gen_mat;
input_mat(7,:) =0.0;



train_data_Collection_file = './data/estimate_parameters_data/MTMR_28002/February-12-2019-20_25_04/dataCollection_info.json';
% Reading config file "gc_controller_config_json"
fid = fopen('gc_controller_config.json');
if fid<3
    error('Cannot find file %s', gc_controller_config_json)
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config = jsondecode(str);
[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(true, [4,1,4,4,4,4]);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[dynamic_vec, ~] = mlse(train_data_Collection_file);

g = 9.81;
input_mat = deg2rad(input_mat);
output_mat = [];
for i = 1:size(input_mat,2)
    q1 = input_mat(1,i);
    q2 = input_mat(2,i);
    q3 = input_mat(3,i);
    q4 = input_mat(4,i);
    q5 = input_mat(5,i);
    q6 = input_mat(6,i);
    torque_pos = analytical_regressor_pos_mat(g, q1, q2, q3, q4, q5, q6)*dynamic_vec;
    torque_neg = analytical_regressor_neg_mat(g, q1, q2, q3, q4, q5, q6)*dynamic_vec;
    output_mat = [output_mat, (torque_pos+torque_neg)/2];
end

save('../data/MLSE4POL_Model/MLSE4POL_configs_torques_10000','input_mat','output_mat');
