addpath('./gen_code')


% joint limits
joint_pos_upper_limit = [30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
coupling_index_list = {[2,3]};
coupling_upper_limit = [41];
coupling_lower_limit = [-11];

traj_pivot_points_num = 100000;

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
    output_mat = [output_mat, CAD_analytical_regressor(9.81, q1, q2, q3, q4, q5, q6)*CAD_dynamic_vec];
end

save('../data/CAD_Model/CAD_pos_10000','input_mat');
save('../data/CAD_Model/CAD_tor_10000','output_mat');
