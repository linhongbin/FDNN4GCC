% joint limits
joint_pos_upper_limit = [30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
coupling_index_list = {[2,3]};
coupling_upper_limit = [41];
coupling_lower_limit = [-11];

traj_pivot_points_num = 10;

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
end

traj_pivot_points = gen_mat;

save('data/traj_test/traj_pivot_points', 'traj_pivot_points');