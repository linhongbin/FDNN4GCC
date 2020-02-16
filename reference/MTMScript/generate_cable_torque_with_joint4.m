% s = rng;
% joint_pos_upper_limit = [30,45,34,190,175,40];
% joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
% coupling_index_list = {[2,3]};
% coupling_upper_limit = [41];
% coupling_lower_limit = [-11];

%Decoupling data Collection

data_num = 500;
sample_num = 10;
steady_time = 0.5;

joint_init_pose = [0,0,0,0,0,0,0];
Move_Joint_No = 4;
move_deta = 1;
joint_upper_pos = 190;
joint_lower_pos = -80;
repeat_time = 10;


% 
% gen_mat = [];
% bool_gen_mat = [];
% for i=1:data_num
%     alpha = rand(1,6);
%     gen_mat = cat(2,gen_mat,diag(alpha)*joint_pos_upper_limit.'+(diag([1,1,1,1,1,1]) -diag(alpha))*joint_pos_lower_limit.');
%     bool_gen_mat = cat(2,bool_gen_mat, hw_joint_space_check(gen_mat(:,end).',joint_pos_upper_limit,joint_pos_lower_limit,...
%         coupling_index_list,coupling_upper_limit,coupling_lower_limit));
% end
% 
% gen_mat = gen_mat(:, bool_gen_mat == 1);

mtm_arm = mtm('MTMR')
desired_effort = [];
current_position = []

range_size = size(joint_lower_pos:move_deta:joint_upper_pos,2);


joint_current_pos = joint_init_pose;
for i=1:repeat_time
    for k= joint_lower_pos:move_deta:joint_upper_pos
        joint_current_pos(Move_Joint_No) = k;
        mtm_arm.move_joint(deg2rad(joint_current_pos));
        pause(steady_time);
        sample_index = size(desired_effort,2);
        disp(sprintf("%d/%d, repeat No.%d",sample_index, range_size*2*repeat_time, i));
        for j=1:sample_num
            pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
            [~, ~, desired_effort(:,sample_index+1,j)] = mtm_arm.get_state_joint_desired();
            [current_position(:,sample_index+1,j), ~, ~] = mtm_arm.get_state_joint_current();
        end
    end
    for k= joint_upper_pos:-move_deta:joint_lower_pos
        joint_current_pos(Move_Joint_No) = k;
        mtm_arm.move_joint(deg2rad(joint_current_pos));
        pause(steady_time);
        sample_index = size(desired_effort,2);
        disp(sprintf("%d/%d, repeat No.%d",sample_index, range_size*2*repeat_time, i));
        for j=1:sample_num
            pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
            [~, ~, desired_effort(:,sample_index+1,j)] = mtm_arm.get_state_joint_desired();
            [current_position(:,sample_index+1,j), ~, ~] = mtm_arm.get_state_joint_current();
        end
    end
end

% 
% for i=1:size(gen_mat,2)
%     mtm_arm.move_joint(deg2rad([gen_mat(:,i).',0]));
%     pause(steady_time);
%     for j=1:sample_num
%         pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
%         [~, ~, desired_effort(:,i,j)] = mtm_arm.get_state_joint_desired();
%         [current_position(:,i,j), ~, ~] = mtm_arm.get_state_joint_current();
%     end
%     disp(sprintf("%d//%d",i,size(gen_mat,2)));
% end
save('./cable_force/Joint4_cable_force_with_pos_neg_dir.mat',...
    'desired_effort',...
    'current_position',...
    'Move_Joint_No',...
    'repeat_time');