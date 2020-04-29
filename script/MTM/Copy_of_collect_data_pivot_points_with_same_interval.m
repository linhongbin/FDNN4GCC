% Requirement: open a terminal and type: 
%     $  cd <path-to-matlab-directory>
%     $  rosbag record -o ./data/traj_test/traj_test.bag /dvrk/MTMR/state_joint_current /dvrk/MTMR/state_joint_desired


% Function: move to pivot point and stop for 5 seconds. Afteward move to the subsequent pivot point.
load('data/FFNN/pivot_points.mat')
steady_time = 5;
mtm_arm = mtm('MTMR')
desired_effort = [];
current_position = [];

config_mat(7,:) = 0.0;
sample_num = 10;
steady_time = 0.3;

tic 
for k= 1:size(config_mat,2)
    mtm_arm.move_joint(deg2rad(config_mat(:,k).'));
    pause(steady_time);
    for j=1:sample_num
        pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
        [~, ~, desired_effort(:,k,j)] = mtm_arm.get_state_joint_desired();
        [current_position(:,k,j), ~, ~] = mtm_arm.get_state_joint_current();
    end
    duration = toc;
    fprintf('(%d/%d), predict time: %s seconds left\n', k,size(config_mat,2), datestr(seconds(duration*(size(config_mat,2)-k)/k),'HH:MM:SS'))
end

duration = toc;
duration_time = datestr(seconds(duration),'HH:MM:SS');

save('data/FFNN/collect_data_same_intervals','current_position','desired_effort','duration','duration_time')