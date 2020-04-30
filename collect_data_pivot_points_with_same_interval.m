% Requirement: open a terminal and type: 
%     $  cd <path-to-matlab-directory>
%     $  rosbag record -o ./data/traj_test/traj_test.bag /dvrk/MTMR/state_joint_current /dvrk/MTMR/state_joint_desired


% Function: move to pivot point and stop for 5 seconds. Afteward move to the subsequent pivot point.
ARM_NAME = 'MTMR'
SN = '31519'
load_file = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data', 'desired_pivot_points.mat')
load(load_file)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % collect in positive direction
% mtm_arm = mtm('MTMR')
% desired_effort = [];
% current_position = [];
% 
% config_mat(7,:) = 0.0;
% sample_num = 10;
% steady_time = 0.3;
% 
% tic 
% for k= 1:size(config_mat,2)
%     mtm_arm.move_joint(deg2rad(config_mat(:,k).'));
%     pause(steady_time);
%     for j=1:sample_num
%         pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
%         [~, ~, desired_effort(:,k,j)] = mtm_arm.get_state_joint_desired();
%         [current_position(:,k,j), ~, ~] = mtm_arm.get_state_joint_current();
%     end
%     duration = toc;
%     fprintf('(%d/%d), predict time: %s seconds left\n', k,size(config_mat,2), datestr(seconds(duration*(size(config_mat,2)-k)/k),'HH:MM:SS'))
% end
% 
% duration = toc;
% duration_time = datestr(seconds(duration),'HH:MM:SS');
% 
% save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data')
% save(fullfile(save_path, 'Real_MTMR_pos_4096'),'current_position')
% save(fullfile(save_path, 'Real_MTMR_tor_4096'),'desired_effort')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% collect in negative, reverse direction
desired_effort_reverse = [];
current_position_reverse = [];

sample_num = 10;
steady_time = 0.3;
config_mat(7,:) = 0.0;
config_mat_reverse = flip(config_mat, 2);
tic 
for k= 1:size(config_mat_reverse,2)
    mtm_arm.move_joint(deg2rad(config_mat_reverse(:,k).'));
    pause(steady_time);
    for j=1:sample_num
        pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
        [~, ~, desired_effort_reverse(:,k,j)] = mtm_arm.get_state_joint_desired();
        [current_position_reverse(:,k,j), ~, ~] = mtm_arm.get_state_joint_current();
    end
    duration = toc;
    fprintf('(%d/%d), predict time: %s seconds left\n', k,size(config_mat_reverse,2), datestr(seconds(duration*(size(config_mat_reverse,2)-k)/k),'HH:MM:SS'))
end

duration = toc;
duration_time = datestr(seconds(duration),'HH:MM:SS');

save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data')
save(fullfile(save_path, 'Real_MTMR_pos_4096_reverse'),'current_position_reverse')
save(fullfile(save_path, 'Real_MTMR_tor_4096_reverse'),'desired_effort_reverse')

