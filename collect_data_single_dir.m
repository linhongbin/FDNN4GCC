function [current_position, desired_effort] = collect_data_single_dir(ARM_NAME, path2desired_pivot_points)
    % ARM_NAME = 'MTMR'
    % SN = '31519'
    % N = 200
    % root_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N', int2str(N)],'raw_data')
    % load_file = fullfile(root_path, 'desired_pivot_points.mat')
    load(path2desired_pivot_points)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % collect in positive direction
    mtm_arm = mtm(ARM_NAME)
    desired_effort = [];
    current_position = [];

    config_mat(7,:) = 0.0; % 0 Position for all testing points of Joint 7
    sample_num = 10; % collect 10 times
    steady_time = 0.3; % stay for 0.3 sec before collecting data

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

%     save(fullfile(root_path, 'Real_MTMR_pos'),'current_position')
%     save(fullfile(root_path, 'Real_MTMR_tor'),'desired_effort')


end

