function [current_position, desired_effort] = collect_data(ARM_NAME, path2desired_pivot_points, is_reverse)
    %  Author(s):  Hongbin LIN, Samuel Au
    %  comments: collect data based on the pivot points representing the desired positions for data collection.
    %            is_reverse=true, if you want to collect from the first pivot point to the last
    %            is_reverse=false, if you want to collect from the last pivot point to the first
       
            % ARM_NAME = 'MTMR'
            % SN = '31519'
            % load_file = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data', 'desired_pivot_points.mat')
    load(path2desired_pivot_points)


    mtm_arm = mtm(ARM_NAME)
    desired_effort = [];
    current_position = [];

    config_mat(7,:) = 0.0;
    sample_num = 10;
    steady_time = 0.3;

    tic 
    
    if ~is_reverse
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % collect fromt the first pivot point to the last
    
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


        % 
        % save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data')
        % save(fullfile(save_path, 'Real_MTMR_pos_4096'),'current_position')
        % save(fullfile(save_path, 'Real_MTMR_tor_4096'),'desired_effort')
        
    else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % collect fromt the last pivot point to the first
        desired_effort = [];
        current_position = [];

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
                [~, ~, desired_effort(:,k,j)] = mtm_arm.get_state_joint_desired();
                [current_position(:,k,j), ~, ~] = mtm_arm.get_state_joint_current();
            end
            duration = toc;
            fprintf('(%d/%d), predict time: %s seconds left\n', k,size(config_mat_reverse,2), datestr(seconds(duration*(size(config_mat_reverse,2)-k)/k),'HH:MM:SS'))
        end



            % save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data')
            % save(fullfile(save_path, 'Real_MTMR_pos_4096_reverse'),'current_position_reverse')
            % save(fullfile(save_path, 'Real_MTMR_tor_4096_reverse'),'desired_effort_reverse')
    end
    
    duration = toc;
    duration_time = datestr(seconds(duration),'HH:MM:SS');
    
    fprintf('Elapse time for the data collection is %s\n', duration_time)
end

