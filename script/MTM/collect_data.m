[input_mat,ready_input_mat] = get_input_mat('neg');
input_mat = deg2rad(input_mat);
ready_input_mat = deg2rad(ready_input_mat);

mtm_arm = mtm('MTMR')
desired_effort = [];
current_position = [];

ready_input_mat(7,:) = 0.0;
input_mat(7,:) = 0.0;
sample_num = 10;
steady_time = 0.3;

tic 
for k= 1:size(input_mat,2)
    mtm_arm.move_joint(ready_input_mat(:,k));
    pause(0.1);
    mtm_arm.move_joint(input_mat(:,k));
    pause(steady_time);
    for j=1:sample_num
        pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
        [~, ~, desired_effort(:,k,j)] = mtm_arm.get_state_joint_desired();
        [current_position(:,k,j), ~, ~] = mtm_arm.get_state_joint_current();
    end
    duration = toc;
    %fprintf('(%d/%d), predict time: %s seconds left\n', k,size(config_mat,2), datestr(seconds(duration*(size(config_mat,2)-k)/k),'HH:MM:SS'))
end

duration = toc;
duration_time = datestr(seconds(duration),'HH:MM:SS');




function [input_mat,ready_input_mat] = get_input_mat(dir)
hyst_act_angle= 3;
joint_pos_upper_limit =[30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
upper_limit = joint_pos_upper_limit;
lower_limit = joint_pos_lower_limit;
margin_arr = 5*[1,1,1,1,1,1];
coup_margin = 7;
coupling_upper_limit = 41;
coupling_lower_limit = -11;
% coupling_index_list = {[2,3]};
% coupling_upper_limit = [41];
% coupling_lower_limit = [-11];

jnt_sample_arr = [1,5,5,5,5,5];

upper_limit = upper_limit - margin_arr;
lower_limit = lower_limit + margin_arr;
coup_upper_limit = coupling_upper_limit-coup_margin;
coup_lower_limit = coupling_lower_limit+coup_margin;

X={};
for i=1:6
    if i~=3
        X{i} = linspace(lower_limit(i), upper_limit(i), jnt_sample_arr(i));
    end
end

    
input_mat = [];
output_mat = [];
count =0;
for idx1 = 1:jnt_sample_arr(1)
    for idx2 = 1:jnt_sample_arr(2)
        for idx3 = 1:jnt_sample_arr(3)
            jnt_min = max(lower_limit(3), coup_lower_limit-X{2}(idx2));
            jnt_max = min(upper_limit(3), coup_upper_limit-X{2}(idx2));
            X{3} = linspace(jnt_min, jnt_max, jnt_sample_arr(3));
            for idx4 = 1:jnt_sample_arr(4)
                for idx5 = 1:jnt_sample_arr(5)
                    for idx6 = 1:jnt_sample_arr(6)
                        input = [0;X{2}(idx2);X{3}(idx3);X{4}(idx4);X{5}(idx5);X{6}(idx6)];
                        input_mat = [input_mat, input];
                    end
                end
            end
        end
    end
end
    mistakes_count = 0;
    for i = 1:size(input_mat,2)
        if ~hw_joint_space_check(input_mat(:,i).',joint_pos_upper_limit,joint_pos_lower_limit,...
                {[2,3]},[coupling_upper_limit],[coupling_lower_limit])
            mistakes_count = mistakes_count+1;
        end
    end
    fprintf('mistake count of input_mat is %d\n',mistakes_count);
    
    if strcmp(dir,'neg')
        input_mat = flip(input_mat,2);
    end
    
    % get ready state
    ready_input_mat = [];
    for i = 1:size(input_mat,2)
        ready_input = input_mat(:,i).';
        for k = 1:6
            if strcmp(dir,'pos')
                if i ==1
                    ready_input(k) = ready_input(k) - hyst_act_angle;
                elseif input_mat(k,i)-input_mat(k,i-1)<0.0
                    ready_input(k) = ready_input(k) - hyst_act_angle;
                end
            elseif strcmp(dir,'neg')
                if i ==1
                    ready_input(k) = ready_input(k) + hyst_act_angle;
                elseif input_mat(k,i)-input_mat(k,i-1)>0.0
                    ready_input(k) = ready_input(k) + hyst_act_angle;
                end
            else
                error('dir error');
            end
        end
        ready_input_mat = [ready_input_mat,ready_input.'];
    end
    
    mistakes_count = 0;
    for i = 1:size(ready_input_mat,2)
        if ~hw_joint_space_check(ready_input_mat(:,i).',joint_pos_upper_limit,joint_pos_lower_limit,...
                {[2,3]},[coupling_upper_limit],[coupling_lower_limit])
            mistakes_count = mistakes_count+1;
        end
    end
    fprintf('mistake count of input_mat is %d\n',mistakes_count);
end

