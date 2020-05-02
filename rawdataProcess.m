function rawdataProcess(root_path, is_dual)

    % ARM_NAME = 'MTMR'
    % SN = '31519'
    sensitive_deg = 1;
    D = 6;
    % process uniform raw data
    % N = 4;
    % root_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N', int2str(N)])
    

    data_path = fullfile(root_path, 'raw_data');
    jnt_pos_file = fullfile(data_path, 'joint_pos.mat');
    jnt_tor_file = fullfile(data_path, 'joint_tor.mat');
    [input_mat_1, output_mat_1] = rawdata2SinCosInput(jnt_pos_file,jnt_tor_file, sensitive_deg, D);
    
    if is_dual
        jnt_pos_file = fullfile(data_path, 'joint_pos_reverse.mat');
        jnt_tor_file = fullfile(data_path, 'joint_tor_reverse.mat');
        [input_mat_2, output_mat_2] = rawdata2SinCosInput(jnt_pos_file,jnt_tor_file, sensitive_deg, D);
        input_mat = [input_mat_1; input_mat_2];
        output_mat = [output_mat_1; output_mat_2];
        save_path = fullfile(root_path, 'D6_SinCosInput', 'dual', 'data');
    else
        input_mat = input_mat_1;
        output_mat = output_mat_1;
        save_path = fullfile(root_path, 'D6_SinCosInput', 'data');
    end
    
    if ~exist(save_path, 'dir')
       mkdir(save_path)
    end
    save(fullfile(save_path, ['D', int2str(D),'_SinCosInput']), 'input_mat', 'output_mat');

%     % process random raw data
% %     N = 200;
%     root_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N', int2str(N)])
%     data_path = fullfile(root_path, 'raw_data')
%     save_path = fullfile(root_path, 'D6_SinCosInput', 'data')
%     jnt_pos_file = fullfile(data_path, 'Real_MTMR_pos.mat');
%     jnt_tor_file = fullfile(data_path, 'Real_MTMR_tor.mat');
%     [input_mat, output_mat] = rawdata2SinCosInput(jnt_pos_file,jnt_tor_file,sensitive_deg);
%     if ~exist(save_path, 'dir')
%        mkdir(save_path)
%     end
%     save(fullfile(save_path, ['N', int2str(N),'_SinCosInput']), 'input_mat', 'output_mat');


end

function [input_mat, output_mat] = rawdata2SinCosInput(jnt_pos_file, jnt_tor_file, sensitive_deg, D)
    load_file = fullfile(jnt_pos_file);
    load(load_file)
    load_file = fullfile(jnt_tor_file);
    load(load_file)
    Torques_data = torques_data_process(current_position, desired_effort, 'mean', 0.3);
    input_mat = [];
    output_mat = [];
    for i=1:size(Torques_data,3)
        input_mat = [input_mat,Torques_data(:,1,i)];
        output_mat = [output_mat,Torques_data(:,2,i)];
    end
    
    % signals for Joint (i>D) are discarded
    input_mat = input_mat(1:D,:).';
    output_mat = output_mat(1:D,:).';
    
    % trigonometric transformation
    last_input_mat = [zeros(1,D);input_mat(1:end-1,:)];
    delta_mat = input_mat - last_input_mat;
    u_mat = zeros(size(delta_mat));
    tmp = arrayfun(@step_func, delta_mat);
    for i = 1:size(delta_mat,1)
        if i == 1
            u_mat(i,:) = tmp(i,:);
        else
            for j = 1:size(delta_mat,2)
                if abs(delta_mat(i,j)) > deg2rad(sensitive_deg)
                    u_mat(i,j) = tmp(i,j);
                else
                    u_mat(i,j) = u_mat(i-1,j);
                end
            end
        end
    end

    input_mat = [sin(input_mat), cos(input_mat), u_mat];
end

function y = step_func(x)
    if x>=0
        y = 1;
    else
        y = 0;
    end
end

function Torques_data = torques_data_process(current_position, desired_effort, method, std_filter)
    %current_position = current_position(:,:,1:10);
    %desired_effort = desired_effort(:,:,1:10);
    d_size = size(desired_effort);
    Torques_data = zeros(7,2,d_size(2));
    %First Filter out Point out of 1 std, then save the date with its index whose value is close to mean
    for i=1:d_size(2)
        for j=1:d_size(1)
            for k=1:d_size(3)
                effort_data_array(k)=desired_effort(j,i,k);
                position_data_array(k)=current_position(j,i,k);
            end
            effort_data_std = std(effort_data_array);
            effort_data_mean = mean(effort_data_array);
            if effort_data_std<0.0001
                effort_data_std = 0.0001;
            end
            %filter out anomalous data out of 1 standard deviation
            select_index = (effort_data_array <= effort_data_mean+effort_data_std*std_filter)...
                &(effort_data_array >= effort_data_mean-effort_data_std*std_filter);

            effort_data_filtered = effort_data_array(select_index);
            position_data_filtered = position_data_array(select_index);
            if size(effort_data_filtered,2) == 0
                effort_data_filtered =effort_data_array;
                position_data_filtered = position_data_array;
            end
            effort_data_filtered_mean = mean(effort_data_filtered);
            position_data_filtered_mean = mean(position_data_filtered);
            for e = 1:size(effort_data_filtered,2)
                if e==1
                    final_index = 1;
                    min_val =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                else
                    abs_result =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                    if(min_val>abs_result)
                        min_val = abs_result;
                        final_index = e;
                    end
                end
            end
            if(strcmpi(method,'mean'))
                Torques_data(j,1,i) = position_data_filtered_mean;
                Torques_data(j,2,i) = effort_data_filtered_mean;
            elseif(strcmpi(method,'min_abs_error'))
                Torques_data(j,1,i) = current_position(j,i,final_index);
                Torques_data(j,2,i) = desired_effort(j,i,final_index);
            else
                error('Method argument is wrong, please pass: mean or min_abs_error.')
            end
        end
    end

end