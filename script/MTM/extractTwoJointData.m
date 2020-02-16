data_path = 'D:\Ben\dVRK\MTMR_28002\MTMR_28002\June-12-2019-15-27-16\dataCollection_info.json'
[Torques_data_pos, Torques_data_neg] = mlse(data_path);
input_mat_pos = [];
output_mat_pos = [];
input_mat_neg = [];
output_mat_neg = [];
for i = 1:size(Torques_data_pos,3)
     input_mat_pos(:,i) = Torques_data_pos(:,1,i);
     output_mat_pos(:,i) = Torques_data_pos(:,2,i);
end
for i = 1:size(Torques_data_neg,3)
     input_mat_neg(:,i) = Torques_data_neg(:,1,i);
     output_mat_neg(:,i) = Torques_data_neg(:,2,i);
end

input_mat_pos = input_mat_pos(2:6,:).';
output_mat_pos = output_mat_pos(2:6,:).';
input_mat_neg = input_mat_neg(2:6,:).';
output_mat_neg = output_mat_neg(2:6,:).';

function [Torques_data_pos, Torques_data_neg] = mlse(dataCollection_info_str)
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    argument_checking(dataCollection_info_str)

    % General Setting
    output_file_str = '';

    % Read JSON config input file dataCollection_info_str
    fid = fopen(dataCollection_info_str);
    if fid<3
        error('Cannot read file %s', dataCollection_info_str)
    end
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    config = jsondecode(str);

    % Get the path
    [input_data_path_with_date, ~, ~] = fileparts(dataCollection_info_str);

    % display data input root path
    disp(' ');
    fprintf('data path for MLSE : ''%s'' \n', input_data_path_with_date);
    disp(' ');


    % create config_LSE objects
    config_lse_list=setting_lse(config,input_data_path_with_date);

    Torques_data_pos = [];
    Torques_data_neg = [];
    % Multi-steps MLSE from Joint#6 to Joint#1.
    for i=6:-1:1
        if i==6
            [Torques_pos_data, Torques_neg_data] = lse_mtm_one_joint(config_lse_list{i});
        elseif i==1
            [Torques_pos_data, Torques_neg_data] = lse_mtm_one_joint(config_lse_list{i},config_lse_list{i+1});
        else
            [Torques_pos_data, Torques_neg_data] = lse_mtm_one_joint(config_lse_list{i},config_lse_list{i+1});
        end
        Torques_data_pos = cat(3, Torques_data_pos, Torques_pos_data);
        Torques_data_neg = cat(3, Torques_data_neg, Torques_neg_data);
    end

end

function argument_checking(input_data_path_with_date)
    if ischar(input_data_path_with_date) ==0
        error('%s is not a char object', input_data_path_with_date)
    end
end



function [Torques_pos_data, Torques_neg_data] = lse_mtm_one_joint(config_lse_joint, previous_config)
    %  Institute: The Chinese University of Hong Kong
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    
    fprintf('LSE for joint %d started..\n', config_lse_joint.Joint_No);

    if ~exist('previous_config')
         [Torques_pos_data, Torques_neg_data] = lse_model(config_lse_joint);
    else
        % if there is 'previous_config', we pass the path to the result of previous step of LSE to lse_model
        [Torques_pos_data, Torques_neg_data] = lse_model(config_lse_joint,...
            previous_config.output_param_path);
    end
end

function  [Torques_pos_data, Torques_neg_data] = lse_model(config_lse_joint1,...
                                            old_param_path)

    %  Institute: The Chinese University of Hong Kong
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05


    lse_obj = lse_preparation(config_lse_joint1);
    Torques_pos_data = lse_obj.Torques_pos_data;
    Torques_neg_data = lse_obj.Torques_neg_data;
end


function lse_obj = lse_preparation(config_lse_joint)

    % create lse_obj inheriting config_lse_joint
    lse_obj.Joint_No = config_lse_joint.Joint_No;
    lse_obj.std_filter = config_lse_joint.std_filter;
    lse_obj.g_constant = config_lse_joint.g_constant;
    lse_obj.Is_Plot = config_lse_joint.Is_Plot;
    lse_obj.issave_figure = config_lse_joint.issave_figure;
    lse_obj.Input_Pos_Data_Path = config_lse_joint.input_pos_data_path;
    lse_obj.Input_Neg_Data_Path = config_lse_joint.input_neg_data_path;
    lse_obj.input_pos_data_files = config_lse_joint.input_pos_data_files;
    lse_obj.input_neg_data_files = config_lse_joint.input_neg_data_files;
    lse_obj.new_param_save_path = config_lse_joint.output_param_path;
    lse_obj.new_fig_pos_save_path = config_lse_joint.output_pos_fig_path;
    lse_obj.new_fig_neg_save_path = config_lse_joint.output_neg_fig_path;
    lse_obj.prior_param_index = config_lse_joint.prior_param_index;
    lse_obj.prior_param_values = config_lse_joint.prior_param_values;
    lse_obj.Output_Param_Joint_No = config_lse_joint.Output_Param_Joint_No;
    lse_obj.std_filter = config_lse_joint.std_filter;
    lse_obj.fit_method = config_lse_joint.fit_method;

    % check the given joint path exist
    if exist(lse_obj.Input_Pos_Data_Path)==0
        error('Cannot find input data folder: %s. Please check that input data folder exists.', lse_obj.Input_Pos_Data_Path);
    end
    if exist(lse_obj.Input_Neg_Data_Path)==0
        error('Cannot find input data folder: %s. Please check that input data folder exists.', lse_obj.Input_Neg_Data_Path);
    end

    data_path_struct_list = dir(lse_obj.input_pos_data_files);
    lse_obj.Torques_pos_data_list = {};
    lse_obj.theta_pos_list = {};
    for i=1:size(data_path_struct_list,1)
        load(strcat(data_path_struct_list(i).folder,'/',data_path_struct_list(i).name));
        lse_obj.Torques_pos_data_list{end+1} = torques_data_process(current_position,...
            desired_effort,...
            'mean',...
            lse_obj.std_filter);
        lse_obj.theta_pos_list{end+1} = int32(Theta);
    end

    data_path_struct_list = dir(lse_obj.input_neg_data_files);
    lse_obj.Torques_neg_data_list = {};
    lse_obj.theta_neg_list = {};
    for i=1:size(data_path_struct_list,1)
        load(strcat(data_path_struct_list(i).folder,'/',data_path_struct_list(i).name));
        lse_obj.Torques_neg_data_list{end+1} = torques_data_process(current_position,...
            desired_effort,...
            'mean',...
            lse_obj.std_filter);
        lse_obj.theta_neg_list{end+1} = int32(Theta);
    end

    % Append List Torques Data
    lse_obj.Torques_pos_data = [];
    for j = 1:size(lse_obj.Torques_pos_data_list,2)
        lse_obj.Torques_pos_data = cat(3,lse_obj.Torques_pos_data,lse_obj.Torques_pos_data_list{j});
    end
    lse_obj.Torques_neg_data = [];
    for j = 1:size(lse_obj.Torques_neg_data_list,2)
        lse_obj.Torques_neg_data = cat(3,lse_obj.Torques_neg_data,lse_obj.Torques_neg_data_list{j});
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

    % Tick out the data collecting from some joint configuration which reaches limits and have cable force effect.
    Torques_data = Torques_data(:,:,3:end-1);
end

function config_lse_list=setting_lse(config, data_input_root_path)
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    %General Setting
    Is_Plot = false;
    issave_figure = false;
    std_filter = 0.3;
    g_constant = 9.8;
    fit_method = '4POL';
    %Generate config obj for lse
    Joint_No = 6;
    config_lse_joint6 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint6'],fit_method,g_constant);

    Joint_No = 5;
    config_lse_joint5 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint5'],fit_method,g_constant);

    Joint_No = 4;
    config_lse_joint4 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint4'],fit_method,g_constant);

    Joint_No = 3;
    config_lse_joint3 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint3'],fit_method,g_constant);

    Joint_No = 2;
    config_lse_joint2 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint2'],fit_method,g_constant);

    Joint_No = 1;
    config_lse_joint1 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint1'],fit_method,g_constant);
    
    config_lse_list = {config_lse_joint1,config_lse_joint2,config_lse_joint3,config_lse_joint4,config_lse_joint5,config_lse_joint6};

end

