% joint limits
% Read 'dataCollection_config_customized.json' file
ARM_NAME = 'MTMR'
SN = '31519'
root_path = fullfile('data', [ARM_NAME, '_',SN], 'real')

fid = fopen(fullfile(root_path, 'dataCollection_config_customized.json'));
if fid<3
    error('cannot open file dataCollection_config_customized.json, please check the path')
end
raw = fread(fid, inf);
str = char(raw');
config = jsondecode(str);
fclose(fid);

joint_pos_upper_limit = config.joint_pos_upper_limit.'; 
joint_pos_lower_limit = config.joint_pos_lower_limit.'; 
coupling_index_list = {config.coupling_index_list.'};
coupling_upper_limit = [config.coupling_upper_limit]; 
coupling_lower_limit = [config.coupling_lower_limit];

% pivot points of each joint
N = 4;
joint_pivot_num_list = N * [1,1,1,1,1,1];

% margin to the joint limit
joint_lim_margin_list = [2, 2, 2, 2, 2, 2];

pivot_mat = [];
joint3_coup_pivot_mat = [];
for i = 1:size(joint_pos_upper_limit,2)
    % treat joint 3 as special case dealing with coupling limits
    if(i==3)
        pivot_mat = [pivot_mat,zeros(joint_pivot_num_list(3),1)];
    else
        pivot_mat = [pivot_mat,...
                              linspace(joint_pos_lower_limit(i)+joint_lim_margin_list(i),...
                              joint_pos_upper_limit(i)-joint_lim_margin_list(i),...
                              joint_pivot_num_list(i))'];
    end
end
for i = 1:size(pivot_mat(:,2),1)
    joint3_coup_pivot_mat = [joint3_coup_pivot_mat,...
                      linspace(max(joint_pos_lower_limit(3)+joint_lim_margin_list(3),...
                          coupling_lower_limit(1)-pivot_mat(i,2)+joint_lim_margin_list(3)),...
                      min(joint_pos_upper_limit(3)-joint_lim_margin_list(3),...
                         coupling_upper_limit(1)-pivot_mat(i,2)-joint_lim_margin_list(3)),...
                      joint_pivot_num_list(3))'];

end

config_mat = joint_all_combs(pivot_mat, joint3_coup_pivot_mat);




mistakes_count = 0
for i = 1:size(config_mat,2)
    if ~hw_joint_space_check(config_mat(:,i).',joint_pos_upper_limit,joint_pos_lower_limit,...
            coupling_index_list,coupling_upper_limit,coupling_lower_limit)
        mistakes_count = mistakes_count+1;
    end
end


pivot_mat_safeCheck = pivot_mat([1,end],:) 
joint3_coup_pivot_mat_safeCheck = joint3_coup_pivot_mat([1,end], [1,end]);
config_mat_safeCheck = joint_all_combs(pivot_mat_safeCheck, joint3_coup_pivot_mat_safeCheck);


pivot_points_path = fullfile(root_path, 'uniform', 'raw_data', ['N', int2str(N)])
if ~exist(pivot_points_path, 'dir')
   mkdir(pivot_points_path)
end
save(fullfile(pivot_points_path, 'desired_pivot_points.mat'));

function config_mat = joint_all_combs(pivot_mat, joint3_coup_pivot_mat)
    config_mat = [];
    joint_pivot_num = size(pivot_mat,1);
    index_list = [1,1,1,1,1,1];
    while ~isequal(index_list,[1,1,1,1,1,1]*joint_pivot_num)
        vec =[];
        for i = 1:6
            if i==3
                vec = [vec;joint3_coup_pivot_mat(index_list(3),index_list(2))];
            else
                vec = [vec;pivot_mat(index_list(i),i)];
            end
        end
        config_mat = [config_mat, vec];
        index_list = count_index(index_list, joint_pivot_num);
    end
    
    % do for the case :[1,1,1,1,1,1]*joint_pivot_num
    vec =[];
    for i = 1:6
        vec = [vec;pivot_mat(index_list(i),i)];
    end
    config_mat = [config_mat, vec];
end

function index_list_output = count_index(index_list_input, overflow_num)
    i = 6;
    index_list_input(i) = index_list_input(i)+1;
    while(index_list_input(i)>overflow_num) 
        index_list_input(i) = 1;
        i = i-1;
        index_list_input(i) = index_list_input(i)+1;
    end
    index_list_output = index_list_input;
end