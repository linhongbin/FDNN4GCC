% joint limits
% Read 'dataCollection_config_customized.json' file
function generate_config_pivot_points_random(ARM_NAME, SN, data_num)
% ARM_NAME = 'MTMR'
% SN = '31519'
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

% data_num = 4000;
s = rng;


config_mat = [];
bool_gen_mat = [];
cnt = 0;
while cnt ~= data_num
    alpha = rand(1,6);
    q = diag(alpha)*joint_pos_upper_limit.'+(diag([1,1,1,1,1,1]) -diag(alpha))*joint_pos_lower_limit.';
    if hw_joint_space_check(q.', joint_pos_upper_limit, joint_pos_lower_limit,...
                            coupling_index_list,coupling_upper_limit,coupling_lower_limit)
        cnt = cnt+1;
        config_mat = [config_mat,q];
    end
end

pivot_points_path = fullfile(root_path, 'random',['N', int2str(data_num)], 'raw_data')
if ~exist(pivot_points_path, 'dir')
   mkdir(pivot_points_path)
end
save(fullfile(pivot_points_path, 'desired_pivot_points.mat'));


