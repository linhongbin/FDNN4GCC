%% Experiment 
addpath('./gen_code');
file_str = './data/estimate_parameters_data/MTMR_28002/February-12-2019-20_25_04/dataCollection_info.json';
load('data/drift_test/drift_test_pivot_points.mat')

drift_test_pivot_points(7,:) = 0; 
test_pos_mat = drift_test_pivot_points;
% read json files
fid = fopen('gc_controller_config.json');
if fid<3
    error('Cannot find file %s', gc_controller_config_json)
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config = jsondecode(str);
config.ARM_NAME = 'MTMR';

fid = fopen('gc_test_config.json');
if fid<3
    error('Cannot read file "gc_test_config.json"')
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config_tmp = jsondecode(str);
config.GC_Test  = config_tmp.GC_Test;




% Gravity compensation test

start_joint_pos_mat = [];
end_joint_pos_mat = [];


% WPI
[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(false);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[output_dynamic_matrix, ~] = slse(file_str);
config.GC_controller.dynamic_params_vec =output_dynamic_matrix;
[start_joint_pos_mat(:,:,end+1),end_joint_pos_mat(:,:,end+1),SLSE_WPI_pose_mat_cell_list]  = gravity_compensation_test(test_pos_mat,config);


%CAD
[start_joint_pos_mat(:,:,end+1),end_joint_pos_mat(:,:,end+1),CAD_pose_mat_cell_list]  = CAD_gravity_compensation_test(test_pos_mat,config);


% 4POL
POL = 4;
[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(true,POL*[1,1,1,1,1,1]);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[output_dynamic_matrix, ~] = mlse(file_str);
config.GC_controller.dynamic_params_vec =output_dynamic_matrix;
[start_joint_pos_mat(:,:,end+1),end_joint_pos_mat(:,:,end+1),MLSE_4POL_pose_mat_cell_list]  = gravity_compensation_test(test_pos_mat,config);

save('./data/drift_test/drift_test_CAD_SLSE_MLSE4POL','CAD_pose_mat_cell_list','SLSE_WPI_pose_mat_cell_list','MLSE_4POL_pose_mat_cell_list','start_joint_pos_mat','end_joint_pos_mat');

