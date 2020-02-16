% test gc controller from WPI


addpath('./gen_code')
train_data_Collection_file = './data/estimate_parameters_data/MTMR_28002/February-12-2019-20_25_04/dataCollection_info.json';
% Reading config file "gc_controller_config_json"
fid = fopen('gc_controller_config.json');
if fid<3
    error('Cannot find file %s', gc_controller_config_json)
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config = jsondecode(str);


[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(true, [1,0,0,1,1,1]);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[WPI_dynamic_vec, ~] = slse(train_data_Collection_file);

% % Spawn GC Controllers and test
mtm_arm = mtm('MTMR');
mtm_gc_controller= controller(mtm_arm,...
    WPI_dynamic_vec,...
    config.GC_controller.safe_upper_torque_limit,...
    config.GC_controller.safe_lower_torque_limit,...
    config.GC_controller.beta_vel_amplitude,...
    9.81,...
    'MTMR');


% Move to gc controller start joint position and wait until MTM remains static
mtm_gc_controller.mtm_arm.move_joint(deg2rad(config.GC_controller.GC_init_pos));
pause(2.5);

% Start gc controller
mtm_gc_controller.start_gc();


% Assign output struct
GC_controllers.controller = mtm_gc_controller;

