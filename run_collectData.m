%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modify the following setting for you dVRK
ARM_NAME = 'MTML'
SN = '41878'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% wizard program to identify the custom joint limits for specific MTM
dataCollection_config_customized_str = wizard4JntLimit(ARM_NAME, SN);

% generate pivot points for training data
N_train = 2; % param for systematic sampling, 4 points for each joint.
[config_mat, config_mat_safeCheck] = generate_config_pivot_points_with_same_interval(dataCollection_config_customized_str, N_train);
pivot_points_path_train = fullfile('data', [ARM_NAME,'_',SN], 'real', 'uniform', ['N', int2str(N_train)],'raw_data');
if ~exist(pivot_points_path_train, 'dir')
   mkdir(pivot_points_path_train);
end
save(fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), 'config_mat' ,'N');

% generate pivot points for validating data
N_validate = 3; % param for random sampling, 180 randomly sampled points
config_mat = generate_config_pivot_points_random(dataCollection_config_customized_str, N_validate);
pivot_points_path_validate = fullfile('data', [ARM_NAME,'_',SN], 'real', 'random', ['N', int2str(N_validate)] ,'raw_data');
if ~exist(pivot_points_path_validate, 'dir')
   mkdir(pivot_points_path_validate);
end
save(fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), 'config_mat' ,'N');

% generate pivot points for testing data
N_test = 4; % param for random sampling, 20 randomly sampled points
config_mat = generate_config_pivot_points_random(dataCollection_config_customized_str, N_test);
pivot_points_path_test = fullfile('data', [ARM_NAME,'_',SN], 'real', 'random', ['N', int2str(N_test)] ,'raw_data');
if ~exist(pivot_points_path_test, 'dir')
   mkdir(pivot_points_path_test);
end
save(fullfile(pivot_points_path_test, 'desired_pivot_points.mat'), 'config_mat' ,'N');

% collision checking for safety during data collection. 
% It ensure MTM in long data collection process not to hit environment once it passes.
% safeCollisionCheck(config_mat_safeCheck, ARM_NAME);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%collect training data, about 4 hour.

% in non-reverse order 
is_reverse = false;
[current_position, desired_effort] = collect_data(ARM_NAME,...
                            fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), is_reverse); % 2 hour
save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)],'raw_data');
save(fullfile(save_path, 'Real_MTMR_pos'),'current_position');
save(fullfile(save_path, 'Real_MTMR_tor'),'desired_effort');

% in reverse order
is_reverse = true;
[current_position, desired_effort] = collect_data(ARM_NAME,...
                            fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), is_reverse); % 2 hour
save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)],'raw_data');
save(fullfile(save_path, 'Real_MTMR_pos_reverse'),'current_position');
save(fullfile(save_path, 'Real_MTMR_tor_reverse'),'desired_effort');

% collect validating data
is_reverse = false;
[current_position, desired_effort] = collect_data(ARM_NAME,...
                            fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), is_reverse); 
save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_validate)],'raw_data');
save(fullfile(save_path, 'Real_MTMR_pos'),'current_position');
save(fullfile(save_path, 'Real_MTMR_tor'),'desired_effort');


% collect testing data
is_reverse = false;
[current_position, desired_effort] = collect_data(ARM_NAME,...
                            fullfile(pivot_points_path_test, 'desired_pivot_points.mat'), is_reverse); 
save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_test)],'raw_data');
save(fullfile(save_path, 'Real_MTMR_pos'),'current_position');
save(fullfile(save_path, 'Real_MTMR_tor'),'desired_effort');



% data processsing
root_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)]);
is_dual = true;
rawdataProcess(root_path, is_dual);

root_path =  fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_validate)]);
is_dual = false;
rawdataProcess(root_path, is_dual);

root_path =  fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_test)]);
is_dual = false;
rawdataProcess(root_path, is_dual);










