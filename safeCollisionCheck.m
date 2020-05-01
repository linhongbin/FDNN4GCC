ARM_NAME = 'MTMR'
SN = '31519'
load_file = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', 'raw_data', 'desired_pivot_points.mat')
load(load_file)
mtm_arm = mtm(ARM_NAME)

config_mat_safeCheck = config_mat_safeCheck(:, [2:2:end]);
for i = 1:size(config_mat_safeCheck, 2)
    pos = [config_mat_safeCheck(:,i).', 0];
    mtm_arm.move_joint(deg2rad(pos));

    disp(sprintf("finish: (%d / %d)", i, size(config_mat_safeCheck, 2)))
end
