% Requirement: open a terminal and type: 
%     $  cd <path-to-matlab-directory>
%     $  rosbag record -o ./data/traj_test/traj_test.bag /dvrk/MTMR/state_joint_current /dvrk/MTMR/state_joint_desired


% Function: move to pivot point and stop for 5 seconds. Afteward move to the subsequent pivot point.
load('data/traj_test/traj_pivot_points.mat')
steady_time = 5;
mtm_arm = mtm('MTMR')
desired_effort = [];
current_position = []
for i=1:size(traj_pivot_points,2)
    mtm_arm.move_joint(deg2rad([traj_pivot_points(:,i).']));
    pause(steady_time);
    disp(sprintf('%d/%d',i,size(traj_pivot_points,2)));
end