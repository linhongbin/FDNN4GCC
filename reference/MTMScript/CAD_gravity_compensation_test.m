function [start_joint_pos_mat,end_joint_pos_mat,pose_mat_cell_list] = CAD_gravity_compensation_test(test_pose_mat, config)

    mtm_gc_controller = online_test_init();
    start_joint_pos_mat = [];
    end_joint_pos_mat = [];
    pose_mat_cell_list = {};
    for i= 1:size(test_pose_mat,2)
        [pos_mat,~] = online_gc_drift_test(mtm_gc_controller,test_pose_mat(:,i).',config);
        disp(sprintf('drift test for CAD param is finish(%d/%d)', i, size(test_pose_mat,2)));
        start_joint_pos_mat = cat(2,start_joint_pos_mat, pos_mat(:,1));
        end_joint_pos_mat = cat(2,end_joint_pos_mat, pos_mat(:,end));
        pose_mat_cell_list{end+1} = pos_mat;
    end
end

function mtm_gc_controller = online_test_init()
    % % Spawn GC Controllers and test
    mtm_gc_controller=  CAD_controller('MTMR',CAD_dynamic_vec());
end
function [pos_mat,vel_mat] = online_gc_drift_test(mtm_gc_controller, test_pos,config)
    %disp('Started test estimating drift at a fixed position');
    mtm_gc_controller.mtm_arm.move_joint(deg2rad(test_pos));
    pause(0.5); % to make sure PID is stable
    %disp('Measuring drift...');
    mtm_gc_controller.start_gc_with_vel_safestop(config.GC_Test.ONLINE_GC_DRT.safe_vel_limit);
    rate = config.GC_Test.ONLINE_GC_DRT.rate;
    delta_time = 1/rate;
    duration = config.GC_Test.ONLINE_GC_DRT.duration;
    pos_mat = [];
    vel_mat = [];
    count = 0;
    while(~mtm_gc_controller.is_drift_vel_exceed_limit &&  count<=duration*rate)
        msg = receive(mtm_gc_controller.sub_pos);
        pos_mat = cat(2, pos_mat, msg.Position);
        vel_mat = cat(2, vel_mat, msg.Velocity);
        pause(delta_time);
        count = count +1;
    end
    if(~mtm_gc_controller.is_drift_vel_exceed_limit)
        mtm_gc_controller.stop_gc();
        is_test_success = true;
    else
        for i=1:20
            msg = receive(mtm_gc_controller.sub_pos);
            pos_mat = cat(2, pos_mat, msg.Position);
            vel_mat = cat(2, vel_mat, msg.Velocity);
            pause(delta_time);
        end
        mtm_gc_controller.stop_gc();
        disp('velocity exceed safe limit');
        is_test_success = false;
    end
    
    %online_gc_drift_vel_plot(vel_mat, duration, delta_time, config.GC_Test.ONLINE_GC_DRT.safe_vel_limit);
    
end
