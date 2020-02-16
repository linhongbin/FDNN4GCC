function [start_joint_pos_mat,end_joint_pos_mat,pose_mat_cell_list] = gravity_compensation_test(test_pose_mat, config)

    mtm_gc_controller = online_test_init(config);
    %mtm_gc_controller.start_gc();
    start_joint_pos_mat = [];
    end_joint_pos_mat = [];
    pose_mat_cell_list = {};
    for i= 1:size(test_pose_mat,2)
        [pos_mat,~] = online_gc_drift_test(mtm_gc_controller,test_pose_mat(:,i).', config);
        disp(sprintf('drift test for MLSE param is finish(%d/%d)', i, size(test_pose_mat,2)));
        start_joint_pos_mat = cat(2,start_joint_pos_mat, pos_mat(:,1));
        end_joint_pos_mat = cat(2,end_joint_pos_mat, pos_mat(:,end));
        pose_mat_cell_list{end+1} = pos_mat;
    end
end

function mtm_gc_controller = online_test_init(config)
    % % Spawn GC Controllers and test
    mtm_arm = mtm(config.ARM_NAME);
    mtm_gc_controller= controller(mtm_arm,...
        config.GC_controller.dynamic_params_vec,...
        config.GC_controller.safe_upper_torque_limit,...
        config.GC_controller.safe_lower_torque_limit,...
        config.GC_controller.beta_vel_amplitude,...
        9.81,...
        config.ARM_NAME);
    
end


function [pos_mat,vel_mat] = online_gc_drift_test(mtm_gc_controller, test_pos,config)
    %disp('Started test estimating drift at a fixed position');
    mtm_gc_controller.mtm_arm.move_joint(deg2rad(test_pos));
    % pause(5.0); % to make sure PID is stable
    %disp('Measuring drift...');
    pause(0.5);
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
        disp('velocity exceed safe limit');
        mtm_gc_controller.stop_gc();
        is_test_success = false;
    end
    
    %online_gc_drift_vel_plot(vel_mat, duration, delta_time, config.GC_Test.ONLINE_GC_DRT.safe_vel_limit);
    
end

function online_gc_drift_vel_plot(vel_mat, duration, delta_time, safe_vel_limit)
    figure
    for i = 1:7
        subplot(7,1,i);
        x = delta_time: delta_time:size(vel_mat, 2)*delta_time;
        x_limit = delta_time: delta_time: duration; 
        plot(x, vel_mat(i,:), x_limit, safe_vel_limit(i)*ones(1,size(x_limit,2)), 'r--', x_limit, -safe_vel_limit(i)*ones(1,size(x_limit,2)), 'r--');
        title(sprintf('Velocity of joint #%d for drift test within %.1f seconds', i, duration));
        if max(vel_mat(i,:),[],2) <=safe_vel_limit(i)
            axis([0 duration -1.5*safe_vel_limit(i) 1.5*safe_vel_limit(i)])
        else
            axis tight
        end
        if i==3
            ylabel('Joint Velocity(rad/s)') 
        elseif i==7
            xlabel('Time(s)') 
        end
    end
end

function gc_dynamic_params = param_vec_checking(input_vec, rows, columns)
    [rows_t, columns_t] = size(input_vec);
    if rows==rows_t && columns == columns_t
        gc_dynamic_params = input_vec;
    elseif rows==columns_t && rows == columns_t
        gc_dynamic_params = input_vec';
    else
        error('size of dynamic vector is not correct. Current size is (%d, %d). Vector size for gc controller should be (%d, %d)',...
                rows_t, columns_t, rows, columns);
    end
end
