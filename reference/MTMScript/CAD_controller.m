classdef CAD_controller < handle
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    properties(Access = public)
        pub_tor
        sub_pos
        ARM_NAME
        mtm_arm
        drift_test_safe_vel_limit
        is_drift_vel_exceed_limit
        dynamic_vec
        msg_counter_buffer
        is_disp_init_info
    end

    methods(Access = public)
        % Class constructor
        function obj = CAD_controller(ARM_NAME, dynamic_vec)
            obj.mtm_arm = mtm(ARM_NAME);
            obj.pub_tor = rospublisher(['/dvrk/',ARM_NAME,'/set_effort_joint']);
            obj.sub_pos = rossubscriber(['/dvrk/',ARM_NAME,'/state_joint_current']);
            obj.ARM_NAME = ARM_NAME;
            obj.dynamic_vec = dynamic_vec;
            obj.is_disp_init_info = true;
            obj.msg_counter_buffer= 0;
        end

        % Callback function of pose subscriber when start gc controller
        function callback_gc_publisher(obj, q, q_dot)
            if(~obj.is_disp_init_info)
                fprintf('GC of %s starts, you can move %s now. If you need to stop gc controller, call "mtm_gc_controller.stop_gc()".\n',obj.ARM_NAME,obj.ARM_NAME);
                obj.is_disp_init_info = true;
            end

            if(obj.msg_counter_buffer==0)
                fprintf('.');
            end

            if(obj.msg_counter_buffer == 100)
                obj.msg_counter_buffer = 0;
            else
                obj.msg_counter_buffer = obj.msg_counter_buffer+1;
            end
            % Calculate predict torques
            Torques = obj.base_controller(q);

            % Publish predict torques
            msg = rosmessage(obj.pub_tor);
            for i =1:7
                msg.Effort(i) = Torques(i);
            end
            send(obj.pub_tor, msg);
        end
        
        % Callback function of pose subscriber when start gc controller
        function callback_gc_pub_with_vel_safestop(obj, q, q_dot)
            for i=1:7
                if(abs(q_dot(i))>obj.drift_test_safe_vel_limit(i))
                    obj.is_drift_vel_exceed_limit = true;
                end
            end
            
            if obj.is_drift_vel_exceed_limit 
                msg = rosmessage(obj.pub_tor);
                for i =1:7
                    msg.Effort(i) = 0.0;
                end
            else
                % Calculate predict torques
                Torques = obj.base_controller(q);
                % Publish predict torques
                msg = rosmessage(obj.pub_tor);
                for i =1:7
                    msg.Effort(i) = Torques(i);
                end
                % for testing
%                 msg.Effort(5) = 0.05;
            end
            send(obj.pub_tor, msg);
        end

        % Base controller to calculate the predict torque
        function Torques = base_controller(obj, q)
            g = 9.81;
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4);
            q5 = q(5);
            q6 = q(6);
            Torques =  CAD_analytical_regressor(g,q1,q2,q3,q4,q5,q6) *  obj.dynamic_vec;
            Torques(5) = 0;
        end

        % call this function to start the gc controller
        function start_gc(obj)
            % Apply GC controllers
            callback_MTM = @(src,msg)(obj.callback_gc_publisher(msg.Position));
            obj.sub_pos = rossubscriber(['/dvrk/',obj.ARM_NAME,'/state_joint_current'],callback_MTM,'BufferSize',10);
        end

        % call this function to stop the gc controller and move to origin pose
        function stop_gc(obj)
            obj.sub_pos = rossubscriber(['/dvrk/',obj.ARM_NAME,'/state_joint_current']);
            obj.mtm_arm.move_joint([0,0,0,0,0,0,0]);
            disp('gc_controller stopped');
        end
        
        % call this function to start the gc controller
        function start_gc_with_vel_safestop(obj, safe_vel_limit)
            % Apply GC controllers
            obj.drift_test_safe_vel_limit = safe_vel_limit;
            obj.is_drift_vel_exceed_limit = false;
            callback_MTM = @(src,msg)(obj.callback_gc_pub_with_vel_safestop(msg.Position,...
                                                                            msg.Velocity));
            obj.sub_pos = rossubscriber(['/dvrk/',obj.ARM_NAME,'/state_joint_current'],callback_MTM,'BufferSize',10);
        end
        
        function set_zero_tor_output_joint(obj, Zero_Output_Joint_No)
            obj.Zero_Output_Joint_No = Zero_Output_Joint_No
        end
    end
end

