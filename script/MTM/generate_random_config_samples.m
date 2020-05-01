mtm_arm = mtm('MTMR')
data_num = 4000;
sample_num = 10;
steady_time = 0.5;

s = rng;
joint_pos_upper_limit = [30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];
coupling_index_list = {[2,3]};
coupling_upper_limit = [41];
coupling_lower_limit = [-11];


gen_mat = [];
bool_gen_mat = [];
for i=1:data_num
    alpha = rand(1,6);
    gen_mat = cat(2,gen_mat,diag(alpha)*joint_pos_upper_limit.'+(diag([1,1,1,1,1,1]) -diag(alpha))*joint_pos_lower_limit.');
    bool_gen_mat = cat(2,bool_gen_mat, hw_joint_space_check(gen_mat(:,end).',joint_pos_upper_limit,joint_pos_lower_limit,...
        coupling_index_list,coupling_upper_limit,coupling_lower_limit));
end

gen_mat = gen_mat(:, bool_gen_mat == 1);


desired_effort = [];
current_position = []
for i=1:size(gen_mat,2)
    mtm_arm.move_joint(deg2rad([gen_mat(:,i).',0]));
    pause(steady_time);
    for j=1:sample_num
        pause(0.01); % pause 10ms assuming dVRK console publishes at about 100Hz so we get different samples
        [~, ~, desired_effort(:,i,j)] = mtm_arm.get_state_joint_desired();
        [current_position(:,i,j), ~, ~] = mtm_arm.get_state_joint_current();
    end
    disp(sprintf('%d//%d',i,size(gen_mat,2)));
end
current_date_time =datestr(datetime('now'),'mmmm-dd-yyyy-HH-MM-SS');
file_str = strcat('data/','rand_config_',current_date_time,'.mat');
save(file_str,...
    'desired_effort',...
    'current_position');



function is_in_joint_space =  hw_joint_space_check(q, q_upper_limit, q_lower_limit, varargin)

    % function:
        % check if joint position,q, is within the joint space of hardware, which fullfill:
        %     q_lower_limit <= sum(q) <= q_upper_limit
        %     coupling_lower_limit <= sum(q_coupling) <= coupling_upper_limit  (if any)
    % arguments:
        % q: array of joint position 
        % q_upper_limit: upper limit for joint position
        % q_lower_limit: lower limit for joint position
        % coupling_index_list: cell list for index of coupling joint
        % coupling_upper_limit: arrary of coupling summation upper limit
        % coupling_lower_limit: arrary of coupling summation lower limit
        
    % example:
        % is_in_joint_space = hw_joint_space_check([0,2,3,0,0,0,0], [0,3,4,0,0,0,0],-1*ones(1,7),{[2,3]}, [6], [5])

    % input argument parser
   p = inputParser;
   is_array = @(x) size(x,1) == 1;
   addRequired(p, 'q', is_array);
   addRequired(p, 'q_upper_limit', is_array);
   addRequired(p, 'q_lower_limit', is_array);
   addOptional(p, 'coupling_index_list', {} , @iscell);
   addOptional(p, 'coupling_upper_limit', [] );
   addOptional(p, 'coupling_lower_limit', [] );
   parse(p,q, q_upper_limit, q_lower_limit,varargin{:});
   coupling_index_list =  p.Results.coupling_index_list;
   coupling_upper_limit =  p.Results.coupling_upper_limit;
   coupling_lower_limit =  p.Results.coupling_lower_limit;
   
    % q_lower_limit <= sum(q) <= q_upper_limit
   if all(q_upper_limit>=q) &&  all(q_lower_limit<=q)
       is_in_joint_space = true;
       % check coupling summation limit if exist
       if(~isempty(coupling_index_list))
           for j=1:size(coupling_index_list,2)
               q_coupling = q(coupling_index_list{j});
               % coupling_lower_limit <= sum(q_coupling) <= coupling_upper_limit
               if all(coupling_upper_limit(j)>=sum(q_coupling)) &&  all(coupling_lower_limit(j)<=sum(q_coupling))
                   is_in_joint_space =  true;
               else
                   is_in_joint_space =  false;
                   return;
               end
           end
       end
   else
       is_in_joint_space =  false;
       return;
   end
end
