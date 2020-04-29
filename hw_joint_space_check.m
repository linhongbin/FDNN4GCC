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
