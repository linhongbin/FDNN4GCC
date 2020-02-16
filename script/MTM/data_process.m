Torques_data = torques_data_process(current_position, desired_effort, 'mean', 0.3);
input_mat = [];
output_mat = [];
for i=1:size(Torques_data,3)
    input_mat = [input_mat,Torques_data(:,1,i)];
    output_mat = [output_mat,Torques_data(:,2,i)];
end
input_mat = input_mat(2:6,:).';
output_mat = output_mat(2:6,:).';