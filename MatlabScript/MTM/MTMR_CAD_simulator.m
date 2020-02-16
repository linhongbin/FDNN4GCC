% addpath('./model/CAD')

% joint limits
upper_limit = deg2rad([30,45,34,190,175,40]);
lower_limit = deg2rad([-30,-14,-34,-80,-85,-40]);
% coupling_index_list = {[2,3]};
% coupling_upper_limit = [41];
% coupling_lower_limit = [-11];

sample_num_each_joint = 4;
X1 = linspace(lower_limit(1), upper_limit(1), sample_num_each_joint);
X2 = linspace(lower_limit(2), upper_limit(2), sample_num_each_joint);
X3 = linspace(lower_limit(3), upper_limit(3), sample_num_each_joint);
X4 = linspace(lower_limit(4), upper_limit(4), sample_num_each_joint);
X5 = linspace(lower_limit(5), upper_limit(5), sample_num_each_joint);
X6 = linspace(lower_limit(6), upper_limit(6), sample_num_each_joint);
    
input_mat = [];
output_mat = [];
count =0;
for idx1 = 1:size(X1,2)
    for idx2 = 1:size(X2,2)
        for idx3 = 1:size(X3,2)
            for idx4 = 1:size(X4,2)
                for idx5 = 1:size(X5,2)
                    for idx6 = 1:size(X6,2)
                        input = [X1(idx1);X2(idx2);X3(idx3);X4(idx4);X5(idx5);X6(idx6)];
                        output = MTMR28002Gravity(input(1),input(2),input(3),input(4),input(5),input(6));
                        input_mat = [input_mat; input.'];
                        output_mat = [output_mat; output.'];
                    end
                end
            end
            count = count +1;
            file_name = sprintf('MTMR_CAD_sim_%d.mat', count);
            %save(fullfile('data' ,file_name), 'input_mat', 'output_mat')
            input_mat = [];
            output_mat = [];
        end
        fprintf('Progress = %d %%\n',int32(double(idx2+(idx1-1)*size(X2,2))*100/double((size(X1,2)* size(X2,2)))))
    end
end
