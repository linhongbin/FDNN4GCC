test_order_num = 15;
addpath('./gen_code')
train_data_Collection_file = '.\data\estimate_parameters_data\MTMR_28002\February-12-2019-20_25_04\dataCollection_info.json';
% test_data_Collection_file = 'D:\Ben\GC_data_stable\MTMR_28002\February-13-2019-19_32_07\dataCollection_info.json';
test_random_data_Collection_dat = '.\data\optimized_pol_order\test_samples_coup_all_random.mat';
g_constant = 9.81;
train_RMS_arr = [];
test_RMS_arr = [];
test_random_RMS_arr = [];
load(test_random_data_Collection_dat);
test_random_Torques_data = torques_data_process(current_position,...
        desired_effort,...
        'mean',...
         0.3);

for i = 1:test_order_num
    [Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(true,i*[1,1,1,1,1,1]);
    symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
    [output_dynamic_matrix, train_Torques_data] = mlse(train_data_Collection_file);
    train_result = RMS_test(train_Torques_data, 5, output_dynamic_matrix, 'pos', g_constant);
%   test_result = RMS_test(test_Torques_data, 5, output_dynamic_matrix, 'pos', g_constant);
%   test_Torques_data = get_test_data(test_data_Collection_file);
    test_random_result = RMS_test(test_random_Torques_data, 5, output_dynamic_matrix, 'pos', g_constant)
    train_RMS_arr(end+1) = train_result;
%     test_RMS_arr(end+1) = test_result;
    test_random_RMS_arr(end+1) = test_random_result;
  
end

plot_RMS(train_RMS_arr,test_random_RMS_arr);


function RMS = RMS_test(Torques_data, Joint_No,  param_vec,direction, g)
    err_array =[];
    for i = 1:size(Torques_data,3)
        measure_torque = Torques_data(Joint_No,2,i);
        if strcmp(direction,'pos')
        R = analytical_regressor_pos_mat(g,...
                                         Torques_data(1,1,i),...
                                         Torques_data(2,1,i),...
                                         Torques_data(3,1,i),...
                                         Torques_data(4,1,i),...
                                         Torques_data(5,1,i),...
                                         Torques_data(6,1,i));
        end
        if strcmp(direction,'neg')
        R = analytical_regressor_neg_mat(g,...
                                             Torques_data(1,1,i),...
                                             Torques_data(2,1,i),...
                                             Torques_data(3,1,i),...
                                             Torques_data(4,1,i),...
                                             Torques_data(5,1,i),...
                                             Torques_data(6,1,i));
        end
        predict_torque = R*param_vec;
        err_array(end+1) = measure_torque -predict_torque(Joint_No);
    end
    RMS = sqrt(sum(err_array.^2)/size(Torques_data,3));
end


function Torques_data = torques_data_process(current_position, desired_effort, method, std_filter)
    %current_position = current_position(:,:,1:10);
    %desired_effort = desired_effort(:,:,1:10);
    d_size = size(desired_effort);
    Torques_data = zeros(7,2,d_size(2));
    %First Filter out Point out of 1 std, then save the date with its index whose value is close to mean
    for i=1:d_size(2)
        for j=1:d_size(1)
            for k=1:d_size(3)
                effort_data_array(k)=desired_effort(j,i,k);
                position_data_array(k)=current_position(j,i,k);
            end
            effort_data_std = std(effort_data_array);
            effort_data_mean = mean(effort_data_array);
            if effort_data_std<0.0001
                effort_data_std = 0.0001;
            end
            %filter out anomalous data out of 1 standard deviation
            select_index = (effort_data_array <= effort_data_mean+effort_data_std*std_filter)...
                &(effort_data_array >= effort_data_mean-effort_data_std*std_filter);

            effort_data_filtered = effort_data_array(select_index);
            position_data_filtered = position_data_array(select_index);
            if size(effort_data_filtered,2) == 0
                effort_data_filtered =effort_data_array;
                position_data_filtered = position_data_array;
            end
            effort_data_filtered_mean = mean(effort_data_filtered);
            position_data_filtered_mean = mean(position_data_filtered);
            for e = 1:size(effort_data_filtered,2)
                if e==1
                    final_index = 1;
                    min_val =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                else
                    abs_result =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                    if(min_val>abs_result)
                        min_val = abs_result;
                        final_index = e;
                    end
                end
            end
            if(strcmpi(method,'mean'))
                Torques_data(j,1,i) = position_data_filtered_mean;
                Torques_data(j,2,i) = effort_data_filtered_mean;
            elseif(strcmpi(method,'min_abs_error'))
                Torques_data(j,1,i) = current_position(j,i,final_index);
                Torques_data(j,2,i) = desired_effort(j,i,final_index);
            else
                error('Method argument is wrong, please pass: mean or min_abs_error.')
            end
        end
    end

    % Tick out the data collecting from some joint configuration which reaches limits and have cable force effect.
    Torques_data = Torques_data(:,:,3:end-1);
end

function plot_RMS(train_RMS_arr, test_RMS_arr)
    i = 1:size(train_RMS_arr,2)
    
    train_RMS = plot(i,train_RMS_arr,'-o','LineWidth',3,...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor',[.49 1 .63],...
        'MarkerSize',10)
    hold on
    test_RMS = plot(i,test_RMS_arr,'-o','LineWidth',3,...
        'MarkerEdgeColor','r',...
        'MarkerFaceColor',[.49 1 .63],...
        'MarkerSize',10)
    hold off
    set(gca,'FontSize',20)
    xlabel('Order of polynomial function');
    ylabel('E_{RMS}');
    legend('training data','test data','location','northeast')
    ylabel('$\epsilon_{RMS}$ (Nm)','Interpreter','latex','fontweight','bold');
%     save2pdf('./data/pol_rms');
end

