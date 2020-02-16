Coupling_train_data_Collection_file =  './data/estimate_parameters_data/MTMR_28002/Couple_test/dataCollection_info.json';

Decoupling_train_data_Collection_file1 =  './data/estimate_parameters_data/MTMR_28002/Decouple_test1/dataCollection_info.json';
Decoupling_train_data_Collection_file2 =  './data/estimate_parameters_data/MTMR_28002/Decouple_test2/dataCollection_info.json';
Decoupling_train_data_Collection_file3 =  './data/estimate_parameters_data/MTMR_28002/Decouple_test3/dataCollection_info.json';

test_random_data_Collection_dat = "./data/data_collection_strategy_rms/test_samples_coup_all_random.mat";
addpath('./gen_code')
g_constant = 9.81;
train_RMS_arr = [];
test_RMS_arr = [];
test_random_RMS_arr = [];
load(test_random_data_Collection_dat);
test_Torques_data = torques_data_process(current_position,...
        desired_effort,...
        'mean',...
         0.3);

%MLSE
pol_order = 4;
% [Regressor_Matrix_Pos,Regressor_Matrix_Neg, Parameter_matrix, bool_Regressor_Mat] = symbolic_gc_dynamic(pol_order);
% symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
%      
MLSE_dynamic_param_mat = [];
Condition_num_mat = [];
[MLSE_dynamic_param_mat(:,1), ~, Condition_num_mat(1,:)] = mlse(Coupling_train_data_Collection_file);
[MLSE_dynamic_param_mat(:,2), ~, Condition_num_mat(2,:)] = mlse(Decoupling_train_data_Collection_file1);
[MLSE_dynamic_param_mat(:,3), ~, Condition_num_mat(3,:)] = mlse(Decoupling_train_data_Collection_file2);
[MLSE_dynamic_param_mat(:,4), ~, Condition_num_mat(4,:)] = mlse(Decoupling_train_data_Collection_file3);


figure(1)
bar(Condition_num_mat(:,3:6).');
font_size = 24;
box on
grid on
set(gca, 'YScale', 'log')
set(gca,'FontSize',font_size)
ylabel('Condition Number'); 
ylim([0 exp(14)]);
lgd = legend('Two-joint','One-joint 1','One-joint 2','One-joint 3','location','north');
set(lgd,'FontName','Times New Roman');
set(gca,'XTickLabel',{'Joint 3','Joint 4','Joint 5','Joint 6'});
% save2pdf('./data/dataCollection_methods_cond_num');
%xlabel('{\it q_4} (?)','Interpreter','tex')
% ylabel(['$\tau_','0'+4,'$ (Nm)'],'Interpreter','latex','fontweight','bold'); 

RMS_mat = [];
for i = 1:4
    RMS_mat(:,i) = compute_RMS('MLSE' ,test_Torques_data, MLSE_dynamic_param_mat(:,i), g_constant);
end

% 
% % bar_dat = [CAD_RMS_result,MLSE_1POL_RMS_result,SLSE_RMS_result,MLSE_RMS_result];
% bar_dat = [CAD_RMS_result,MLSE_1POL_RMS_result,MLSE_RMS_result];
figure(2)
box on
grid on
bar(RMS_mat(3:6,:));
set(gca, 'YScale', 'log')
% ylim([0 3])
set(gca,'XTickLabel',{'Joint 3','Joint 4','Joint 5','Joint 6'});
set(gca,'FontSize',font_size);
ylabel('$\epsilon_{RMS}$(Nm)','Interpreter','latex','fontweight','bold');
legend('Two-joint','One-joint 1','One-joint 2','One-joint 3','location','best');
set(lgd,'FontName','Times New Roman');
% save2pdf('./data/dataCollection_methods_RMS');
% set(gca,'FontSize',20)
% xlabel('Joint No(#)');
% ylabel('E_{RMS}');
% % legend('CAD','MLSE-1POL','SLSE-4POL','MLSE-4POL')
% legend('CAD','1POL','4POL')


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
    %Torques_data = Torques_data(:,:,3:end-1);
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
    xlabel('Polynomial Order');
    ylabel('E_{RMS}');
    legend('training data','testing data')
end

function RMS_vec = compute_RMS(method,Torques_data, param_vec, g)
    err_mat =[];
    if strcmp(method, 'MLSE') || strcmp(method, 'SLSE')
        
        for i = 1:size(Torques_data,3)
            R_pos = analytical_regressor_pos_mat(g,...
                                             Torques_data(1,1,i),...
                                             Torques_data(2,1,i),...
                                             Torques_data(3,1,i),...
                                             Torques_data(4,1,i),...
                                             Torques_data(5,1,i),...
                                             Torques_data(6,1,i));
            R_neg = analytical_regressor_neg_mat(g,...
                                                 Torques_data(1,1,i),...
                                                 Torques_data(2,1,i),...
                                                 Torques_data(3,1,i),...
                                                 Torques_data(4,1,i),...
                                                 Torques_data(5,1,i),...
                                                 Torques_data(6,1,i));
            measure_torque = Torques_data(:,2,i);
            predict_torque = (R_pos*param_vec+R_neg*param_vec)/2;
            err_mat = cat(2,err_mat,measure_torque -predict_torque);
        end
    elseif strcmp(method, 'CAD')
        for i = 1:size(Torques_data,3)
            measure_torque = Torques_data(:,2,i);
            predict_torque = CAD_analytical_regressor(g,...
                                                 Torques_data(1,1,i),...
                                                 Torques_data(2,1,i),...
                                                 Torques_data(3,1,i),...
                                                 Torques_data(4,1,i),...
                                                 Torques_data(5,1,i),...
                                                 Torques_data(6,1,i)) * param_vec;                 
            err_mat = cat(2,err_mat,measure_torque -predict_torque);
        end
    else
        return;
    end
    RMS_vec = [];
    for i=1:7
        RMS_vec = cat(1,RMS_vec, sqrt(sum(err_mat(i,:).^2)/size(Torques_data,3)));
    end
end

