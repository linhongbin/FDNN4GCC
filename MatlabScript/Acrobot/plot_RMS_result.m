net_list = {'ReLuNet.mat','SigmoidNet.mat','Lagrangian_SinNet.mat','dynamic_model.mat','SinNet.mat','VanillaSinSigmoid_Net.mat'}
legend_list = {'ReLu Net','Sigmoid Net','Lagrangian-Sin Net','Dynamic Model', 'Sin Net','Vanilla-Sin-Sigmoid Net'}
% net_list = {'ReLuNet.mat','SigmoidNet.mat','Lagrangian_SinNet.mat','SinNet.mat'}
% legend_list = {'ReLu Net','Sigmoid Net','Lagrangian-Sin Net', 'Sin Net'}
% N_arr = [5,8,15];
% std_arr = [1, 5, 9];
N_arr = [2,3,4,5,6,7,8,9,10,12,15,17,20];
std_arr = [1];

result_list ={};
for j=1:size(std_arr,2)
    for k=1:size(N_arr,2) 
        result_mat = [];
        for i=1:numel(net_list)
            file_name=sprintf('N%d_std%d/result/%s', N_arr(k), std_arr(j), net_list{i});
            rel_rms_vec = cal_rms(file_name);
            result_mat = [result_mat,rel_rms_vec]; 
        end
        result_list{end+1} = result_mat;
    end
end

plot_rms(1, 1, legend_list,N_arr, result_list)
%plot_rms(2, 1, legend_list,N_arr, result_list(4:6))
%plot_rms(3, 1, legend_list,N_arr, result_list(7:9))
plot_rms(2, 2, legend_list,N_arr, result_list)
% plot_rms(5, 2, legend_list,N_arr, result_list(4:6))
% plot_rms(6, 2, legend_list,N_arr, result_list(7:9))

function plot_rms(plt_idx, jnt_no, legend_list,N_arr, result_list)
    linewidth = 3;
    mksize =6;
    net_result_list ={};
    for j=1:size(result_list{1},2)
        result_mat = [];
        for i=1:numel(result_list)
            result_mat = [result_mat, result_list{i}(:,j)]
        end
        net_result_list{end+1} = result_mat;
    end
    figure(plt_idx)
    hold on
    fntsize = 20;
    p_list = []
    for i=1:numel(net_result_list)
        p_list(end+1) = plot(N_arr, net_result_list{i}(jnt_no,:),'-o','LineWidth',linewidth,'MarkerSize',mksize);
    end
    hold off
    set(gca,'FontSize',fntsize);
    legend(p_list, legend_list,'location','best');
    xlabel('$N$','interpreter','latex','FontSize',fntsize)
    ylabel(sprintf("$\\epsilon_{RMS_%d} (\\%%)$", jnt_no),'interpreter','latex','FontSize',fntsize)
    ylim([0,1])
end



function rel_rms_vec = cal_rms(load_file)
    load(load_file)
    target_mat = [];
    for i=1:size(test_input_mat,1)
        q1 = test_input_mat(i,1);
        q2 = test_input_mat(i,2);
        G = Acrobot_dynamics(q1, q2, [0,0], true);
        target_mat = [target_mat;G.'];
    end
    rel_rms = sqrt(sum((test_output_mat-target_mat).^2)./sum((target_mat).^2));
    rel_rms_vec = rel_rms.';
end