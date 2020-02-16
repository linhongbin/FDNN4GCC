%load('ReLuNet.mat')
%load('SigmoidNet.mat')
%load('Lagrangian_SinNet.mat')
%load('SinNet.mat')
%load('dynamic_model.mat')
load('SinLogNet.mat')
plot_result(1, train_input_mat, train_output_mat, test_input_mat, test_output_mat)
plot_result(2, train_input_mat, train_output_mat, test_input_mat, test_output_mat)
function plot_result(joint_index, train_input_mat, train_output_mat, test_input_mat, test_output_mat)
    figure(joint_index)
    fntsize = 20;
    hold on
    test_jnt_sam_num = sqrt(size(test_input_mat,1))
    surfplot = surf(rad2deg(reshape(test_input_mat(:,1),[test_jnt_sam_num,test_jnt_sam_num])),...
                     rad2deg(reshape(test_input_mat(:,2),[test_jnt_sam_num,test_jnt_sam_num])),...
                    reshape(test_output_mat(:,joint_index),[test_jnt_sam_num,test_jnt_sam_num]));
    alpha 0.7
    scatterplot = scatter3(rad2deg(train_input_mat(:,1)),...
                           rad2deg(train_input_mat(:,2)),...
                           train_output_mat(:,joint_index),'filled','ro');
    scatterplot.MarkerFaceAlpha = 0.8;
    hold off
    view(3)
    xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
    ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
    zlabel('$\tau_1(N.m)$','interpreter','latex','FontSize', fntsize)
    ticks = [-180 -90 0 90 180]
    set(gca,'FontSize',fntsize);
    xticks(ticks);
    yticks(ticks);
    % ticks = [min(Z1,[],'all'), 0, max(Z1,[],'all')];
    % zticks(ticks)
%     title(sprintf('Train data and ReLu Net predict surface for Joint %d of Acrobot', joint_index))
    legend([surfplot, scatterplot], {'predict surface', 'Training data'});

end
