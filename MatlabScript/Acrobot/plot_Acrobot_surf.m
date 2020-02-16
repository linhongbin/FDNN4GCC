plot_result(1, input_mat, output_mat)
plot_result(2, input_mat, output_mat)

function plot_result(joint_index, train_input_mat, train_output_mat)
    figure(joint_index)
    fntsize = 20;
    hold on
    jnt_sam_num = sqrt(size(train_input_mat,1))
    surfplot = surf(rad2deg(reshape(train_input_mat(:,1),[jnt_sam_num,jnt_sam_num])),...
                     rad2deg(reshape(train_input_mat(:,2),[jnt_sam_num,jnt_sam_num])),...
                    reshape(train_output_mat(:,joint_index),[jnt_sam_num,jnt_sam_num]));
    alpha 0.7
    hold off
    view(3)
    xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
    ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
    zlabel(sprintf('$\\tau_%d(N.m)$',joint_index),'interpreter','latex','FontSize', fntsize)
    ticks = [-180 -90 0 90 180]
    set(gca,'FontSize',fntsize);
    xticks(ticks);
    yticks(ticks);
    % ticks = [min(Z1,[],'all'), 0, max(Z1,[],'all')];
    % zticks(ticks)
%     title(sprintf('Train data and ReLu Net predict surface for Joint %d of Acrobot', joint_index))
    legend([surfplot], {'predict surface'});

end
