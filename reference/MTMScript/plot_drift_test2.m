load('./data/drift_test/drift_test_CAD_SLSE1POL_MLSE4POL.mat');
addpath('./gen_code')
MTM_model = MTM_DH_Model;
tranl_err_mat = [];
ori_err_mat = [];
pd_trans_cell = {};
pd_ori_cell = {};
font_size = 24;
line_width = 2;
color_cell_list = {'b','g','r'};

start_joint_pos_mat = [];
end_joint_pos_mat = [];

for i=1:size(CAD_pose_mat_cell_list,2)
    start_joint_pos_mat(:,i,1) = CAD_pose_mat_cell_list{i}(:,1);
end
for i=1:size(CAD_pose_mat_cell_list,2)
    start_joint_pos_mat(:,i,2) = SLSE1POL_pose_mat_cell_list{i}(:,1);
end
for i=1:size(CAD_pose_mat_cell_list,2)
    start_joint_pos_mat(:,i,3) = MLSE_4POL_pose_mat_cell_list{i}(:,1);
end

for i=1:size(CAD_pose_mat_cell_list,2)
    end_joint_pos_mat(:,i,1) = CAD_pose_mat_cell_list{i}(:,end);
end
for i=1:size(CAD_pose_mat_cell_list,2)
    end_joint_pos_mat(:,i,2) = SLSE1POL_pose_mat_cell_list{i}(:,end);
end
for i=1:size(CAD_pose_mat_cell_list,2)
    end_joint_pos_mat(:,i,3) = MLSE_4POL_pose_mat_cell_list{i}(:,end);
end

% transform to cartesian position error and orientation
for j= 1:size(start_joint_pos_mat,3)
    tranl_err_arr = [];
    ori_err_arr = [];
    for i = 1:size(start_joint_pos_mat,2)
        [T_s,~] = FK_Jacob_Geometry(start_joint_pos_mat(:,i,j).', MTM_model.DH, MTM_model.tip ,MTM_model.method);
        [T_e,~] = FK_Jacob_Geometry(end_joint_pos_mat(:,i,j).', MTM_model.DH, MTM_model.tip ,MTM_model.method);
        tranl_err_arr(:,end+1) = norm(T_e(1:3,4) - T_s(1:3,4));
        ori_err_arr(:,end+1) = rad2deg(angleDist(T_e(1:3,1:3), T_s(1:3,1:3)));
    end
    tranl_err_mat(:,j) = tranl_err_arr.';
    ori_err_mat(:,j) = ori_err_arr.';
end

transl_mean_arr = []
transl_mean_arr(:,end+1) = mean(tranl_err_mat(:,1)');
transl_mean_arr(:,end+1) = mean(tranl_err_mat(:,2)');
% position and orientation bar chart
f = figure(1)
set(f,'renderer','painters');
x = 0:0.001:0.3;
hold on
sample_num = size(tranl_err_mat,1);
pd = fitdist(tranl_err_mat(:,1),'normal')
pd_trans_cell{end+1} = pd;
y = pdf(pd,x)*sample_num;
plot(x,y,'b','LineWidth',2)
% area(x.',y.','FaceAlpha',.3)

pd = fitdist(tranl_err_mat(:,2),'normal')
pd_trans_cell{end+1} = pd;
y = pdf(pd,x)*sample_num;
plot(x,y,'g','LineWidth',2)
% area(x.',y.','FaceAlpha',.3)

pd = fitdist(tranl_err_mat(:,3),'normal')
pd_trans_cell{end+1} = pd;
y = pdf(pd,x)*sample_num;
plot(x,y,'r','LineWidth',2)
% area(x.',y.','FaceAlpha',.3)

% histogram(tranl_err_mat(:,2)',25,'FaceColor','g');
% histogram(tranl_err_mat(:,1)',25,'FaceColor','b');
% histogram(tranl_err_mat(:,3)',25,'FaceColor','r');
% plot([transl_mean_arr(1),transl_mean_arr(1)],[0,400],'--','linewidth',line_width,'Color',[0 0 0]+0.1)
% plot([transl_mean_arr(2),transl_mean_arr(2)],[0,400],'--','linewidth',line_width,'Color',[0 0 0]+0.1)
hold off
%bar(tranl_err_mat);
set(gca,'FontSize',font_size)
xlabel('Translational drift (m)','Interpreter','latex');
ylabel('Sample number','Interpreter','latex');
% ylim([0 50])
lgd = legend('CAD','Fontanelli et al.','Our method')
set(lgd,'FontName','Times New Roman');
% save2pdf('./data/drift_test_translation_population_plot')


f = figure(2)
set(f,'renderer','painters');
% bar(ori_err_mat);
% set(gca,'FontSize',font_size)
% ylabel('angle(deg)');
% legend('CAD','4POL')
ori_mean_arr = []
ori_mean_arr(:,end+1) = mean(ori_err_mat(:,1)');
ori_mean_arr(:,end+1) = mean(ori_err_mat(:,2)');


hold on
x = 0:1:200;
sample_num = size(tranl_err_mat,1);
pd = fitdist(ori_err_mat(:,1),'normal')
pd_ori_cell{end+1} = pd;
y = sample_num*pdf(pd,x);
plot(x,y,'b','LineWidth',2)
% area(x.',y.','FaceAlpha',.3)

pd = fitdist(ori_err_mat(:,2),'normal')
y = sample_num*pdf(pd,x);
pd_ori_cell{end+1} = pd;
plot(x,y,'g','LineWidth',2)
% area(x.',y.','FaceAlpha',.3)

pd = fitdist(ori_err_mat(:,3),'normal')
pd_ori_cell{end+1} = pd;
y = sample_num*pdf(pd,x);
plot(x,y,'r','LineWidth',2)
hold off
ylim([0 25])
set(gca,'FontSize',font_size)
xlabel('Rotational drift (Deg)','Interpreter','latex','fontweight','bold');
ylabel('Sample number','Interpreter','latex','fontweight','bold');
lgd = legend('CAD','Fontanelli et al.','Our method')
set(lgd,'FontName','Times New Roman');
% save2pdf('./data/drift_test_orientaion_population_plot')

f = figure(3)
% choose index in one of test data
plot_data_set_index =2;
yticks_mat = [];
yticks_mat(1,:) = [-80 10];
yticks_mat(2,:) = [0 0.12];
yticks_mat(3,:) = [0 40];
ylim_mat = [];
ylim_mat(1,:) = [-100 15];
ylim_mat(2,:) = [0 0.2];
ylim_mat(3,:) = [0 50];
set(f,'renderer','painters');
[ha, pos] =tight_subplot(2,1,[.03 .01],[.2 .15],[.25 0.03]) 

% plot_joint = 5;
% i = plot_joint;

% set(gcf,'position',[0 0 560 520])
% axes(ha(1));
% box on
% hold on
% y = rad2deg(CAD_pose_mat_cell_list{plot_data_set_index}(plot_joint,:));
% x = linspace(0,2,size(y,2));
% plot(x,y ,color_cell_list{1},'LineWidth',line_width);
% y = rad2deg(SLSE1POL_pose_mat_cell_list{plot_data_set_index}(plot_joint,:));
% x = linspace(0,2,size(y,2));
% plot(x,y ,color_cell_list{2},'LineWidth',line_width);
% y = rad2deg(MLSE_4POL_pose_mat_cell_list{plot_data_set_index}(plot_joint,:));
% x = linspace(0,2,size(y,2));
% plot(x,y ,color_cell_list{3},'LineWidth',line_width);
% yticks([yticks_mat(1,1),yticks_mat(1,2)]);   
% ylim([ylim_mat(1,1),ylim_mat(1,2)]); 
% hold off
% ylabel('$q_5$ (Deg)','Interpreter','latex');
% lgd = legend({'CAD','Fontanelli et al.','Our method'},'NumColumns',3,'Location','north');
% set(lgd,'FontName','Times New Roman');
% set(ha(1), 'YTickLabelMode', 'auto')
% grid on
% box on
% set(gca,'FontSize',font_size)

transl_cell_list = {};
ori_cell_list = {};
% CAD
transl_arr = [];
ori_arr = [];
cell_list = CAD_pose_mat_cell_list;
[transl_cell_list{end+1},ori_cell_list{end+1}]  = compute_catesian(CAD_pose_mat_cell_list{plot_data_set_index});
[transl_cell_list{end+1},ori_cell_list{end+1}]  = compute_catesian(SLSE1POL_pose_mat_cell_list{plot_data_set_index});
[transl_cell_list{end+1},ori_cell_list{end+1}]  = compute_catesian(MLSE_4POL_pose_mat_cell_list{plot_data_set_index});




axes(ha(1));
hold on
for i = 1:size(transl_cell_list,2)
    y = transl_cell_list{i};
    x = linspace(0,size(transl_cell_list{i},2)*0.01,size(transl_cell_list{i},2));
    plot(x,y ,color_cell_list{i},'LineWidth',line_width);
end

%     ylim([min([tor_mat(i,:),MLSE_tor_mat(i,:) CAD_tor_mat(i,:)])  max([tor_mat(i,:),MLSE_tor_mat(i,:) CAD_tor_mat(i,:)])])
hold off
set(gca,'XTick',[]);  
ylabel('$\epsilon_d$ (m)','Interpreter','latex');
set(gca,'FontSize',font_size)
set(ha(1), 'YTickLabelMode', 'auto')
grid on
yticks([yticks_mat(2,1),yticks_mat(2,2)]);   
ylim([ylim_mat(2,1),ylim_mat(2,2)]);
box on
lgd = legend({'CAD','Fontanelli et al.','Our method'},'NumColumns',3,'Location','north');
set(lgd,'FontName','Times New Roman');



axes(ha(2));
hold on
for i = 1:size(ori_cell_list,2)
    y = ori_cell_list{i};
    x = linspace(0,size(ori_cell_list{i},2)*0.01,size(ori_cell_list{i},2));
    plot(x,y ,color_cell_list{i},'LineWidth',line_width);
end

hold off

ylabel('$\epsilon_{\theta}$ (Deg)','Interpreter','latex');
xlabel('$t$ (s)','Interpreter','latex')
set(gca,'FontSize',20)
yticks([yticks_mat(3,1),yticks_mat(3,2)]);   
ylim([ylim_mat(3,1),ylim_mat(3,2)]);
grid on
box on

f = figure(4)
% set(gcf,'position',[0 0 560 500])
set(f,'renderer','painters');
cartes_trans_mat = [];
for i=1:size(start_joint_pos_mat,2)
    pos = start_joint_pos_mat(:,i,1).';
    [T_s,~] = FK_Jacob_Geometry(pos, MTM_model.DH, MTM_model.tip ,MTM_model.method);
    cartes_trans_mat = cat(2,cartes_trans_mat,T_s(1:3,4));
end
scatter3(cartes_trans_mat(1,:),cartes_trans_mat(2,:),cartes_trans_mat(3,:),'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75]);
    
xlabel('$x$ (m)','Interpreter','latex','fontweight','bold');
ylabel('$y$ (m)','Interpreter','latex','fontweight','bold');
zlabel('$z$ (m)','Interpreter','latex','fontweight','bold');
set(gca,'FontSize',font_size)
set(ha(2), 'YTickLabelMode', 'auto')
set(ha(2), 'XTickLabelMode', 'auto')
% save2pdf('./data/drift_test_parse_strles_scatter_plot')

% tolerent_translation_err = 0.02;
% % CAD translation scatter
% f = figure(5)
% 
% choose_index = tranl_err_mat(:,1)'<tolerent_translation_err;
% plot3d_scatter_space(f,...
%                      cartes_trans_mat(1,:),...
%                      cartes_trans_mat(2,:),...
%                      cartes_trans_mat(3,:),...
%                      choose_index);
%                  
% % MLSE translation scatter
% f = figure(6)
% choose_index = tranl_err_mat(:,2)'<tolerent_translation_err;
% plot3d_scatter_space(f,...
%                      cartes_trans_mat(1,:),...
%                      cartes_trans_mat(2,:),...
%                      cartes_trans_mat(3,:),...
%                      choose_index);
%                  
%                  
% tolerent_orientation_err = 5;
% % CAD translation scatter
% f = figure(7)
% 
% choose_index = ori_err_mat(:,1)'<tolerent_orientation_err;
% plot3d_scatter_space(f,...
%                      cartes_trans_mat(1,:),...
%                      cartes_trans_mat(2,:),...
%                      cartes_trans_mat(3,:),...
%                      choose_index);
%                  
% % MLSE translation scatter
% f = figure(8)
% choose_index = ori_err_mat(:,2)'<tolerent_orientation_err;
% plot3d_scatter_space(f,...
%                      cartes_trans_mat(1,:),...
%                      cartes_trans_mat(2,:),...
%                      cartes_trans_mat(3,:),...
%                      choose_index);

function plot3d_scatter_space(f,x,y,z,choose_index)
    view(3)
    grid on
    box on
    set(f,'renderer','painters');
    font_size = 20;

    hold on
    scatter3(x(choose_index),...
             y(choose_index),...
             z(choose_index),...
             'MarkerEdgeColor','k',...
             'MarkerFaceColor','g');
    scatter3(x(~choose_index),...
             y(~choose_index),...
             z(~choose_index),...
             'MarkerEdgeColor','k',...
             'MarkerFaceColor','r');
     hold off
    set(gca,'FontSize',font_size)
end


function theta = angleDist(R_act, R_dsr)
    Re = R_act' * R_dsr;
    costheta = 0.5 * (Re(1,1) + Re(2,2) + Re(3,3) - 1);
    if (costheta >= 1.0)
        theta = 0.0;
    elseif (costheta <= -1.0)
        theta = pi;
    else
        theta = acos(costheta);
    end
end

function [transl_arr,ori_arr]  = compute_catesian(joint_pos_mat)
    MTM_model = MTM_DH_Model;
    transl_arr = [];
    ori_arr = []
    [T_s,~] = FK_Jacob_Geometry(joint_pos_mat(:,1)', MTM_model.DH, MTM_model.tip ,MTM_model.method);
    for k =  1:size(joint_pos_mat,2)
        [T_e,~] = FK_Jacob_Geometry(joint_pos_mat(:,k)', MTM_model.DH, MTM_model.tip ,MTM_model.method);
        transl_arr(:,k) = norm(T_e(1:3,4) - T_s(1:3,4));
        ori_arr(:,k) = rad2deg(angleDist(T_e(1:3,1:3), T_s(1:3,1:3)));
    end
end
