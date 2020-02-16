g = 9.81;
POL = 4;

addpath('./gen_code');

train_data_Collection_file = './data/estimate_parameters_data/MTMR_28002/February-12-2019-20_25_04/dataCollection_info.json';
data_file = 'data/traj_test/traj_test.bag';
bagselect = rosbag(data_file);
bSel_current = select(bagselect,'Topic','/dvrk/MTMR/state_joint_current');
bSel_desired = select(bagselect,'Topic','/dvrk/MTMR/state_joint_desired');

current_struct = readMessages(bSel_current);
desired_struct = readMessages(bSel_desired);

Torques_data = [];
pos_mat = [];
tor_mat = [];
vel_mat = [];

for i = 1:min(size(current_struct,1),size(desired_struct,1))
    Torques_data = cat(3,Torques_data,[current_struct{i}.Position, desired_struct{i}.Effort]);
    pos_mat = cat(2,pos_mat, current_struct{i}.Position);
    vel_mat = cat(2,pos_mat, current_struct{i}.Velocity);
    tor_mat = cat(2,tor_mat, current_struct{i}.Effort);
end
duration = current_struct{end}.Header.Stamp.Sec - current_struct{1}.Header.Stamp.Sec;

decapitate_index = 200;
pos_mat = pos_mat(:,1:end-decapitate_index);
tor_mat = tor_mat(:,1:end-decapitate_index);
vel_mat = vel_mat(:,1:end-decapitate_index);
x = linspace(0,duration,size(pos_mat,2));

RMS_time_index = 10.5:7.8:80;

font_size = 20;
line_width = 2;


f = figure(1)
box on
set(f,'renderer','painters');
hold on
for i = 1:size(pos_mat,1)
    plot(x, rad2deg(pos_mat(i,:)),'LineWidth',line_width); 
end
% % Want to plot out split RMS_time_index 
% for i = 1:size(RMS_time_index,2)
%     plot([RMS_time_index(i),RMS_time_index(i)],[-100 200],'--');
% end
hold off
xlabel('$t$ (s)','Interpreter','latex')
ylabel('Joint Position (Deg)','Interpreter','latex','fontweight','bold')

%set(gca,'XTick',[]);
lgd = legend({'q1','q2','q3','q4','q5','q6','q7'},'NumColumns',4,'Location','north');
set(lgd,'FontName','Times New Roman');

set(gca,'FontSize',font_size)
xticks([0 40 80])
yticks([-100 0 200])
xlim([0, 80]);
ylim([-100,300]);
% save2pdf('./data/predict_traj_pose_plot');


CAD_dynamic_param_vec = CAD_dynamic_vec();
MLSE_tor_mat = [];
SLSE_tor_mat = [];
WPI_tor_mat = [];
CAD_tor_mat = [];

% 
CAD_tor_mat = compute_tor_mat('CAD', pos_mat,vel_mat);

[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(false);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[WPI_dynamic_vec, ~] = slse(train_data_Collection_file);
WPI_tor_mat = compute_tor_mat('WPI', pos_mat, vel_mat,WPI_dynamic_vec);

[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(true,POL*[1,1,1,1,1,1]);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);
[MLSE_dynamic_vec, ~] = mlse(train_data_Collection_file);
MLSE_tor_mat = compute_tor_mat('MLSE', pos_mat, vel_mat, MLSE_dynamic_vec);


f = figure(2)
yticks_mat = [];
yticks_mat(1,:) = [-0.4 0.6];
yticks_mat(2,:) = [0 0.9];
yticks_mat(3,:) = [-0.2 0.7];
yticks_mat(4,:) = [-0.2 0.25];
yticks_mat(5,:) = [-0.08 0.08];
yticks_mat(6,:) = [-0.04 0.06];
yticks_mat(7,:) = [-0.001 0.0003];
line_width = 1.5;
model_line_width = 1;

[ha, pos] =tight_subplot(7,1,[.03 .01],[.07 .04],[.15 0.03]) 
box on
set(f,'renderer','painters');
font_size = 15
for i = 1:7
    axes(ha(i));
    hold on
    choose_index = mod(1:size(tor_mat(i,:),2),20)==0;
    plot(x(choose_index), tor_mat(i,choose_index),'k','LineWidth',line_width);
    plot(x, CAD_tor_mat(i,:),'b','LineWidth',model_line_width);
    plot(x, WPI_tor_mat(i,:),'g','LineWidth',model_line_width);
    plot(x, MLSE_tor_mat(i,:),'r','LineWidth',model_line_width);
    hold off
    if i~=7
        set(gca,'XTick',[]);
    end
    if i ==1
         lgd = legend({'measured', 'CAD', 'SLSE' ,'Our method'} ,'NumColumns',4,'Location','northoutside')
         set(lgd,'FontName','Times New Roman');
set(lgd,'FontName','Times New Roman');
    end
    ylabel(['$\tau_','0'+i,'$ (Nm)'],'Interpreter','latex','fontweight','bold');    
    set(gca,'FontSize',font_size)
    yticks(yticks_mat(i,:));
    xlim([0, 80]);
    ylim(yticks_mat(i,:));
    set(ha(i), 'YTickLabelMode', 'auto')
    box on
end
set(ha(i), 'XTickLabelMode', 'auto')
xticks([0 40 80]);
xlabel('$t (s)$','Interpreter','latex')
set(gcf,'position',[10,10,600,900])
% save2pdf('./data/predict_traj_torque_plot');

choose_index_list = [round(RMS_time_index*duration), size(CAD_tor_mat,2)];
CAD_RMS_err_mat = CAD_tor_mat(:,choose_index_list)-tor_mat(:,choose_index_list);
CAD_RMS_vec = sqrt(sum(CAD_RMS_err_mat.^2,2)/size(CAD_RMS_err_mat,2))

WPI_RMS_err_mat = WPI_tor_mat(:,choose_index_list)-tor_mat(:,choose_index_list);
WPI_RMS_vec = sqrt(sum(WPI_RMS_err_mat.^2,2)/size(WPI_RMS_err_mat,2))

MLSE_RMS_err_mat = MLSE_tor_mat(:,choose_index_list)-tor_mat(:,choose_index_list);
MLSE_RMS_vec = sqrt(sum(MLSE_RMS_err_mat.^2,2)/size(MLSE_RMS_err_mat,2))

function tor_mat = compute_tor_mat(method, pos_mat, vel_mat, dynamic_vec)
    g = 9.81;
    beta_vel_amplitude = [2,2,2,2,8,6,3];
    tor_mat = [];
    for i = 1:size(pos_mat,2)
        q1 = pos_mat(1,i);
        q2 = pos_mat(2,i);
        q3 = pos_mat(3,i);
        q4 = pos_mat(4,i);
        q5 = pos_mat(5,i);
        q6 = pos_mat(6,i);
        v = vel_mat(:,i);
        if strcmp(method,'CAD')
            torque = CAD_analytical_regressor(g,q1,q2,q3,q4,q5,q6)*CAD_dynamic_vec;    
        elseif strcmp(method,'WPI') 
            Regressor_Matrix_Pos = analytical_regressor_pos_mat(g,q2,q3,q4,q5,q6);
            Regressor_Matrix_Neg = analytical_regressor_neg_mat(g,q2,q3,q4,q5,q6);        
            torque = (Regressor_Matrix_Pos*dynamic_vec + Regressor_Matrix_Neg*dynamic_vec)/2;            
        elseif  strcmp(method,'MLSE')
            Regressor_Matrix_Pos = analytical_regressor_pos_mat(g,q1,q2,q3,q4,q5,q6);
            Regressor_Matrix_Neg = analytical_regressor_neg_mat(g,q1,q2,q3,q4,q5,q6);  
            Torques_pos = Regressor_Matrix_Pos*dynamic_vec;
            Torques_neg = Regressor_Matrix_Neg*dynamic_vec;
            alpha = sin_vel(v,beta_vel_amplitude);
            torque = diag(alpha)*Torques_pos+(diag([1,1,1,1,1,1,1])-diag(alpha))*Torques_neg;
        end
         tor_mat = cat(2,tor_mat,torque);
    end
end
function sign_vel = sin_vel(joint_vel_vec, amplitude_vec)
    sign_vel = [];
    for i=1:7
        if joint_vel_vec(i) >= abs(amplitude_vec(i))
            sign_vel(end+1,:) = 1;
        elseif joint_vel_vec(i) <= -abs(amplitude_vec(i))
            sign_vel(end+1,:) = 0;
        else
            sign_vel(end+1,:) = 0.5+sin(pi*(joint_vel_vec(i)/amplitude_vec(i))/2)/2;
        end
    end
end