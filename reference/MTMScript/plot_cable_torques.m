data_file = './data/cable_force/Joint4_cable_force_with_pos_neg_dir.mat';
addpath('./gen_code')
load(data_file);
Torques_data = torques_data_process(current_position,...
    desired_effort,...
    'mean',...
     0.3);
 
 color = ['g','r','b','y','m','c','k'];
 index = 1;
 hold on;
 font_size = 20;
 
 % We only want to plot 6 repeat points
data_mat_pos = [];
data_mat_neg = [];
figure(1)
for j=1:6
    x = [];
    y = [];
    for i=index:size(Torques_data,3)/repeat_time+index-1
        x = cat(2,x ,rad2deg(Torques_data(Move_Joint_No,1,i)));
        y = cat(2,y ,Torques_data(Move_Joint_No,2,i));     
        if i<(index+size(Torques_data,3)/repeat_time+index-1)/2
            data_mat_pos(end+1,:) = [rad2deg(Torques_data(Move_Joint_No,1,i)), Torques_data(Move_Joint_No,2,i)];
        else
            data_mat_neg(end+1,:) = [rad2deg(Torques_data(Move_Joint_No,1,i)), Torques_data(Move_Joint_No,2,i)];
        end
    end
    sz = 10;
    scatter(x,y,sz,color(j), 'filled');
    index = size(Torques_data,3)/repeat_time+index; 
end
set(gca,'FontSize',font_size)
% xlabel('Joint position(deg)');
% ylabel('Joint torque(N.m)');

box on
xlabel('{\it q_4} (?)','Interpreter','tex')
ylabel(['$\tau_','0'+4,'$ (Nm)'],'Interpreter','latex','fontweight','bold');  

figure(2)
p_pos = polyfit(data_mat_pos(:,1).',data_mat_pos(:,2).',4);
x_pos = linspace(-70,180);
y_pos = polyval(p_pos,x_pos);
subplot(2,1,1)
hold on
scatter(data_mat_pos(:,1).',data_mat_pos(:,2).',sz,'k', 'filled');
plot(x_pos, y_pos)
hold off



subplot(2,1,2)
hold on
p_neg = polyfit(data_mat_neg(:,1).',data_mat_neg(:,2).',4);
x_neg = linspace(-75,180);
y_neg = polyval(p_neg,x_neg);
scatter(data_mat_neg(:,1).',data_mat_neg(:,2).',sz,'k', 'filled');
plot(x_neg, y_neg)
hold off

figure(1)
x = linspace(-82,190);
plot(x,(polyval(p_pos,x)+polyval(p_neg,x))/2,'k','LineWidth',2)


% txt = '\leftarrow positive direction';
% index = -50;
% text(index, 0.1,txt,'FontSize',font_size)
% x = [.2 .6];
% y = [.1 .5];
% annotation('textarrow',x,y)
% x = [.2 .6];
% y = [.1 .5];
% annotation('textarrow',x,y)

% txt = 'negative direction';
% index = -50;
% text(index, polyval(p_neg,index),txt,)

x = [0.65 0.65];
y = [0.4 0.45];
annotation('textarrow',x,y,'String','Negative direction','FontSize',font_size)

x = [0.45 0.45];
y = [0.85 0.80];
annotation('textarrow',x,y,'String','Positive direction','FontSize',font_size)

x = [0.65 0.6];
y = [0.65 0.65];

text(0,0,'Mean','FontSize',font_size);
%annotation('text',x,y,'String','Mean','FontSize',font_size)

hold off


set(gcf,'position',[10,10,500,400])
% save2pdf('./data/cable_torque')

% legend('training data','testing data')


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