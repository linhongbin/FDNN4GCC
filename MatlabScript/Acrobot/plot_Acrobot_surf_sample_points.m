jnt_sample_num = 34;
fntsize = 20;
jnt_range = linspace(-pi,pi,jnt_sample_num);
[X, Y] = meshgrid(jnt_range);
Z1 = zeros(size(X));
Z2 = zeros(size(X));
Z1_noise = zeros(size(X));
Z2_noise = zeros(size(X));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        G = Acrobot_dynamics(X(i,j), Y(i,j), [0,0], false);
        Z1(i,j) = G(1);
        Z2(i,j) = G(2);
        Z1_noise(i,j) = G(1);
        Z2_noise(i,j) = G(2);
    end
end
figure(1)
hold on
surfplot = surf(rad2deg(X),rad2deg(Y),Z1);
markersize = 12;
splot = scatter3(rad2deg(X(:)),rad2deg(Y(:)), Z1_noise(:),markersize,'filled','ro');
splot.MarkerFaceAlpha = 0.2;
alpha 0.7
hold off
view(3)
xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
zlabel('$\tau_1(N.m)$','interpreter','latex','FontSize', fntsize)
ticks = [-180 -90 0 90 180]
set(gca,'FontSize',fntsize);
xticks(ticks);
yticks(ticks);
ticks = [min(Z1,[],'all'), 0, max(Z1,[],'all')];
zticks(ticks)
%title('Measuring and ground-truth gravity for Joint 1 of Acrobot')
legend([surfplot, splot], {'Ground-truth gravity', 'Measuring gravity'});


figure(2)
hold on
surfplot = surf(rad2deg(X),rad2deg(Y),Z2);
markersize = 12;
splot = scatter3(rad2deg(X(:)),rad2deg(Y(:)), Z2_noise(:),markersize,'filled','ro');
splot.MarkerFaceAlpha = 0.8;
alpha 0.7
hold off
view(3)
xlabel('$q_1(^{\circ})$','interpreter','latex','FontSize',fntsize)
ylabel('$q_2(^{\circ})$','interpreter','latex','FontSize', fntsize)
zlabel('$\tau_2(N.m)$','interpreter','latex','FontSize', fntsize)
ticks = [-180 -90 0 90 180]
set(gca,'FontSize',fntsize);
xticks(ticks);
yticks(ticks);
ticks = [min(Z2,[],'all'), 0, max(Z2,[],'all')];
zticks(ticks)
%title('Measuring and ground-truth gravity for Joint 2 of Acrobot')
legend([surfplot, splot], {'Ground-truth gravity', 'Measuring gravity'});

