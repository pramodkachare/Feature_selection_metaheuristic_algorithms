%% Plot transitory behavior of MH positions
close all
clear
clc

%%
path = '..\CEC2017';
files = dir(path);
files = files(3:end);

algo = 'FLA';
CEC_ver = 19;

%% Generate result directory
if ~isdir([path, '_hist_plots']);
    mkdir([path, '_hist_plots']);
end
%%
for jj = 1:length(files)
    load(fullfile(path, [algo, '_cec', num2str(jj), '.mat']));
    [v, ind] = min(Best_F_FLA);
    x1 = zeros(1, size(conv_curve_FLA, 2));
    x2 = zeros(1, size(conv_curve_FLA, 2));
    for ii = 1:size(conv_curve_FLA, 2)
        x1(ii) = P_hist_FLA{ind}{ii}(1);
        x2(ii) = P_hist_FLA{ind}{ii}(2); 
    end

    %% Display
%     figure();
    h = func_plot(jj, CEC_ver);
    xmin = min(h.XData);  xmax = max(h.XData);
    ymin = min(h.YData);  ymax = max(h.YData);
    hold on
    plot([xmin;xmax], [ymin; ymax], '--k')
    scatter(x1, x2, '*k')
    axis([xmin, xmax, ymin, ymax]);
    scatter(x1(end), x2(end), 'or', 'MarkerFaceColor', 'red')
    hold off
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperPosition', [0 0 5 4.5]);

    saveas(gcf, fullfile([path, '_hist_plots'], [algo, '_cec', num2str(jj), '.png']));
    close
end