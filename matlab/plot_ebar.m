function [out] = plot_ebar(data, groups)
    %% data: N x K, group: K x 1
    xs = 1:length(groups);
    data_mean = mean(data, 1);
    [upper, lower] = bt_trial_avg(data);
    bar(xs, data_mean);
    set(gca, 'XTick', xs, 'XTickLabels', groups);
    hold on;
    er = errorbar(xs,data_mean,data_mean-lower,upper-data_mean);
    er.Color = [0, 0, 0];
    er.LineStyle = 'none';  
    hold off
end