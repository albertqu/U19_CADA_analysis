function [upper, lower] = bt_trial_avg(data)
% Assume data is a N x T matrix where N is number of trials and T is length
% of time window
  mean_func = @(x)(mean(x, 1));
  ci = bootci(1000,mean_func,data);
  upper = ci(1, :);
  lower = ci(2, :);
end

function [out] = example_use()
  % example
  data = zeros(30, 40);
  xs = 1:40;
  
  for i=1:30
      data(i, :) = sin(xs * 2 * pi / 10);
  end
  data = data + normrnd(0, 1, 30, 40);
  [btupper, btlower] = bt_trial_avg(data);
  data_mean = mean(data, 1);
  x_vector = [xs, fliplr(xs)];
  patch = fill(x_vector, [btupper,fliplr(btlower)], 'm');
  set(patch, 'edgecolor', 'none');
  set(patch, 'FaceAlpha', 0.3);
  hold on;
  plot(xs, data_mean, 'k');
  hold off;
end
