function [lag, r2] = find_lag(x, y)
% Assume x is longer than y, lagging y to achieve lowest regression loss (unity)
    r2 = 0;
    lag = 0;
    for i=1:(length(x)-length(y))
        x1 = x(i:i+length(y)-1);
        X = [ones(length(x1), 1) x1];
        b = X\y;    
        y_hat = X*b;
        r2_i = 1 - sum((y - y_hat).^2)/sum((y - mean(y)).^2);
        if r2_i > r2
            lag = i;
            r2 = r2_i;
        end
    end
end