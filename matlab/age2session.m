function [session] = age2session(age)
    if mod(age, 1) == 0
        session = sprintf('p%d', age);
    else
        digit = mod(age, 1);
        if abs(digit-0.05) <= 1e-10
            session = sprintf('p%d_session0', floor(age));
        else
            splt = split(string(digit), '.');
            session = sprintf('p%d_session%s', floor(age), char(splt(end)));
        end
    end
end