clear global
warning('off','MATLAB:unknownElementsNowStruc');
warning('off','MATLAB:timer:incompatibleTimerLoad');



function [out] = run_for_sessions_chris_csv()
    folder = 'Z:\2ABT\ProbSwitch';
    csvfile = fullfile(folder, 'probswitch_neural_subset_chris.csv');
    expr_tb = readtable(csvfile);
    targ_etb = expr_tb(strcmp(expr_tb.epoch, 'Probswitch'), :);
    for i=1:height(targ_etb)
        ani_name = char(targ_etb.animal{i});
        animal = char(targ_etb.animal{i});
        animal1 = get_animal_id(char(targ_etb.animal_ID{i}));
        if any(ismember(targ_etb.Properties.VariableNames, 'session'))
            session = char(targ_etb.session(i));
        else
            age = char(targ_etb.age(i));
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
        disp(['trying ' animal ' ' session]);
        try
            exper_extract_behavior_data(folder, animal, session, 'chris');
        catch
            disp([animal '_' session '_error']);
        end
        if ~contains(animal, 'BSD')
            fprintf('trying alternative naming for %s %s\n', animal1, session);
            try
                exper_extract_behavior_data(folder, animal1, session, 'chris');
            catch
                disp([animal1 '_' session '_error']);
            end
        end
    end     
end