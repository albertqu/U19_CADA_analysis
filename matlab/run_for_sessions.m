function [out] = run_for_sessions()
    %folder = '/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch';
    folder = 'Z:\2ABT\ProbSwitch';
    animals = {'D1-R35_RV', 'D1-R34_RT', 'RRM026'};
    sessions = {{'p150', 'p151', 'p152', 'p155', 'p156', 'p157', 'p158'},
        {'p149', 'p150', 'p151', 'p154', 'p155', 'p156', 'p157', 'p158'},
        {'p202', 'p203'}};
    for i=1:length(animals)
        for j=1:length(sessions{i})
            animal = char(animals{i});
            session = char(sessions{i}{j});
            try
                exper_extract_behavior_data(folder, animal, session, 'bonsai');
            catch
                disp([animal '_' session '_error']);
            end
        end
    end     
end

function [out] = run_for_sessions_csv()
%     folder = 'Z:\2ABT\ProbSwitch';
    folder = '/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch';
    csvfile = fullfile(folder, 'probswitch_neural_subset_RRM.csv');
    expr_tb = readtable(csvfile);
    targ_etb = expr_tb(expr_tb.recorded==1, :);
    for i=1:height(targ_etb)
        ani_name = char(targ_etb.animal{i});
        if startsWith(ani_name, 'RRM')
            animal = char(targ_etb.animal{i});
        else
            animal = char(targ_etb.animal_ID{i});
        end
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
        try
            exper_extract_behavior_data(folder, animal, session, 'bonsai');
        catch
            disp([animal '_' session '_error']);
        end
    end     
end