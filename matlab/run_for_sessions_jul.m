clear global
warning('off','MATLAB:unknownElementsNowStruc');
warning('off','MATLAB:timer:incompatibleTimerLoad');


function [out] = run_for_sessions_jul()
    folder = 'D:\U19\data\DRD_PS';
    csvfile = fullfile(folder, 'drdps_info_subset.csv');
    expr_tb = readtable(csvfile);
    targ_etb = expr_tb;
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
            exper_extract_behavior_generic(folder, animal, session, 'raw', 'processed', 'root');
        catch
            disp([animal '_' session '_error']);
        end
    end     
end