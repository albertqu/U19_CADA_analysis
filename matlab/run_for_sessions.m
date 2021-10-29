function [out] = run_for_sessions()
    folder = '/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch';
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