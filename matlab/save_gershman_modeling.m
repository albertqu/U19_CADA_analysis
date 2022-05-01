function save_gershman_modeling(folder, animals, animal_sessions)
    %% compile all data from animals and associated sessions
    
    % Step 1: compile_modeling_data for all animals sessions
    
    data = [];
    
    % Step 2: run modeling loops
    tic
    root = 'Z:\Alumni\Chris Hall\Belief State';
    outroot = 'D:\U19\ProbSwitch_FP_data';
    load(fullfile(root, 'compiled_probswitch_data_09122020.mat'),'data');
    [results, bms_results]= fit_models(data);
    %  stability test
    save(fullfile(outroot, 'results.mat'), '-v7.3','results');
    save(fullfile(outroot, 'bms_results'), '-v7.3', 'bms_results');
    %save('all_modeling_output','iter_results')
    toc
    
    % Step 3: split modeling results to animal/session folders (save_modeling_vals)
end