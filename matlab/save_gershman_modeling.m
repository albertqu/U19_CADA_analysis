function save_gershman_modeling(folder, catalog, outpath)
    %% Takes in catalog list, loads from folder, and run modeling and then 
    %% saves modeling output by animal session in outpath.
    % folder: root storage place of raw data, by default **root** style (see get_session_files.m)
    % catalog: **sorted** table containing animal, session, 
    %          a_i, s_j (indicating its storage location in the final data structure
    %          data{a_i, s_j} = animal-session data)
    
    % Step 1: compile_modeling_data for all animals sessions
    [data, data_fname] = compile_modeling_data(folder, catalog, outpath);
    
    % Step 2: run modeling loops
    tic
%     root = 'Z:\Alumni\Chris Hall\Belief State';
%     outroot = 'D:\U19\ProbSwitch_FP_data';
%     load(fullfile(root, 'compiled_probswitch_data_09122020.mat'),'data');
    [results, bms_results]= fit_models(data);
    sparts = split(data_fname, '_');
    sparts2 = split(sparts{end}, '.');
    timestamp = sparts2{1};
    %  stability test (potentially multiple runs?)
    save(fullfile(outpath, ['gmodeling_results_' sparts{end}]), '-v7.3','results');
    save(fullfile(outpath, ['gmodeling_results_bms_' sparts{end}]), '-v7.3', 'bms_results');
    toc
    
    % Step 3: split modeling results to animal/session folders (save_modeling_vals)
    melt_modeling_out(outpath, results, data, catalog, timestamp, 1);
end