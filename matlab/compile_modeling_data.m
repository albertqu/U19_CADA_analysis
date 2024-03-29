function [data, data_fname] = compile_modeling_data(folder, catalog, outpath)
    %% Takes in specified animal and sessions and compile all exper data together
    
    % folder: root storage place of raw data, by default **root** style (see get_session_files.m)
    % catalog: **sorted** table containing animal, session, 
    %          a_i, s_j (indicating its storage location in the final data structure
    %          data{a_i, s_j} = animal-session data)
    
    n_animals = length(unique(catalog.animal));
    %Create empty variable
    for i = 1:n_animals
        data(i).a = [];
        data(i).r = [];
        data(i).c = [];
        data(i).s = [];
    end
    
    for ij = 1:height(catalog)
        animal = char(catalog.animal{ij});
        session = char(catalog.session{ij}); % deal with animal animalID problem
        fnamesEXP = get_session_files(folder, animal, session, {'exper'}, 'root');
        behavior = load(char(fnamesEXP{1}));
        trials = behavior.exper.odor_2afc.param.countedtrial.value;
        rewarded = behavior.exper.odor_2afc.param.result.value;
        rewarded_portside = behavior.exper.odor_2afc.param.cue_port_side.value;
        chosenport=[];
        undocumented=[];
        %Remove unchosen trials
        rewarded = rewarded(1:trials);
        rewarded_portside = rewarded_portside(1:trials);

        no_choice_idx = find(rewarded==3|rewarded==4|rewarded==0);
        rewarded(no_choice_idx)=[];
        rewarded_portside(no_choice_idx)=[];
        outcomes = unique(rewarded);
        if sum(ismember(rewarded,0)) > 0
            if length(outcomes) > 4
                error('ughhhh');
            end
        else
            if length(outcomes) > 3
                error('ughhh');
            end
        end
        trials = trials-length(no_choice_idx);
        a_i = catalog.a_i(ij);
        s_j = catalog.s_j(ij);
        session = s_j*ones(1,trials);
        for k = 1:trials
            %Matrix of reward history at chosen port (only one can be nonzero at
            %each trial)
            %[L_rewarded R_rewarded L_unrewarded R_unrewarded]
            if rewarded_portside(k) == 1 && rewarded(k) == 1.2 
                chosenport(k) = 2; %right port
            elseif rewarded_portside(k) == 2 && rewarded(k) == 1.2
                chosenport(k) = 1; %left port
            elseif rewarded_portside(k) == 1 && rewarded(k) == 2 
                chosenport(k) = 1; %left port
            elseif rewarded_portside(k) == 1 && rewarded(k) == 1.1 
                chosenport(k) = 2; %right port
            elseif rewarded_portside(k) == 2 && rewarded(k) == 2
                chosenport(k) = 2; %right port
            elseif rewarded_portside(k) == 2 && rewarded(k) == 1.1
                chosenport(k) = 1; %left port
            else 
                undocumented = [undocumented; rewarded_portside(k), rewarded(k)];
            end
        end
        fprintf('ij:%d, a_i: %d\n', ij, a_i);
        data(a_i).s = [data(a_i).s session];
        data(a_i).a = [data(a_i).a chosenport];
        rewarded(rewarded==1.1) = 0; % correct_unrewarded_idx (omission)
        rewarded(rewarded==2) = 0; % incorrect_idx
        rewarded(rewarded==1.2) = 1; % reward_idx
        if unique(rewarded) > 2
            error('try again friend!');
        end
        data(a_i).r = [data(a_i).r rewarded];
        cue_left_idx = find(rewarded_portside == 2);
        cue_right_idx = find(rewarded_portside == 1);
        rewarded_portside(cue_left_idx) = 1;
        rewarded_portside(cue_right_idx) = 2;
        data(a_i).c = [data(a_i).c rewarded_portside];
    end
    timestamp = datestr(now,'yyyymmddHHMMSS');
    data_fname = fullfile(outpath, sprintf('compiled_probswitch_data_%s.mat', timestamp));
    save(data_fname,'data');
end