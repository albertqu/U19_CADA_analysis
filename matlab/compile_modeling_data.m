function [data] = compile_modeling_data(folder, animals, animal_sessions)
    %% Takes in specified animal and sessions and compile all exper data together
    % Option 1: takes in cell of animals, and cells of sessions for each
    % animal (in sorted order) and saves

    %Create empty variable
    for i = 1:length(animal_sessions)
        data(i).a = [];
        data(i).r = [];
        data(i).c = [];
        data(i).s = [];
    end

    for i = 1:length(animal_sessions) %Loop through animals
        for j = 1:length(animal_sessions{i}) %Loop through sessions of an animal
            animal = animals{i};
            behavior = Animal_Behavior_Data{i}{j};
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
                    error('ughhhh')
                end
            else
                if length(outcomes) > 3
                    error('ughhh')
                end
            end
            trials = trials-length(no_choice_idx);
            session = j*ones(1,trials);
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
            data(i).s = [data(i).s session];
            data(i).a = [data(i).a chosenport];
            omission_idx = find(rewarded==1.1);
            incorrect_idx = find(rewarded==2);
            reward_idx = find(rewarded==1.2);
            rewarded(omission_idx) = 0;
            rewarded(incorrect_idx) = 0;
            rewarded(reward_idx) = 1;
            if unique(rewarded) > 2
                error('try again friend!')
            end
            data(i).r = [data(i).r rewarded];
            cue_left_idx = find(rewarded_portside == 2);
            cue_right_idx = find(rewarded_portside == 1);
            rewarded_portside(cue_left_idx) = 1;
            rewarded_portside(cue_right_idx) = 2;
            data(i).c = [data(i).c rewarded_portside];
        end
    end
    pathname = fileparts('/Volumes/Christopher/Wilbrecht Lab/Belief State/');
    savename = fullfile(pathname,'compiled_probswitch_data_07212020');
    save(savename,'data')
end