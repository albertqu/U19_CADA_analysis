%%% Analyze RR Photometry Data
clear global
warning('off','MATLAB:unknownElementsNowStruc');
warning('off','MATLAB:timer:incompatibleTimerLoad');

% root for saving data
root_save = '/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_Data/';
root_load = '/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_Raw/';

[Animal_Behavior_Data,celltype,FP_Data,data_notes,hemispheres,filenames,AnimalList,sessions,modeling_var,modeling_note,GLM_signals]=Load_BeliefState_Data_prime(root_load);
%modeling_var is either Q value, choice probability, belief state, etc. each animal has a cell, full of
%cells of sessions. modeling_note{2} will specific what the modeling_var
%is, modeling_note{1} will tell what model was used 

time_window=[-2000:50:2000]; %msec

% root_load = '/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/CADA_data/ProbSwitch_FP_Data/';

% Booleans to decide what to plot, analyze, and save. 1=plot; 2=don't plot;
use_BSD_name = 0; % whether to use BSD alias
load_lazy_name = 1; % whether Data is in lazy loading format
detail_stats = 0; % whether to compute the legacy behavior times from Chris' code.
plot_main_events = 0;
plot_multiple_events = 0;
plot_raw = 0; %Plot entire session with markings where different events occur
plot_sub1 = 0; %Only plot trials with ITI < sub_num seconds
sub_num = 1.5;
save_FP_data = 0;
format_GLM = 1; % whether to format data for GLM input or not
lazy_save_GLM = 1; % whether to save GLM in individual files.
plot_behavior = 0;
plot_behavior_ex = 0; %Plot example 300 trials of behavior from trials 900-1200 (can edit below)
plot_prev_next_poke = 0; %whether to include marks for previous and upcoming pokes in plots
plot_movement_comp = 0; %split nose out responses by speed of movement
choose_model = 'regression_4_trials'; %match fieldname in "modeling_var" of model you want to use
plot_modeling = 0; %Separate signals by modeling variables (Q values, choice probabilities, etc.)
plot_modeling1 = 0;
plot_modeling2 = 0;
plot_GLM_signals = 0; %decide whether or not to plot GLM comparisons
plot_green_GLM = 0;
plot_red_GLM = 0;
process_signals = 0; % separate, baseline, and compute df/f & z-score
regression_model = 0; % 1 = individual trial fit; 2 = accumulated trials fit
RL_model = 0; %1 = alpha,beta; 2 = pure bayesian (no trial length knowledge); 3 = bayesian + block length
%4 = bayesian+softmax; 5 = alphaposneg beta; 6 = bayes+blocklength+softmax;
%7 = belief_state+blocks; 8 = belief_state (no blocks)

if regression_model == 1
    log_regression = 4; % How many trials back to regress over
elseif regression_model == 2
    log_regression = 3; % How many trials back to regress over
else
    log_regression = 0;
end

% Set what analysis measure to use
dff_or_zscore='zscore';

data=[];  
%allmodels = getfield(modeling_var.models,choose_model);

for i = 1:length(FP_Data) % Loop through animals
    
    for j = 1:length(FP_Data{i}) % Loop through days of an animal
        disp(filenames{i}{j});
        if save_FP_data
            if isfile('analyzed_test_GLM_FP_data.mat')
                load('analyzed_test_GLM_FP_data.mat');
            end 
        end
        name_parts = strsplit(char(filenames{i}{j}),'_');
        
        hemisphere = '';
        if sum(contains(name_parts,'LH'))
            hemisphere = 'LH';
        else
            hemisphere = 'RH';
        end
        if strcmp(hemisphere,'LH')
            region = hemispheres{i}{1};
        else
            region = hemispheres{i}{2};
        end
            
        for k = 1:length(name_parts)
            age = name_parts{k};
            if contains(age,'p') && ~isnan(str2double(age(2:end)))
                if k < length(name_parts)
                    if contains(name_parts(k+1),'session')
                        session_name = strcat(name_parts{k},'_',name_parts{k+1});
                        break
                    else
                        session_name = name_parts{k};
                    end
                else
                    session_name = name_parts{k};
                    break
                end
            end
        end
        if hemisphere
            session_name = strcat(session_name, '_FP_', hemisphere);
        end
        if use_BSD_name
            day_folder = strcat(AnimalList{i},'_',session_name);
        else
            day_folder = strcat(name_parts{1}, '_', name_parts{2}, '_', session_name);
        end
        tomake = sprintf(strcat(root_save,'%s'),day_folder);
        f_exists = exist(tomake,'dir');
        if ~f_exists
            mkdir(tomake);
            %addpath(day_folder)
        end
        
        pathname = fileparts(root_save);
        pathname = fullfile(pathname, day_folder);
        pload = root_load;
        %pload = fullfile(pload, day_folder);
        green = FP_Data{i}{j}{1};
        red = FP_Data{i}{j}{2};
        fileID = FP_Data{i}{j}{3};
        Analog_LV_timestamp = FP_Data{i}{j}{4};
        MetaData = FP_Data{i}{j}{5};
        behavior = Animal_Behavior_Data{i}{j};
        if load_lazy_name
            green = readmatrix(string(fullfile(pload, green))); %readmatrix is preferred
            red = readmatrix(string(fullfile(pload, red)));
            Analog_LV_timestamp = readmatrix(string(fullfile(pload, Analog_LV_timestamp)));
            behavior = load(string(fullfile(pload, behavior)));
            fileID = fopen(string(fullfile(pload, fileID)));
            if ~isempty(MetaData)
                MetaData = readmatrix(string(fullfile(pload, MetaData)));
            end
        end
        green = green(:,1:2);
        red = red(:,1:2);
        Analog_LV_timestamp = Analog_LV_timestamp(:,1);
        exper = behavior.exper;
        Analog_LV = fread(fileID,'double');
        
        % Pull out data from model of choice
        if ~isempty(modeling_var)
            if isfield(modeling_var{i}(j).latents,'Qs')
                Qs = getfield(modeling_var{i}(j).latents,'Qs'); % Action Values
            end
            if isfield(modeling_var{i}(j).latents,'Ps')
               Ps = getfield(modeling_var{i}(j).latents,'Ps'); % Choice Probabilities
            end
            if isfield(modeling_var{i}(j).latents,'Bs')
               Bs = getfield(modeling_var{i}(j).latents,'Bs'); % Belief States
            end
            if isfield(modeling_var{i}(j).latents,'rpe')
               rpe = getfield(modeling_var{i}(j).latents,'rpe'); % Belief States
            end
        end
        %Qs = modeling_var{1}{i}{j};
        %Ps = modeling_var{2}{i}{j};
        %% Correct Computer Times (Analog & FP)
        Analog_LV_time = correct_LV_timestamps(Analog_LV_timestamp);
        % TODO: save green & red timestamps in corrected new files and
        % double check about metadata
        green_time = correct_FP_timestamps(green, MetaData);
        red_time = correct_FP_timestamps(red, MetaData); % check if MetaData is good for both
        % Without metadata, ~(50-25)ms jitters still exist (possibly dropped frames)
        
        %% Obtain behavior times from exper structure
        trial_event_mat = get_2AFC_ITI_EventTimes(behavior);
        
        %% Sync trial_event time & FP time
        % Analog LV
        %%figure(783);clf
        LV_threshold=2;    % volt (0~5 V)
        Digital_LV=Analog_LV>LV_threshold;
        Digital_LV_on_time=Analog_LV_time(find([0;diff(Digital_LV)]>0));
        Digital_LV_off_time=Analog_LV_time(find([0;diff(Digital_LV)]<0));
        % sanity check LV duration=24ms
        % plot(LV_off_time-LV_on_time);shg

        % find LV time in exper
        n_trial_events=length(exper.rpbox.param.trial_events.value);
        valid_LV_event=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([17 8 44],n_trial_events,1))==0,2));
        Von_event=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([44 8 48],n_trial_events,1))==0,2));
        LVon_event=valid_LV_event.*NaN;
        for k=1:length(LVon_event)
            LVon_event(k)=Von_event(find(Von_event>valid_LV_event(k),1,'first'));
        end
        Expert_LV_on_time=exper.rpbox.param.trial_events.value(LVon_event,2);

        valid_RV_event=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([7 8 44],n_trial_events,1))==0,2));
        RVon_event=valid_RV_event.*NaN;
        for k=1:length(RVon_event)
            RVon_event(k)=Von_event(find(Von_event>valid_RV_event(k),1,'first'));
        end
        Expert_RV_on_time=exper.rpbox.param.trial_events.value(RVon_event,2);

        % sanity check Expert_LV_on (in ms) is close to LV_on_time (in ms)
        LV1_on_time=Digital_LV_on_time(1:2:end);
        if length(LV1_on_time)~=length(Expert_LV_on_time) %need to go back and fix
            mod_LV1_on_time = LV1_on_time;
            mod_LV1_on_time = mod_LV1_on_time(length(mod_LV1_on_time)-length(Expert_LV_on_time)+1:end);
            disp('Extra LV_on_time detected. Assuming these are valve test before the behavior session. Please double check!!!');
            LV1_on_time = mod_LV1_on_time;
        end
        temp=(LV1_on_time-Expert_LV_on_time'*1000);
%         LV1_on_time=Digital_LV_on_time(1:2:end);
%         temp=(LV1_on_time-Expert_LV_on_time'*1000);
        % plot(Analog_LV_time-temp(1),[1 diff(Analog_LV_time(1:end))]);hold on % sanity check
        % plot(LV1_on_time-temp(1),(LV1_on_time-Expert_LV_on_time'*1000)-temp(1),'rd');shg
        interF = griddedInterpolant(Expert_LV_on_time,LV1_on_time, 'linear');
        trial_event_FP_time = interF(exper.rpbox.param.trial_events.value(:,2));
        trial_mat_FP_times = interF(trial_event_mat(2, :)); % before and after correction timestamps lie on identity line.
        % [red_415, red_signal] = split_channels(red, trigger_mode);
        % [green_415, green_signal] = split_channels(green, trigger_mode);
        
        % Here A CRITICAL DECISION has to be made as to whether normalize
        % trial_event_FP_time and green/red by the minimum.
        relativeT0 = min([min(green_time), min(red_time), min(trial_event_FP_time)]);
        green_time = green_time - relativeT0;
        red_time = red_time - relativeT0;
        trial_event_FP_time = trial_event_FP_time - relativeT0;
        trial_event_mat(2, :) = trial_mat_FP_times - relativeT0;

        first_LV_on_time = trial_event_FP_time(valid_LV_event);
        first_RV_on_time = trial_event_FP_time(valid_RV_event);
        first_valve_on_time = [first_LV_on_time;first_RV_on_time];
        first_valve_on_time = sort(first_valve_on_time);
        
        
        %% Logitic Regression -- to be completed
        if log_regression
            [regression_data] = ProbSwitch_Log_Regression(exper,regression_model,log_regression); %log_regression tells how many trials back
            if regression_model == 1
                save_suffix = 'indiv_trial_regression';
            elseif regression_model == 2
                save_suffix = 'accum_trial_regression';
            end    
            name = strcat(filenames{i}{j},'_',save_suffix);
            name = sprintf('%s_%d_trials_back',name,log_regression);
            regression_pathname = fileparts('/Volumes/Christopher/Wilbrecht Lab/Belief State/BeliefState_Regression/');
            %regression_pathname = strcat(regression_pathname,'/',day_folder,'/');
            savename = fullfile(regression_pathname,name);
            save(savename,'regression_data')
        end
        
        %% Computational Modeling -- to be completed
        if RL_model
            [RL_data] = ProbSwitch_RL(exper,RL_model);
            if RL_model == 1
                save_suffix = 'alpha_beta';
            elseif RL_model == 2
                save_suffix = 'prob_bayesian';
            elseif RL_model == 3
                save_suffix = 'prob_bayesian_blocklength';
            elseif RL_model == 4
                save_suffix = 'bayesian_softmax';
            elseif RL_model == 5
                save_suffix = 'alphaposneg_beta';
            elseif RL_model == 6
                save_suffix = 'bayes_block_softmax';
            elseif RL_model == 7
                save_suffix = 'belief_state_blocks';
            elseif RL_model == 8
                save_suffix = 'belief_state';
            end
            name = strcat(filenames{i}{j},'_',save_suffix);
            RL_pathname = fileparts('/Volumes/Christopher/Wilbrecht Lab/Belief State/BeliefState_RL/');
            savename = fullfile(RL_pathname,name);
            save(savename,'RL_data')
        end
        
        % Saving processed data
        %% format data for GLM analysis
        if format_GLM
            % in: [event; trial_event_FP_time; trial], FP_time
            % value saving
            out.value.trial_event_mat = trial_event_mat;
            counted_trial=exper.odor_2afc.param.countedtrial.value;
            out.value.outcome = exper.odor_2afc.param.result.value(1:counted_trial);
            out.value.port_side = exper.odor_2afc.param.port_side.value(1:counted_trial);
            out.value.cue_port_side = exper.odor_2afc.param.cue_port_side.value(1:counted_trial);
            out.value.green_time = green_time;
            out.value.red_time = red_time;
            % maybe gershman modeling and ITI calculations

            % trials & time saving
            evts = trial_event_mat(1, :);
            tefp_times = trial_event_mat(2, :);
            etrial = trial_event_mat(3, :);

            todos = {{(evts == 1) | (evts == 11), 'center_in'},
                     {evts == 11, 'initiate'}, 
                     {evts == 2, 'center_out'},
                     {evts == 3, 'left_in'},
                     {evts == 5, 'right_in'},
                     {evts > 70, 'outcome'},
                     {(evts == 4) | (evts == 44), 'left_out'},
                     {(evts == 6) | (evts == 66), 'right_out'},
                     {evts == 73, 'missed'},
                     {evts == 74, 'abort'}};
            for k=1:length(todos)
                pair = todos{k};
                sels = pair{1};
                label = pair{2};
                sel_times = tefp_times(sels);
                sel_trials = etrial(sels);
                dupls = (diff(sel_times) == 0);
                if sum(dupls) > 0
                    valids = logical([(~dupls) 1]);
                    out.trials.(label) = sel_trials(valids);
                    out.time.(label) = sel_times(valids);
                else
                    out.trials.(label) = sel_trials;
                    out.time.(label) = sel_times;
                end
            end

            % hemisphere: 0: left, 1:right, region: DMS: 0, NAc:1
            out.notes.hemisphere = strcmp(hemisphere, 'RH');
            out.notes.region = strcmp(region, 'NAc');

            %if isfield(modeling_var{i}(j).latents,'Qs')
            %    GLM{i}(j).value.Qs = Qs;
            %    GLM{i}(j).value.relative_Qs = Qs(:,1) - Qs(:,2);
            %end
            %if isfield(modeling_var{i}(j).latents,'Ps')
            %    GLM{i}(j).value.Ps = Ps;
            %end
            %if isfield(modeling_var{i}(j).latents,'Bs')
            %    GLM{i}(j).value.Bs = Bs;
            %end
            %if isfield(modeling_var{i}(j).latents,'rpe')
            %    GLM{i}(j).value.rpe = rpe;
            %end
            save(fullfile(pathname, strcat(day_folder, '_', 'processed_data.mat')), '-v7.3', 'out');
        end 

        %% Find Reward Rate
        if detail_stats
            bin_start = trial_event_FP_time(1);
            bin_end = bin_start + 300000;
            water_rewards = [];
            while bin_end+300000 <= trial_event_FP_time(end)
                if bin_end > trial_event_FP_time(end)
                    bin_end = trial_event_FP_time(end);
                end
                event_bin=find(trial_event_FP_time<=bin_end & trial_event_FP_time>=bin_start);
                LV_water_bin = event_bin(ismember(event_bin,valid_LV_event));
                RV_water_bin = event_bin(ismember(event_bin,valid_RV_event));
                water_reward_bin = 2*(length(LV_water_bin)+length(RV_water_bin));
                water_rewards = [water_rewards water_reward_bin];
                bin_start = bin_end+1;
                bin_end = bin_end + 300000;
            end 
            %data.rewardspermin = water_rewards;
            num_bins = 1:length(water_rewards);
            %figure(3);clf
            %plot(num_bins,water_rewards)
    %%
            counted_trial=exper.odor_2afc.param.countedtrial.value;

            %All_Exper_CLed_on_event=find((prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([3 9],n_trial_events,1))==0,2)|prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([5 19],n_trial_events,1))==0,2)|...
            %    prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([3 19],n_trial_events,1))==0,2)|prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([5 9],n_trial_events,1))==0,2))...
            %    &~(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([9 3 9],n_trial_events,1))==0,2)|prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([19 5 19],n_trial_events,1))==0,2)|...
            %    (prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([19 3 19],n_trial_events,1))==0,2)|prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([9 5 9],n_trial_events,1))==0,2))));
            All_Exper_CLed_on_event=[];
            for m = 1:counted_trial
                event_occured=find(exper.rpbox.param.trial_events.value(:,1)==m-1 &...
                (exper.rpbox.param.trial_events.value(:,5)==9 | exper.rpbox.param.trial_events.value(:,5)==19),1,'first');
                if event_occured
                    All_Exper_CLed_on_event=[All_Exper_CLed_on_event;event_occured];
                end
            end
            time_up_event=find(exper.rpbox.param.trial_events.value(:,3)==10 & exper.rpbox.param.trial_events.value(:,5)==10);
            time_up_trial = exper.rpbox.param.trial_events.value(time_up_event,1)+1;
            time_up_trial = unique(time_up_trial);
            CLed_on_time=trial_event_FP_time(All_Exper_CLed_on_event);
            CLed_on_trials=exper.rpbox.param.trial_events.value(All_Exper_CLed_on_event,1);
            All_Exper_CLed_on_time=exper.rpbox.param.trial_events.value(All_Exper_CLed_on_event,2);
            Last_Exper_CLed_on_time=All_Exper_CLed_on_time(end-9:end);
            % plot(FP_CLed_on_time-temp(1),(FP_CLed_on_time-Last_Exper_CLed_on_time*1000)-temp(1)-25,'ko');shg % light takes one frame to go bright and be thresholded/detected
            All_Outcomes = [first_LV_on_time;first_RV_on_time;CLed_on_time];
            All_Outcomes = sort(All_Outcomes);
            %%
            %Center in/out, and Side in/out events

            temp_center_in_idx = find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([11 1 12],n_trial_events,1))==0,2)|...
                                 prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([1 1 2],n_trial_events,1))==0,2));
            center_in_idx=[];
            for m = 1:counted_trial
                next_c_in = temp_center_in_idx(find(exper.rpbox.param.trial_events.value(temp_center_in_idx,1)==m-1,1,'first'));
                center_in_idx = [center_in_idx; next_c_in];
            end
            %center_out_idx = find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([14 2 15],n_trial_events,1))==0,2)|...
            %                      prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([4 2 5],n_trial_events,1))==0,2)|...
            %                      prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([2 58],n_trial_events,1))==0,2)|...
            %                      prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([2 68],n_trial_events,1))==0,2));
            valid_left_in_idx = find(prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([3 33],n_trial_events,1))==0,2)|...
                                     prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([3 9],n_trial_events,1))==0,2));
            valid_right_in_idx = find(prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([5 34],n_trial_events,1))==0,2)|...
                                      prod((exper.rpbox.param.trial_events.value(:,4:5)-repmat([5 19],n_trial_events,1))==0,2));
            center_trials = exper.rpbox.param.trial_events.value(center_in_idx,1);

            left_in_idx=[];right_in_idx=[];
            left_out_idx=[];right_out_idx=[];center_out_idx=[];
            left_out_choice_trials=[];right_out_choice_trials=[];
            for m = 1:length(center_in_idx)
               center_out_idx = [center_out_idx; find(exper.rpbox.param.trial_events.value(center_in_idx(m):end,4)==2,1,'first')+center_in_idx(m)-1];
               if m == 1
                   next_l_out = []; %left out
                   next_r_out = []; %right out
               else
                   next_l_out = find(exper.rpbox.param.trial_events.value(center_in_idx(m-1):center_in_idx(m),4)==4,1,'last')+center_in_idx(m-1)-1; %left out
                   next_r_out = find(exper.rpbox.param.trial_events.value(center_in_idx(m-1):center_in_idx(m),4)==6,1,'last')+center_in_idx(m-1)-1; %right out
               end
               next_l_in = valid_left_in_idx(find(valid_left_in_idx > center_in_idx(m),1,'first'));
               next_r_in = valid_right_in_idx(find(valid_right_in_idx > center_in_idx(m),1,'first'));
               if next_l_in < next_r_in
                   next_r_in = [];
               elseif next_r_in < next_l_in
                   next_l_in = [];
               end
               if ~isempty(next_l_out) && ~isempty(next_r_out)
                   if next_l_out < next_r_out
                       next_l_out = [];
                   elseif next_r_out < next_l_out
                       next_r_out = [];
                   end
               end
               if m > 1
                   if ~isempty(next_l_out)
                       left_out_choice_trials = [left_out_choice_trials; center_trials(m-1)];
                   elseif ~isempty(next_r_out)    
                       right_out_choice_trials = [right_out_choice_trials; center_trials(m-1)];
                   end
               end
               left_in_idx = [left_in_idx; next_l_in];
               left_out_idx = [left_out_idx; next_l_out];
               right_in_idx = [right_in_idx; next_r_in];
               right_out_idx = [right_out_idx; next_r_out];
            end

            center_in_time = trial_event_FP_time(center_in_idx);
            center_in_time = unique(center_in_time); 
            center_out_time = trial_event_FP_time(center_out_idx);
            center_out_time = unique(center_out_time);
            left_in_choice_time = trial_event_FP_time(left_in_idx);
            left_in_choice_time = unique(left_in_choice_time); %removes redundant choice times that appear due to choice omissions
            right_in_choice_time = trial_event_FP_time(right_in_idx);
            right_in_choice_time = unique(right_in_choice_time);
            left_out_choice_time = trial_event_FP_time(left_out_idx);
            left_out_choice_time = unique(left_out_choice_time);
            right_out_choice_time = trial_event_FP_time(right_out_idx);
            right_out_choice_time = unique(right_out_choice_time);
            left_in_choice_trials = exper.rpbox.param.trial_events.value(left_in_idx,1);
            right_in_choice_trials = exper.rpbox.param.trial_events.value(right_in_idx,1);


            %% Separate Different Trial Types
            %result: 1.2=reward, 1.1 = correct omission, 2 = incorrect, 3 = no
            %choice
            %port_side: 2 = left, 0 = omission, 1 = right
            %cue_port_side: 2 = left, 1 = right

            %Conver trials to index 1 rather than index 0
            center_trials = center_trials+1;
            center_trials = unique(center_trials);
            right_in_choice_trials = right_in_choice_trials+1;
            right_in_choice_trials = unique(right_in_choice_trials);
            right_out_choice_trials = right_out_choice_trials+1;
            right_out_choice_trials = unique(right_out_choice_trials);
            left_in_choice_trials = left_in_choice_trials+1;
            left_in_choice_trials = unique(left_in_choice_trials);
            left_out_choice_trials = left_out_choice_trials+1;
            left_out_choice_trials = unique(left_out_choice_trials);
            CLed_on_trials = CLed_on_trials+1;

            data{i}(j).behavior.left_trials = left_in_choice_trials;
            data{i}(j).behavior.right_trials = right_in_choice_trials;

            rewarded_center_in_time = center_in_time(exper.odor_2afc.param.result.value(center_trials)==1.2);
            rewarded_center_out_time = center_out_time(exper.odor_2afc.param.result.value(center_trials)==1.2);
            rewarded_center_trials = center_trials(exper.odor_2afc.param.result.value(center_trials)==1.2);
            rewarded_left_in_time = left_in_choice_time(exper.odor_2afc.param.result.value(left_in_choice_trials)==1.2);
            rewarded_left_out_time = left_out_choice_time(exper.odor_2afc.param.result.value(left_out_choice_trials)==1.2);
            rewarded_right_trials = exper.rpbox.param.trial_events.value(valid_RV_event,1)+1;
            rewarded_left_trials = exper.rpbox.param.trial_events.value(valid_LV_event,1)+1;

            cued_left_trials = find(exper.odor_2afc.param.cue_port_side.value(1:counted_trial)==2);
            cued_right_trials = find(exper.odor_2afc.param.cue_port_side.value(1:counted_trial)==1);
            first_left_block = [cued_left_trials(1); cued_left_trials(ismember(cued_left_trials,cued_right_trials+1))'];
            first_left_block = unique(first_left_block);
            first_right_block = [cued_right_trials(1); cued_right_trials(ismember(cued_right_trials,cued_left_trials+1))'];
            first_right_block = unique(first_right_block);

            correct_left_omission_trials = cued_left_trials(exper.odor_2afc.param.result.value(cued_left_trials)==1.1);
            correct_right_omission_trials = cued_right_trials(exper.odor_2afc.param.result.value(cued_right_trials)==1.1);
            incorrect_left_unrewarded_trials = cued_right_trials(exper.odor_2afc.param.result.value(cued_right_trials)==2);
            incorrect_right_unrewarded_trials = cued_left_trials(exper.odor_2afc.param.result.value(cued_left_trials)==2);

            correct_left_trials = [rewarded_left_trials' correct_left_omission_trials];
            correct_left_trials = sort(correct_left_trials);
            data{i}(j).behavior.correct_left_trials = correct_left_trials;
            data{i}(j).behavior.frac_left_correct = length(correct_left_trials)/length(left_in_choice_trials);
            correct_right_trials = [rewarded_right_trials' correct_right_omission_trials];
            correct_right_trials = sort(correct_right_trials);
            data{i}(j).behavior.correct_right_trials = correct_right_trials;
            data{i}(j).behavior.frac_right_correct = length(correct_right_trials)/length(right_in_choice_trials);

            %rewarded_left_trials = left_in_choice_trials(exper.odor_2afc.param.result.value(left_in_choice_trials)==1.2);
            rewarded_right_in_time = right_in_choice_time(exper.odor_2afc.param.result.value(right_in_choice_trials)==1.2);
            rewarded_right_out_time = right_out_choice_time(exper.odor_2afc.param.result.value(right_out_choice_trials)==1.2);
            %rewarded_right_trials = right_in_choice_trials(exper.odor_2afc.param.result.value(right_in_choice_trials)==1.2);
            left_trials_center_in_time = center_in_time(ismember(center_trials,left_in_choice_trials));
            left_trials_center_out_time = center_out_time(ismember(center_trials,left_in_choice_trials));
            right_trials_center_in_time = center_in_time(ismember(center_trials,right_in_choice_trials));
            right_trials_center_out_time = center_out_time(ismember(center_trials,right_in_choice_trials));
            CLed_on_right_time = CLed_on_time(ismember(CLed_on_trials,right_in_choice_trials));
            CLed_on_right_trials = CLed_on_trials(ismember(CLed_on_time,CLed_on_right_time));
            CLed_on_left_time = CLed_on_time(ismember(CLed_on_trials,left_in_choice_trials));
            CLed_on_left_trials = CLed_on_trials(ismember(CLed_on_time,CLed_on_left_time));
        end
        
        %Find fraction stay given recent reward history
        
        %Separate correct omission and incorrect unrewarded outcome trials
        
        %Stay and Switch centered at choice before stay, or switch 

        %Stay and Switch centered at second choice of stay, or switch 
    
        %Centered at poke to switch or stay trial, not preceding trial

        %Find time between pokes
        
        %Separate data based on preceding ITI's
     
        % Separate data based on modeling input (Q values, choice
        % probabilities, belief states, etc.
        
        %Separate different types of correct and incorrect trial types
        
        %% Behavioral Plots
        figure_num = 4;
        
        if plot_behavior_ex
            frac_l_choice = [];
            for m = 1:counted_trial
                if m < 3
                    frac_l_choice = [frac_l_choice; sum(ismember(left_in_choice_trials,1:5))/5];    
                elseif m > counted_trial - 2 
                    frac_l_choice = [frac_l_choice; sum(ismember(left_in_choice_trials,left_in_choice_trials(end-4):left_in_choice_trials(end)))/5];
                else
                    frac_l_choice = [frac_l_choice; sum(ismember(left_in_choice_trials,m-2:m+2))/5];
                end
            end
            
            left_blocks={};
            right_blocks={};
            n=1;
            block_start=1;
            mod_cued_left_trials = cued_left_trials(cued_left_trials>=900 & cued_left_trials<=1200);
            for m = 1:length(mod_cued_left_trials)
                if m == length(mod_cued_left_trials)
                    break
                end
                if mod_cued_left_trials(m+1) == mod_cued_left_trials(m)+1
                    continue
                else
                    left_blocks{n}=mod_cued_left_trials(block_start:m);
                    block_start = m+1; 
                    n = n+1;
                end
            end
            n=1;
            block_start=1;
            mod_cued_right_trials = cued_right_trials(cued_right_trials>=900 & cued_right_trials <=1200);
            for m = 1:length(mod_cued_right_trials)
                if m == length(mod_cued_right_trials)
                    break
                end
                if mod_cued_right_trials(m+1) == mod_cued_right_trials(m)+1
                    continue
                else
                    right_blocks{n}=mod_cued_right_trials(block_start:m);
                    block_start = m+1; 
                    n = n+1;
                end
            end
            
            trials = 1:length(frac_l_choice);
            figure(figure_num)
            plot(trials(900:1200),frac_l_choice(900:1200),'k','LineWidth',3);hold on;
            y_rew_l_trials = 1.075*ones(1,length(rewarded_left_trials(rewarded_left_trials>=900 & rewarded_left_trials<=1200)));
            y_rew_r_trials = -0.075*ones(1,length(rewarded_right_trials(rewarded_right_trials>=900 & rewarded_right_trials<=1200)));
            y_unrew_l_trials = 1.05*ones(1,length(CLed_on_left_trials(CLed_on_left_trials>=900 & CLed_on_left_trials<=1200)));
            y_unrew_r_trials = -0.05*ones(1,length(CLed_on_right_trials(CLed_on_right_trials>=900 & CLed_on_right_trials<=1200)));
            plot(rewarded_left_trials(rewarded_left_trials>=900 & rewarded_left_trials<=1200),y_rew_l_trials,'go')
            plot(CLed_on_left_trials(CLed_on_left_trials>=900 & CLed_on_left_trials<=1200),y_unrew_l_trials,'rx')
            plot(rewarded_right_trials(rewarded_right_trials>=900 & rewarded_right_trials<=1200),y_rew_r_trials,'go')
            plot(CLed_on_right_trials(CLed_on_right_trials>=900 & CLed_on_right_trials<=1200),y_unrew_r_trials,'rx')
            for m = 1:length(left_blocks)
                y_left_block = 1.2*ones(1,length(left_blocks{m}));
                plot(left_blocks{m},y_left_block,'color',[0 0 1],'LineWidth',20)
            end
            for m = 1:length(right_blocks)
                y_right_block = 1.2*ones(1,length(right_blocks{m}));
                plot(right_blocks{m},y_right_block,'color',[1 0 0],'LineWidth',20)
            end
            axis([900 1200 -0.1 1.3])
            xlabel('trial number')
            ylabel('fraction left choice')
            title('Behavioral Summary')
            savename = strcat(filenames{i}{j},'_Behavior_Summary','.png');
            figfile = fullfile(pathname,savename);
            saveas(gcf,figfile);close;
            close;
            figure_num = figure_num + 1;
            
        end
        if plot_behavior
            %% Prob switch across block
            left_blocks={};
            right_blocks={};
            n=1;
            block_start=1;
            mod_cued_left_trials = cued_left_trials;
            for m = 1:length(mod_cued_left_trials)
                if m == length(mod_cued_left_trials)
                    break
                end
                if mod_cued_left_trials(m+1) == mod_cued_left_trials(m)+1
                    continue
                else
                    left_blocks{n}=mod_cued_left_trials(block_start:m);
                    block_start = m+1; 
                    n = n+1;
                end
            end
            n=1;
            block_start=1;
            mod_cued_right_trials = cued_right_trials;
            for m = 1:length(mod_cued_right_trials)
                if m == length(mod_cued_right_trials)
                    break
                end
                if mod_cued_right_trials(m+1) == mod_cued_right_trials(m)+1
                    continue
                else
                    right_blocks{n}=mod_cued_right_trials(block_start:m);
                    block_start = m+1; 
                    n = n+1;
                end
            end
            left_block_switch = [];
            right_block_switch = [];
            for m = 1:length(left_blocks)
                for n = 1:length(left_blocks{m})
                    if ismember(left_blocks{m}(n),right_in_choice_trials)
                        left_block_switch(end+1) = n;
                    end
                end
            end
            for m = 1:length(right_blocks)
                for n = 1:length(right_blocks{m})
                    if ismember(right_blocks{m}(n),left_in_choice_trials)
                        right_block_switch(end+1) = n;
                    end
                end
            end
            figure(figure_num)
            histogram(left_block_switch); hold on;
            histogram(right_block_switch)
            legend({'left block switch trial','right block switch trial'})
            xlabel('trial within block')
            ylabel('frequency')
            title('Block switches histogram')
            savename = strcat(filenames{i}{j},'_Block_Switch_Hist','.png');
            figfile = fullfile(pathname,savename);
            saveas(gcf,figfile);close;
            close;
            figure_num = figure_num + 1;
            data{i}(j).behavior.l_block_switches = left_block_switch;
            data{i}(j).behavior.r_block_switches = right_block_switch;
        end
        
        % Plot behavioral mark for different events
    end
end

function [Analog_LV_time] = correct_LV_timestamps(Analog_LV_timestamp)
%% Function takes in analog time stamps and corrects for jitters and provide timestamps for behavior&FP synchronization
% Analog_LV_timestamp: Parsed Binary Files
    %figure(1);clf
    %clf;
    % % General Version
    %         analog_ts_diff = diff(Analog_LV_timestamp);
    %         analog_overtime = (analog_ts_diff-1000)>1;
    %         analog_undertime = (analog_ts_diff-1000)<-1;
    %         pos_neg_pair = [0; analog_overtime] & [analog_undertime; 0];
    %         neg_pos_pair = [analog_overtime; 0] & [0; analog_undertime];
    %         timestamp_shift_target1=find(overtime&undertime==1);

    % screen timestamp for overnight change
    if ~isempty(find(diff(Analog_LV_timestamp)<=-86400000*.95,1))
        Analog_LV_timestamp(find(diff(Analog_LV_timestamp)<=-86400000*.95,1)+1:end)=Analog_LV_timestamp(find(diff(Analog_LV_timestamp)<=-86400000*.95,1)+1:end)+86400000;
    end
    % DAQ records every 1 sec at 1000 sample. Analog_LV_timestamp is the time
    % of each new sweep. It should be 1000ms after previous one. checking for
    % jitters here
%         plot(Analog_LV_timestamp,[0;((diff(Analog_LV_timestamp)-1000)>1)-((diff(Analog_LV_timestamp)-1000)<-1)]+35);hold on;
%         plot(Analog_LV_timestamp,[0;diff([0;((diff(Analog_LV_timestamp)-1000)>1)-((diff(Analog_LV_timestamp)-1000)<-1)])]+30);shg
    %Step 1: finding the index of timestamp that come shorter than 1000ms (probably due to previous delay)
    timestamp_shift_target1=find(([0;(diff(Analog_LV_timestamp)-1000)]>1) & ([(diff(Analog_LV_timestamp)-1000);0]<-1));
    timestamp_shift_target2=find([diff([0;((diff(Analog_LV_timestamp)-1000)>1)-((diff(Analog_LV_timestamp)-1000)<-1)])]==-2);
    if sum(timestamp_shift_target1-timestamp_shift_target2)~=0
        error('try again')
    end
    Analog_LV_timestamp1=Analog_LV_timestamp;
    for i=1:length(timestamp_shift_target1)
        Analog_LV_timestamp1(timestamp_shift_target1(i))=Analog_LV_timestamp(timestamp_shift_target1(i)) +diff(Analog_LV_timestamp1([timestamp_shift_target1(i):timestamp_shift_target1(i)+1]))-1000;
    end

    %Step 2: finding the index of timestamp that come shorter than 1000ms (probably due to previous delay 2 seconds ago)
    timestamp_shift_target1=find(([0;(diff(Analog_LV_timestamp1)-1000)]>1) & ([(diff(Analog_LV_timestamp1)-1000);0]<-1));
    for i=1:length(timestamp_shift_target1)
        Analog_LV_timestamp1(timestamp_shift_target1(i))=Analog_LV_timestamp1(timestamp_shift_target1(i)) +diff(Analog_LV_timestamp1([timestamp_shift_target1(i):timestamp_shift_target1(i)+1]))-1000;
    end

    timestamp_shift_target2=find(([0;(diff(Analog_LV_timestamp1)-1000)]>1) & ([(diff(Analog_LV_timestamp1(2:end))-1000); 0; 0]<-1));
    Analog_LV_timestamp2=Analog_LV_timestamp1;
    for i=1:length(timestamp_shift_target2)
        Analog_LV_timestamp2(timestamp_shift_target2(i)+1)=Analog_LV_timestamp1(timestamp_shift_target2(i)+1) +diff(Analog_LV_timestamp2([timestamp_shift_target2(i):2:timestamp_shift_target2(i)+2]))-2000;
        Analog_LV_timestamp2(timestamp_shift_target2(i))=Analog_LV_timestamp1(timestamp_shift_target2(i)) +diff(Analog_LV_timestamp2([timestamp_shift_target2(i):2:timestamp_shift_target2(i)+2]))-2000;
    end

    % plot corrected timestamp difference at -115
%         plot(Analog_LV_timestamp,[1000;diff(Analog_LV_timestamp2)]-1000-115);shg
%         text(Analog_LV_timestamp(1),-117,'corrected timestamp difference');
    % back propagate time in ms, timestamp was recorded for every 1000 samples.
    Analog_LV_time=zeros(1,length(Analog_LV_timestamp)*1000);
    Analog_LV_time(1000:1000:end)=Analog_LV_timestamp2;
    for i=1:length(Analog_LV_timestamp)
        Analog_LV_time([1:999]+(i-1)*1000)=[-999:1:-1]+Analog_LV_time(i*1000);
    end

%         plot(Analog_LV_time,Analog_LV-100);shg % sanity check
%         plot(Analog_LV_time,[1 diff(Analog_LV_time)]-135);shg % sanity check
%         text(Analog_LV_timestamp(1),-136,'double check timestamp difference');
end

function [FP_time] = correct_FP_timestamps(FP, MetaData)
%% Function that takes in FP data and corrects 
% FP: parsed FP file with dlmread; MetaData: Parsed FP MetaData
    % figure(1);clf;hold on;
    % read FP file
    if (~isempty(MetaData)) && (length(MetaData) ~= length(FP))
        disp('Wrong dimension! skip MetaData');
        MetaData = [];
    end
    jitter_tolerance=5; % mSec
    FP_time=FP(:,1);

    % screen timestamp for overnight change
    if ~isempty(find(diff(FP_time)<=-86400000*.95,1))
        FP_time(find(diff(FP_time)<=-86400000*.95,1)+1:end)=FP_time(find(diff(FP_time)<=-86400000*.95,1)+1:end)+86400000;
    end
    % find most common frame_druation
    [N,Edges,Bin]=histcounts(diff(FP_time),[10:0.1:50]);
%     plot(N);
%     plot(find(N==max(N)),N(find(N==max(N))),'ro');shg;
    frame_druation=Edges(find(N==max(N)))+0.05; % mSec
    if length(frame_druation) ~= 1
        disp('WILDLY Bimodal frame durations, cannot execute fp correct!');
    else
        %%
    %     figure(784);clf
    %     subplot(1,2,1);cla;hold on;
    %     plot(FP_time,[16;diff(FP_time)]);

        %Step 1: finding the index of timestamp that come shorter than frame_druationms (probably due to previous delay)
        timestamp_shift_target1=find(([0;(diff(FP_time)-frame_druation)]>jitter_tolerance) & ([(diff(FP_time)-frame_druation);0]<-jitter_tolerance));
        timestamp_shift_target2=find([diff([0;((diff(FP_time)-frame_druation)>jitter_tolerance)-((diff(FP_time)-frame_druation)<-jitter_tolerance)])]==-2);
        if sum(timestamp_shift_target1-timestamp_shift_target2)~=0
            error('try again')
        end
        FP_time1=FP_time;
        for i=1:length(timestamp_shift_target1)
            FP_time1(timestamp_shift_target1(i))=FP_time(timestamp_shift_target1(i)) +diff(FP_time1([timestamp_shift_target1(i):timestamp_shift_target1(i)+1]))-frame_druation;
        end
    %     plot(FP_time1,[16;diff(FP_time1)]-300);

        %Step 2: finding the index of timestamp that come shorter than frame_druationms (probably due to previous delay 2 frames ago)
        timestamp_shift_target1=find(([0;(diff(FP_time1)-frame_druation)]>jitter_tolerance) & ([(diff(FP_time1)-frame_druation);0]<-jitter_tolerance));
        for i=1:length(timestamp_shift_target1)
            FP_time1(timestamp_shift_target1(i))=FP_time1(timestamp_shift_target1(i)) +diff(FP_time1([timestamp_shift_target1(i):timestamp_shift_target1(i)+1]))-frame_druation;
        end

        timestamp_shift_target2=find(([0;(diff(FP_time1)-frame_druation)]>jitter_tolerance) & ([(diff(FP_time1(2:end))-frame_druation); 0; 0]<-jitter_tolerance));
        FP_time2=FP_time1;
        for i=1:length(timestamp_shift_target2)
            FP_time2(timestamp_shift_target2(i)+1)=FP_time1(timestamp_shift_target2(i)+1) +diff(FP_time2([timestamp_shift_target2(i):2:timestamp_shift_target2(i)+2]))-frame_druation*2;
            FP_time2(timestamp_shift_target2(i))=FP_time1(timestamp_shift_target2(i)) +diff(FP_time2([timestamp_shift_target2(i):2:timestamp_shift_target2(i)+2]))-frame_druation*2;
        end

        %Step 3: finding the index of timestamp that come shorter than frame_druationms (probably due to previous delay many frames ago)
        timestamp_shift_target1=find((diff(FP_time2)-frame_druation)<-jitter_tolerance);
        timestamp_shift_target1=timestamp_shift_target1(timestamp_shift_target1>5); % look back 5 frames to find previous delay
        for i=1:length(timestamp_shift_target1)
            timestamp_shift_target2=find([(diff(FP_time2(timestamp_shift_target1(i)-5:timestamp_shift_target1(i)))-frame_druation)]>jitter_tolerance,1,'last');
            FP_time2(timestamp_shift_target1(i)-(5-timestamp_shift_target2):timestamp_shift_target1(i))=FP_time2(timestamp_shift_target1(i)-(5-timestamp_shift_target2):timestamp_shift_target1(i))+diff(FP_time2([timestamp_shift_target1(i):timestamp_shift_target1(i)+1]))-frame_druation;
        end

        % Step 4: Make use of metadata
        if ~isempty(MetaData)
    %         subplot(1,2,2);cla;hold on;
            MetaData_frame=MetaData(:,1);
            MetaData_timestamp=MetaData(:,2);
            MetaData_time=timestampDecoder(MetaData_timestamp);
    %         plot(FP_time2,[16;diff(FP_time2)]);
    %         subplot(1,2,1);
    %         plot(FP_time,[MetaData_time*1000 - (FP_time-FP_time(1))]);
    %         plot(FP_time1,[MetaData_time*1000 - (FP_time1-FP_time1(1))]-300);
    %         subplot(1,2,2);
    %         plot(FP_time2,[MetaData_time*1000 - (FP_time2-FP_time2(1))]);
            frame_diff=(diff(FP_time2)-frame_druation);
            non_skip_idx=find(frame_diff(1:end-2)<jitter_tolerance & frame_diff(2:end-1)<jitter_tolerance & frame_diff(3:end)<jitter_tolerance)+2;
            skipped_idx=find(~ismember(1:length(FP_time2),non_skip_idx));

            FP_time3=FP_time2;
            B = regress(FP_time2(non_skip_idx),[MetaData_time(non_skip_idx)*1000 ones(length(non_skip_idx),1)]);
            FP_time4=[MetaData_time*1000 ones(length(MetaData_time),1)]*B;
            FP_time3(skipped_idx)=FP_time4(skipped_idx);
            %find those still not aligned
            skipped_idx=find((FP_time3-FP_time4)>jitter_tolerance);
            FP_time3(skipped_idx)=FP_time4(skipped_idx);

    %         plot(FP_time3,[1;diff(MetaData_frame)]*10+90);
    %         plot(FP_time3,[MetaData_time*1000- (FP_time3-FP_time3(1))]-240);
    %         frame_diff=(diff(FP_time3)-frame_druation);
    %         plot(FP_time3(non_skip_idx),frame_diff(non_skip_idx-1)-230);

            FP_time=FP_time3;
        else
        % Step 4b:no metadata
            FP_time=FP_time2;
        end
    %     shg;
    end
end

function [times] = timestampDecoder(timeStamps)
%% Helper function for correct_FP_timestamps

    % converts metadata timestamps recorded in bonsai from point grey camera into seconds
    %
    % input:     timeStamps: metadata timestamps recorded in bonsai
    % output:    times: seconds, normalized such that first time is 0
    
        
    % extract first cycle (first 7 bits)
    cycle1 = bitshift(timeStamps, -25);

    % extract second cycle (following 13 bits)
    cycle2 = bitand(bitshift(timeStamps, -12), 8191) / 8000; % 8191 masks all but but last 13 bits
    
    % account for overflows in counter
    times = cycle1 + cycle2;
    overflows = [0; diff(times) < 0];
    times = times + (cumsum(overflows) * 128);
    
    % offset such that first time is zero
    times = times - min(times);
    
end


function [out] = get_modeling_vars(modelings, animal_number, session)
    % Takes in animal and session, returning the modeling variables
%   i.e. the first day of FP is the 11th session in modeling code, 4th FP day is 16th, etc.
    BSD002_modeling_code = [11,12,13,16,21,24,27,32,33,37,42]; 
    BSD002_ages = {'p151_session1','p151_session2','p153','p156','p232','p235','p238','p243','p244','p248','p252'};
    BSD003_modeling_code = [3,4,7,10,13,17,24,28,32,35,42];
    BSD003_ages = {'p147_LH','p147_RH','p221','p224','p227','p231','p238','p242','p246','p249','p256'};
    BSD004_modeling_code = [7,8,13,15,26,30,34,38,41,48];
    BSD004_ages = {'p145','p146','p220','p222','p233','p237','p241','p245','p248','p255'};
    BSD005_modeling_code = [9,10,11,12,23,29,33,37,40,43,47,50];
    BSD005_ages = {'p102','p103','p104','p105','p189','p195','p199','p203','p206','p209','p213','p216'};
    BSD006_modeling_code = [26,43];
    BSD006_ages = {'p140','p159'};
    BSD007_modeling_code = [25,29,34,43];
    BSD007_ages = {'p139','p143','p148','p157'};
    BSD008_modeling_code = [24,28,33,38,41,45];
    BSD008_ages = {'p138','p142','p147','p152','p156','p161'};
    BSD009_modeling_code = [24,25,29,34,37,42,46];
    BSD009_ages = {'p135_session1','p135_session2','p140','p144','p148','p153','p158'};
    maps = {{BSD002_modeling_code, BSD002_ages},
            {BSD003_modeling_code, BSD003_ages},
            {BSD004_modeling_code, BSD004_ages},
            {BSD005_modeling_code, BSD005_ages},
            {BSD006_modeling_code, BSD006_ages},
            {BSD007_modeling_code, BSD007_ages},
            {BSD008_modeling_code, BSD008_ages},
            {BSD009_modeling_code, BSD009_ages}};
    out = maps;
end