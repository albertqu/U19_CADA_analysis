function [out] = exper_extract_behavior_generic(folder, animal, session, datapath, outpath, fmode)
%     Add method to load all sessions including ones without LVTS
    % extract behavior mat
    data_path = fullfile(folder, datapath);
    out_path = fullfile(folder, outpath);
    fnamesEXP = get_session_files(data_path, animal, session, {'exper'}, fmode);
    experf = char(fnamesEXP{1});
    fnamesFP = get_session_files(data_path, animal, session,  {'Binary_Matrix', 'timestamp'}, fmode);
    lvttlf = char(fnamesFP{1});
    lvtsf = char(fnamesFP{2});

    % TODO: change this later
    out_session_folder = fullfile(out_path, animal, session);
    if ~exist(out_session_folder, 'dir')
        mkdir(out_session_folder);
    end
    blog_f = fullfile(out_session_folder, [animal, '_', session, '_', 'behaviorLOG.mat']);
    if ~exist(blog_f, 'file')
        out = exper_extract_beh_data_bonsai(folder, experf, lvttlf, lvtsf, [animal '_' session]);
        save(blog_f, '-v7.3', 'out');
    else
        fprintf('Skipping behavior %s %s\n', animal, session);
    end
end

function [out] = exper_extract_beh_data_bonsai(folder, experf, lvttlf, lvtsf, session_arg)
    % objective: Take in exper data, LV timestamps, Analog_LV, save in hdf5
    % file exper file extracted behavioral events, and additionally save
    % digital_LV_on_time and exper_LV_on_time. 
    % fill in automatic filepath filling
    behavior = load(experf);
    exper = behavior.exper;  
    %% Obtain behavior times from exper structure
    trial_event_mat = get_2AFC_ITI_EventTimes(behavior);
    
    if ~isempty(lvttlf)
        Analog_LV_fileID = fopen(lvttlf);
        Analog_LV = fread(Analog_LV_fileID,'double');
        Analog_LV_timestamp = readmatrix(lvtsf);
        Analog_LV_timestamp = Analog_LV_timestamp(:,1);

        %% Using Analog_LV and Analog_LV_time to align with exper timestamps
        %% Correct Computer Times (Analog & FP)
        Analog_LV_time = correct_LV_timestamps(Analog_LV_timestamp);


        %% Sync trial_event time & FP time
        % Analog LV
        %%figure(783);clf
        LV_threshold=(max(Analog_LV) + min(Analog_LV)) / 2;    % volt (0~5 V)
        Digital_LV=Analog_LV>LV_threshold;
        Digital_LV_on_time=Analog_LV_time(find([0;diff(Digital_LV)]>0));
        Digital_LV_off_time=Analog_LV_time(find([0;diff(Digital_LV)]<0));
        % sanity check LV duration=24ms
        % plot(Digital_LV_off_time-Digital_LV_on_time);shg

        % find LV time in exper
        n_trial_events=length(exper.rpbox.param.trial_events.value);
        valid_LV_event=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([17 8 44],n_trial_events,1))==0,2));
        Von_event=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([44 8 48],n_trial_events,1))==0,2));
        LVon_event=valid_LV_event.*NaN;
        for k=1:length(LVon_event)
            LVon_event(k)=Von_event(find(Von_event>valid_LV_event(k),1,'first'));
        end
        Expert_LV_on_time=exper.rpbox.param.trial_events.value(LVon_event,2);


        % sanity check Expert_LV_on (in ms) is close to LV_on_time (in ms)
        LV1_on_time=Digital_LV_on_time(1:2:end);
        if length(LV1_on_time) > length(Expert_LV_on_time)
            disp('Extra LV_on_time detected. Assuming these are valve test before the behavior session. Please double check!!!');
            [lag, r2] = find_lag(LV1_on_time', Expert_LV_on_time);
            if 1-r2 > 0.001
                disp('warning bad fit');
            end
            LV1_on_time = LV1_on_time(lag:lag+length(Expert_LV_on_time)-1);
        elseif length(LV1_on_time) < length(Expert_LV_on_time)
            disp('Extra Expert_LV_on_time detected. Please double check!!!');
            [lag, r2] = find_lag(Expert_LV_on_time, LV1_on_time');
            if 1-r2 > 0.001
                disp('warning bad fit');
            end
            Expert_LV_on_time = Expert_LV_on_time(lag:lag+length(LV1_on_time)-1);
        end
    end
        
    %interF = griddedInterpolant(LV1_on_time, Expert_LV_on_time, 'linear');
    out.trial_event_mat = trial_event_mat;
    counted_trial=exper.odor_2afc.param.countedtrial.value;
    out.outcome = exper.odor_2afc.param.result.value(1:counted_trial);
    out.port_side = exper.odor_2afc.param.port_side.value(1:counted_trial);
    out.cue_port_side = exper.odor_2afc.param.cue_port_side.value(1:counted_trial);
    if ~isempty(lvttlf)
        out.exper_LV_time = Expert_LV_on_time;
        out.digital_LV_time = LV1_on_time; 
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