function [out] = exper_extract_behavior_data(folder, animal, session, mode)
%     Add method to load all sessions including ones without LVTS
    if contains(mode, 'bonsai')
        FP_path = fullfile(folder, 'BSDML_FP');
        beh_path = fullfile(folder, 'BSDML_exper');
        out_path = fullfile(folder, 'BSDML_processed');
        % change and fix later
        fnamesFP = get_session_files(FP_path, animal, session, {'LVTTL', 'LVTS', 'FP_', 'FPTS'}, 'animal');
        fnamesEXP = get_session_files(beh_path, animal, session, {'exper'}, 'root');
        experf = char(fnamesEXP{1});
        lvttlf = char(fnamesFP{1});
        lvtsf = char(fnamesFP{2});
        
        % TODO: change this later
        out_session_folder = fullfile(out_path, animal, session);
        if ~exist(out_session_folder, 'dir')
            mkdir(out_session_folder);
        end
        for j=3:4
            fp_sep = regexp(fnamesFP{j}, filesep, 'split');
            if ~exist(fullfile(out_session_folder, fp_sep{end}), 'file')
                copyfile(fnamesFP{j}, out_session_folder);
            end
        end
        blog_f = fullfile(out_session_folder, [animal, '_', session, '_', 'behaviorLOG.mat']);
        if ~exist(blog_f, 'file')
            out = exper_extract_beh_data_bonsai(folder, experf, lvttlf, lvtsf, [animal '_' session]);
            save(blog_f, '-v7.3', 'out');
        end
    else
        out = exper_extract_behavior_data_chris(folder, fnames);
    end
end
    

function [out] = exper_extract_beh_data_bonsai(folder, experf, lvttlf, lvtsf, session_arg)
    % objective: Take in exper data, LV timestamps, Analog_LV, save in hdf5
    % file exper file extracted behavioral events, and additionally save
    % digital_LV_on_time and exper_LV_on_time. 
    % fill in automatic filepath filling
    behavior = load(experf);
    exper = behavior.exper;
    Analog_LV_fileID = fopen(lvttlf);
    Analog_LV = fread(Analog_LV_fileID,'double');
    Analog_LV_timestamp = readmatrix(lvtsf);
    Analog_LV_timestamp = Analog_LV_timestamp(:,1);
    
    %% Obtain behavior times from exper structure
    trial_event_mat = get_2AFC_ITI_EventTimes(behavior);
    
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


    % sanity check Expert_LV_on (in ms) is close to LV_on_time (in ms)
    LV1_on_time=Digital_LV_on_time(1:2:end);
    if length(LV1_on_time)~=length(Expert_LV_on_time) %need to go back and fix
        mod_LV1_on_time = LV1_on_time;
        mod_LV1_on_time = mod_LV1_on_time(length(mod_LV1_on_time)-length(Expert_LV_on_time)+1:end);
        disp('Extra LV_on_time detected. Assuming these are valve test before the behavior session. Please double check!!!');
        LV1_on_time = mod_LV1_on_time;
    end
    
    %interF = griddedInterpolant(LV1_on_time, Expert_LV_on_time, 'linear');
    out.trial_event_mat = trial_event_mat;
    counted_trial=exper.odor_2afc.param.countedtrial.value;
    out.outcome = exper.odor_2afc.param.result.value(1:counted_trial);
    out.port_side = exper.odor_2afc.param.port_side.value(1:counted_trial);
    out.cue_port_side = exper.odor_2afc.param.cue_port_side.value(1:counted_trial);
    out.exper_LV_time = Expert_LV_on_time;
    out.digital_LV_time = LV1_on_time; 
    
end


function [out] = exper_extract_behavior_data_chris(folder, FP_Data)
    green = FP_Data{i}{j}{1};
    red = FP_Data{i}{j}{2};
    fileID = FP_Data{i}{j}{3};
    Analog_LV_timestamp = FP_Data{i}{j}{4};
    MetaData = FP_Data{i}{j}{5};
    behavior_file = Animal_Behavior_Data{i}{j};
    
    green = readmatrix(string(fullfile(folder, green))); %readmatrix is preferred
    red = readmatrix(string(fullfile(folder, red)));
    Analog_LV_timestamp = readmatrix(string(fullfile(folder, Analog_LV_timestamp)));
    behavior = load(string(fullfile(folder, behavior)));
    fileID = fopen(string(fullfile(folder, fileID)));
    if ~isempty(MetaData)
        MetaData = readmatrix(string(fullfile(pload, MetaData)));
    end
    
    green = green(:,1:2);
    red = red(:,1:2);
    Analog_LV_timestamp = Analog_LV_timestamp(:,1);
    exper = behavior.exper;
    Analog_LV = fread(fileID,'double');
    
    
    %% Obtain behavior times from exper structure
    trial_event_mat = get_2AFC_ITI_EventTimes(behavior);

    %% Using Analog_LV and Analog_LV_time to align with exper timestamps
    %% Correct Computer Times (Analog & FP)
    Analog_LV_time = correct_LV_timestamps(Analog_LV_timestamp);
    % TODO: save green & red timestamps in corrected new files and
    % double check about metadata
    green_time = correct_FP_timestamps(green, MetaData);
    red_time = correct_FP_timestamps(red, MetaData); % check if MetaData is good for both
    % Without metadata, ~(50-25)ms jitters still exist (possibly dropped frames)


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
    
    % TODO: verify that RV irrelevant for the sake of argument here 
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
    
    %% convert everything to exper time
    interF = griddedInterpolant(LV1_on_time, Expert_LV_on_time, 'linear');
    green_time = interF(green_time);
    red_time = interF(red_time);
    trial_event_time = exper.rpbox.param.trial_events.value(:,2);
    % maybe also sort the timestamps for all options
    % determine if to zero offset the first event
%     relativeT0 = min([min(green_time), min(red_time), min(trial_event_time)]);
%     green_time = green_time - relativeT0;
%     red_time = red_time - relativeT0;
%     trial_event_time = trial_event_time - relativeT0;
%     trial_event_mat(2, :) = trial_event_mat(2, :) - relativeT0;

    first_LV_on_time = trial_event_time(valid_LV_event);
    first_RV_on_time = trial_event_time(valid_RV_event);
    first_valve_on_time = [first_LV_on_time;first_RV_on_time];
    first_valve_on_time = sort(first_valve_on_time);
    
    
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
    save(fullfile(pathname, strcat(day_folder, '_', 'p  rocessed_data.mat')), '-v7.3', 'out');

    
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