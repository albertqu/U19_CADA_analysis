function out=get_2AFC_ITI_EventTimes(varargin);
% out=get_2AFC_ITI_EventTimes(exper_file_name)
% get_2AFC_ITI_EventTimes is a function that reads the odor_2AFC data file
% and output eventID, eventTime and trial in three rows.
% eventID=1:   center port in
% eventID=11:  center port in & initiated new trial
% eventID=2:   center port out 
% eventID=3:   left port in
% eventID=4:   left port out
% eventID=44:  Last left port out
% eventID=5:   right port 1n
% eventID=6:   right port out
% eventID=66:  Last right port out
% eventID=71.1:Correct unrewarded outcome
% eventID=71.2:Correct rewarded outcome
% eventID=72:  Incorrect unrewarded outcome
% eventID=73:  Missed to respond outcome
% eventID=74:  Aborted outcome
% 09/17/2020 Lung-Hao Tai

out = [];
if nargin ==1
    arg = varargin{1}; 
    if isstring(arg) || ischar(arg) || iscellstr(arg)
        filename=arg;  
        full_filename=which(filename);
        dr=dir(full_filename);
        if ~isempty(dr) 
            data=load(full_filename);
        else
            data = [];
        end
    elseif isfield(arg, 'exper')
        data = arg;
    else
        data = [];
    end
else
    disp('Please specify an exper filename in string');
    eval('get_2AFC_ITI_EventTimes');
end

ITI_EventTimes=[];
ITI_EventTimes_n=0;

if ~isempty(data)
    trial_events=data.exper.rpbox.param.trial_events.value;
    if isfield(data.exper,'odor_2afc')
        CountedTrial=data.exper.odor_2afc.param.countedtrial.value;
        Result=data.exper.odor_2afc.param.result.value(1:CountedTrial);
        portside=data.exper.odor_2afc.param.port_side.value(1:CountedTrial);
        schedule=data.exper.odor_2afc.param.schedule.value(1:CountedTrial);
%         ITI=data.exper.odor_2afc.param.iti.trial(1:CountedTrial);
%         lwatervalvedur=data.exper.odor_2afc.param.lwatervalvedur.value;
%         rwatervalvedur=data.exper.odor_2afc.param.rwatervalvedur.value;
%         boxrig=data.exper.control.param.expid.value;
        protocol='odor_2afc';
    else
        error('no session found');
        return;
    end
    for k=1:CountedTrial
        if k==1
            tt1=0;
            try
                if ismember(Result(k),[1.2, 1.3]) % two or three drop H2O 
                    tt2=data.exper.odor_2afc.param.trial_events.trial{k}(end,3);
                else
                    tt2=data.exper.odor_2afc.param.trial_events.trial{k}(:,3);
                    if length(tt2)>1
                        tt2=tt2(1);
                    end
                end
            catch
                tt2=0;
            end
        else
            tt1=tt2;
            if ~isempty(data.exper.odor_2afc.param.trial_events.trial{k})
                if ismember(Result(k),[1.2, 1.3]) % two or three drop H2O
                    tt2=data.exper.odor_2afc.param.trial_events.trial{k}(end,3);
                else
                    tt2=data.exper.odor_2afc.param.trial_events.trial{k}(:,3);
                    if length(tt2)>1
                        tt2=tt2(1);
                    end
                end
            else
                % try to find missing trial_events
                if Result(k)==2 && k<CountedTrial
                    tt3=data.exper.odor_2afc.param.trial_events.trial{k+1}(1,3);
                    temp_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<tt3, 2:4);
                    False =[36 5;37 5;38 5;39 3;40 3;41 3];
                    for j=1:size(temp_te,1)
                        if sum(prod(double(repmat(temp_te(j,2:3),size(False,1),1)==False),2))
                            tt2=temp_te(j,1);
                            data.exper.odor_2afc.param.trial_events.trial{k}=temp_te(j,[2 3 1]);
                            exper=data.exper;
                            save([filename '_New.mat'],'exper');
                            disp(['Saved a new version of ' filename ' to ' filename '_New.mat' ]);
                        end
                    end
                else
                    error(['no trial events in odor_2afc for trial ' num2str(k) ', in file:' filename]);
                end
                % check if the Result(k) is 0, if not, manually add back
                % trial events that corresponding to outcome
            end
        end
        % trial_events = (trial, time, state, chan, next state))
        current_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<=tt2, 2:4);
        % C1in in ITI
        c1in_time=current_te(ismember(current_te(:,2),[9 19 512 0 1 11])&ismember(current_te(:,3),1),1);
        if ~isempty(c1in_time)
            % last C1in is the one trigger a new trial
            new_trial_c1in_time=c1in_time(end);
            ITI_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<new_trial_c1in_time & ismember(trial_events(:,4),[1:6]), 2:4);
            last_poke_out=find(ismember(ITI_te(:,3),[4 6]),1,'last');
            if ~isempty(last_poke_out)
                ITI_te(last_poke_out,3)=ITI_te(last_poke_out,3)*10+ITI_te(last_poke_out,3);
            end
            ITI_EventTimes(:,ITI_EventTimes_n+1:ITI_EventTimes_n+length(ITI_te(:,1)))=[ITI_te(:,3)';ITI_te(:,1)';ones(size(ITI_te(:,1)'))*(k-0.5)];
            ITI_EventTimes_n=ITI_EventTimes_n+length(ITI_te(:,1));
            % the one trigger a new trial [event=1.1 , time, trial=k]
            ITI_EventTimes(:,ITI_EventTimes_n+1)=[11 ;new_trial_c1in_time;k]; %
            ITI_EventTimes_n=ITI_EventTimes_n+1;

            %Done with ITI trial events, now look at trial events in Trial K
            Tk_te=trial_events(trial_events(:,2)>new_trial_c1in_time & trial_events(:,2)<=tt2 & ismember(trial_events(:,4),[1:6]), 2:4);
            ITI_EventTimes(:,ITI_EventTimes_n+1:ITI_EventTimes_n+length(Tk_te(:,1)))=[Tk_te(:,3)';Tk_te(:,1)';ones(size(Tk_te(:,1)'))*k];
            ITI_EventTimes_n=ITI_EventTimes_n+length(Tk_te(:,1));

            %Now look at outcome (tt2) if not already added [event=70+result, time, trial=k]
            ITI_EventTimes(:,ITI_EventTimes_n+1)=[70+Result(k);tt2;k]; %
            ITI_EventTimes_n=ITI_EventTimes_n+1;
        end
        
    end
    out=ITI_EventTimes;
else
    Disp('file not found');
end
