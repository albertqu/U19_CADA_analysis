warning('off','MATLAB:unknownElementsNowStruc');
warning('off','MATLAB:timer:incompatibleTimerLoad');
clear;

thisfile=mfilename;
full_path=which(mfilename);
filesep_idx=strfind(full_path,filesep);
matfile_path=[full_path(1:filesep_idx(end-1)) 'TempNew' filesep];
csvfile_path=[full_path(1:filesep_idx(end-1)) 'BSDML_FP' filesep 'D1-R35-RV' filesep];
data=load([matfile_path 'D1-R35_RV_20210808_Probswitch_p150.mat']);
exper=data.exper;
fileID = fopen([csvfile_path 'D1-R35-RV_Probswitch_p150_LVTTL_2021-08-08T13_11_28.3880448-07_00']);
Analog_LV = fread(fileID,'double');
fclose(fileID);
Analog_LV_timestamp= csvread([csvfile_path 'D1-R35-RV_Probswitch_p150_LVTS_2021-08-08T13_11_28.csv']);
fp_csv = [csvfile_path 'D1-R35-RV_Probswitch_p150_FP_2021-08-08T13_11_53.csv'];
fpts_csv = [csvfile_path 'D1-R35-RV_Probswitch_p150_FPTS_2021-08-08T13_11_28.csv'];
green_ROI_column=4; % Region0G Region3G etc.,
% red_ROI_column=[5 6]; % Region1R Region2R etc.,
red_ROI_column=[]; % Region1R Region2R etc.,

% read FP file
% FrameCounter,Timestamp,Flags,Region0R,Region1G
FP = csvread(fp_csv,1,0);
FPTS = csvread(fpts_csv);
% skip first frame (usually really high value) to help baseline
FP=FP(3:end,:);
FPTS=FPTS(3:end,:);

trigger_mode=1;
% Trigger mode 1 (BSC1) records with 470 and 560 LEDs in phase while 415 is out of phase.
% Trigger mode 2 (BSC2) records with 470 and 560 LEDs out of phase. This trigger mode does not record from the 415 (isosbestic) channel).
% Trigger mode 3 (BSC3) records with all excitation channels (560, 470, and 415) out of phase.),

% FP frame rate and jitter
frame_druation=25; % mSec, now calculated automatically line 465
jitter_tolerance=5; % mSec

% ignore_trials=[]; % calculated later to remove early nan FP data points
time_window=[-2500:25:3000]; % msec
ax_window=[-2000,2500]; % msec
ax=[ax_window(1) ax_window(2) [-1 1].*0.001]; % now automatically adjust Y limit in line 681
bg_fill_color=[.7 .7 .7]; % for 415nm trace
% Hemi_str='L-Hemi, Ca++';
Hemi_str='L-Hemi, DA';

%%
figure(782);clf;
hold on;
% screen timestamp for overnight change
if ~isempty(find(diff(Analog_LV_timestamp)<=-86400000*.95,1))
    Analog_LV_timestamp(find(diff(Analog_LV_timestamp)<=-86400000*.95,1)+1:end)=Analog_LV_timestamp(find(diff(Analog_LV_timestamp)<=-86400000*.95,1)+1:end)+86400000;
end
% DAQ records every 1 sec at 1000 sample. Analog_LV_timestamp is the time
% of each new sweep. It should be 1000ms after previous one. checking for
% jitters here
plot(Analog_LV_timestamp,[0;((diff(Analog_LV_timestamp)-1000)>1)-((diff(Analog_LV_timestamp)-1000)<-1)]+35);
plot(Analog_LV_timestamp,[0;diff([0;((diff(Analog_LV_timestamp)-1000)>1)-((diff(Analog_LV_timestamp)-1000)<-1)])]+30);shg
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
plot(Analog_LV_timestamp,[1000;diff(Analog_LV_timestamp2)]-1000-15);shg
text(Analog_LV_timestamp(1),-18,'corrected timestamp difference');
% back propagate time in ms, timestamp was recorded for every 1000 samples.
Analog_LV_time=zeros(1,length(Analog_LV_timestamp)*1000);
Analog_LV_time(1000:1000:end)=Analog_LV_timestamp2;
for i=1:length(Analog_LV_timestamp)
    Analog_LV_time([1:999]+(i-1)*1000)=[-999:1:-1]+Analog_LV_time(i*1000);
end

plot(Analog_LV_time,Analog_LV);shg % sanity check
plot(Analog_LV_time,[1 diff(Analog_LV_time)]-35);shg % sanity check
text(Analog_LV_timestamp(1),-38,'double check timestamp difference');
%%
% find LV time in exper
n_trial_events=length(exper.rpbox.param.trial_events.value);
valid_LV_event_RwSz3=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([17 8 43],n_trial_events,1))==0,2));
valid_LV_event_RwSz2=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([17 8 44],n_trial_events,1))==0,2));
valid_LV_event_RwSz1=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([17 8 45],n_trial_events,1))==0,2));
valid_LV_event=sort([valid_LV_event_RwSz3' valid_LV_event_RwSz2' valid_LV_event_RwSz1']); %left valve opening events
Von_event_RwSz3=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([43 8 46],n_trial_events,1))==0,2));
Von_event_RwSz32=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([47 8 48],n_trial_events,1))==0,2));
Von_event_RwSz2=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([44 8 48],n_trial_events,1))==0,2));
Von_event_RwSz21=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([49 8 50],n_trial_events,1))==0,2));
Von_event_RwSz1=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([45 8 50],n_trial_events,1))==0,2));
Von_event=sort([Von_event_RwSz3'  Von_event_RwSz32' Von_event_RwSz2' Von_event_RwSz21' Von_event_RwSz1']);
% Von_event=sort([Von_event_RwSz3' Von_event_RwSz2' Von_event_RwSz1']');
LVon_event_RwSz3=valid_LV_event_RwSz3.*NaN;
LVon_event_RwSz32=valid_LV_event_RwSz3.*NaN;
LVon_event_RwSz31=valid_LV_event_RwSz3.*NaN;
LVon_event_RwSz2=valid_LV_event_RwSz2.*NaN;
LVon_event_RwSz21=valid_LV_event_RwSz2.*NaN;
LVon_event_RwSz1=valid_LV_event_RwSz1.*NaN;

for i=1:length(LVon_event_RwSz3)
    LVon_event_RwSz3(i)=Von_event_RwSz3(find(Von_event_RwSz3>valid_LV_event_RwSz3(i),1,'first'));
    %first event number that has the valid left valve opening event for RwSz3,
    %valve opening must go through these steps for reward size == 3.
    LVon_event_RwSz32(i)=Von_event_RwSz32(find(Von_event_RwSz32>valid_LV_event_RwSz3(i),1,'first'));
    LVon_event_RwSz31(i)=Von_event_RwSz21(find(Von_event_RwSz21>valid_LV_event_RwSz3(i),1,'first'));
end
Expert_LV_RwSz3_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz3,2);
Expert_LV_RwSz32_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz32,2);
Expert_LV_RwSz31_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz31,2);

for i=1:length(LVon_event_RwSz2)
    LVon_event_RwSz2(i)=Von_event_RwSz2(find(Von_event_RwSz2>valid_LV_event_RwSz2(i),1,'first'));
    LVon_event_RwSz21(i)=Von_event_RwSz21(find(Von_event_RwSz21>valid_LV_event_RwSz2(i),1,'first'));
end
Expert_LV_RwSz2_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz2,2);
Expert_LV_RwSz21_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz21,2);

for i=1:length(LVon_event_RwSz1)
    LVon_event_RwSz1(i)=Von_event_RwSz1(find(Von_event_RwSz1>valid_LV_event_RwSz1(i),1,'first'));
end
Expert_LV_RwSz1_on_time=exper.rpbox.param.trial_events.value(LVon_event_RwSz1,2);

Expert_LV_on_time=sort([Expert_LV_RwSz1_on_time;Expert_LV_RwSz2_on_time;Expert_LV_RwSz21_on_time;...
    Expert_LV_RwSz3_on_time;Expert_LV_RwSz32_on_time;Expert_LV_RwSz31_on_time]);
%the LV on time at the first point of the sequence
Expert_LV1_on_time=sort([Expert_LV_RwSz1_on_time;Expert_LV_RwSz2_on_time;Expert_LV_RwSz3_on_time]);
Expert_LV1_on_event=sort([LVon_event_RwSz1 LVon_event_RwSz2 LVon_event_RwSz3]);
Expert_LV2_on_time=sort([Expert_LV_RwSz21_on_time;Expert_LV_RwSz32_on_time]);
Expert_LV2_on_event=sort([LVon_event_RwSz21 LVon_event_RwSz32]);
%
% [lia1, loc1]=ismember(Expert_LV_first_on_time,Expert_LV_on_time);% A is found in B
% LV_first_on_time=LV_on_time(loc1); %length(loc1)= number of LV on event including R1,2,3

% Right valve opening event, but not used for analog LV time alignment;
% only left valve information is useful for time alignment.
valid_RV_event_RwSz3=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([7 8 43],n_trial_events,1))==0,2));
valid_RV_event_RwSz2=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([7 8 44],n_trial_events,1))==0,2));
valid_RV_event_RwSz1=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([7 8 45],n_trial_events,1))==0,2));
valid_RV_event=sort([valid_RV_event_RwSz3' valid_RV_event_RwSz2' valid_RV_event_RwSz1']);

RVon_event_RwSz3=valid_RV_event_RwSz3.*NaN;
RVon_event_RwSz32=valid_RV_event_RwSz3.*NaN;
RVon_event_RwSz31=valid_RV_event_RwSz3.*NaN;
RVon_event_RwSz2=valid_RV_event_RwSz2.*NaN;
RVon_event_RwSz21=valid_RV_event_RwSz2.*NaN;
RVon_event_RwSz1=valid_RV_event_RwSz1.*NaN;

for i=1:length(RVon_event_RwSz3)
    RVon_event_RwSz3(i)=Von_event_RwSz3(find(Von_event_RwSz3>valid_RV_event_RwSz3(i),1,'first'));
    RVon_event_RwSz32(i)=Von_event_RwSz32(find(Von_event_RwSz32>valid_RV_event_RwSz3(i),1,'first'));
    RVon_event_RwSz31(i)=Von_event_RwSz21(find(Von_event_RwSz21>valid_RV_event_RwSz3(i),1,'first'));
end
Expert_RV_RwSz3_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz3,2);
Expert_RV_RwSz32_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz32,2);
Expert_RV_RwSz31_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz31,2);

for i=1:length(RVon_event_RwSz2)
    RVon_event_RwSz2(i)=Von_event_RwSz2(find(Von_event_RwSz2>valid_RV_event_RwSz2(i),1,'first'));
    RVon_event_RwSz21(i)=Von_event_RwSz21(find(Von_event_RwSz21>valid_RV_event_RwSz2(i),1,'first'));
end
Expert_RV_RwSz2_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz2,2);
Expert_RV_RwSz21_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz21,2);

for i=1:length(RVon_event_RwSz1)
    RVon_event_RwSz1(i)=Von_event_RwSz1(find(Von_event_RwSz1>valid_RV_event_RwSz1(i),1,'first'));
end
Expert_RV_RwSz1_on_time=exper.rpbox.param.trial_events.value(RVon_event_RwSz1,2);

Expert_RV_on_time=sort([Expert_RV_RwSz1_on_time;Expert_RV_RwSz2_on_time;Expert_RV_RwSz21_on_time;...
    Expert_RV_RwSz3_on_time;Expert_RV_RwSz32_on_time;Expert_RV_RwSz31_on_time]);
Expert_RV1_on_time=sort([Expert_RV_RwSz1_on_time;Expert_RV_RwSz2_on_time;Expert_RV_RwSz3_on_time]);
Expert_RV1_on_event= sort([RVon_event_RwSz3' RVon_event_RwSz2' RVon_event_RwSz1']);
%%
figure(783);clf
% sanity check
subplot(3,10,1:9);
if mean(Analog_LV)>1.5 % floating analog input
    Analog_LV=Analog_LV-mean(Analog_LV);
end
LV_threashold=1;    % volt (0~5 V)
Digital_LV_1v_threshold=Analog_LV>LV_threashold;
Digital_LV_1v_on_time=Analog_LV_time([0;diff(Digital_LV_1v_threshold)]>0);
LV_threashold=4;    % volt (0~5 V)
Digital_LV_4v_threshold=Analog_LV>LV_threashold;
Digital_LV_4v_on_time=Analog_LV_time([0;diff(Digital_LV_4v_threshold)]>0);
spurious_Digital_LV_1v_on_time=Digital_LV_1v_on_time.*NaN;
for i=1:length(spurious_Digital_LV_1v_on_time)
   if sum(Digital_LV_4v_on_time>=Digital_LV_1v_on_time(i) & Digital_LV_4v_on_time<=(Digital_LV_1v_on_time(i)+5))==0
       spurious_Digital_LV_1v_on_time(i)=Digital_LV_1v_on_time(i);
   end
end
if length(Digital_LV_1v_on_time)~=length(Digital_LV_4v_on_time)
    plot(Analog_LV_time,Analog_LV*10-50,'g-');hold on
    plot(spurious_Digital_LV_1v_on_time,-40,'rd');
    plot(Digital_LV_4v_on_time,-10,'bd');
    text(spurious_Digital_LV_1v_on_time(1),-52,'Spurious threshold crossing events found (red dimonds)');
    disp('Spurious threshold crossing events found (red dimonds)');
end

LV_threashold=2;    % volt (0~5 V)
Digital_LV=Analog_LV>LV_threashold;
Digital_LV_on_time=Analog_LV_time(find([0;diff(Digital_LV)]>0));
Digital_LV_off_time=Analog_LV_time(find([0;diff(Digital_LV)]<0));
% sanity check LV duration= 15~25ms
plot(Digital_LV_on_time,Digital_LV_off_time-Digital_LV_on_time,'b-');shg
text(Digital_LV_on_time(10),29,'LV duration (ms), blue line ');
diff_Digital_LV_on_time=diff(Digital_LV_on_time);
plot(Digital_LV_on_time([1e3 diff_Digital_LV_on_time]<200),diff_Digital_LV_on_time(diff_Digital_LV_on_time<200),'r-');shg
text(Digital_LV_on_time(10),129,'LV interval (ms), red line ');

subplot(3,10,10);hold on;
view([90 -90]);
Edges=[0:2:100];
[N,Edges,Bin]=histcounts(Digital_LV_off_time-Digital_LV_on_time,Edges);
plot(Edges(1:end-1)+1,N,'b-');
plot(Edges(find(N==max(N)))+1,N(find(N==max(N))),'bo');shg;
LV_druation=Edges(find(N==max(N)))+1; % mSec

Edges=[50:2:200];
[N,Edges,Bin]=histcounts(diff(Digital_LV_on_time),Edges);
plot(Edges(1:end-1)+1,N,'r-');hold on;
plot(Edges(find(N==max(N)))+1,N(find(N==max(N))),'ro');shg;
LV_interval=Edges(find(N==max(N)))+1; % mSec
xlabel('LV duration (blue) and interval (red) mSec');

% find LV2_on_time
LVn_list=Digital_LV_on_time.*NaN;
LVn_list(1)=1;
for i=2:length(Digital_LV_on_time)
   if abs(diff_Digital_LV_on_time(i-1)-LV_interval)/LV_interval<0.1
       LVn_list(i)=LVn_list(i-1)+1;
   else
       LVn_list(i)=1;
   end
end
LV1_on_time=Digital_LV_on_time(LVn_list==1);
LV2_on_time=Digital_LV_on_time(LVn_list==2);
LV1of2_on_time=LV2_on_time.*NaN;
for i=1:length(LV2_on_time)
    LV1of2_on_time(i)=LV1_on_time(find(LV1_on_time<LV2_on_time(i),1,'last'));
end
% Method 1:try using the largest inter-reward-interval (IRI) rank to align
diff_Expert_LV2_on_time=diff(Expert_LV2_on_time'*1000);
[val,idx]=sort(diff_Expert_LV2_on_time,'descend');
diff_LV2_on_time=diff(LV2_on_time);
[val2,idx2]=sort(diff_LV2_on_time,'descend');
% determine if sliding alignment is necesssary
idx_slide=0;
idx2_slide=0;
if length(idx)==length(idx2)
    idx_slide=0;
    idx2_slide=0;
    starting_rank=1;
elseif length(idx)>length(idx2)
    rep_idx2=repmat(idx2,length(idx)-length(idx2)+1,1);
    rep_idx=rep_idx2.*NaN;
    for i=1:length(idx)-length(idx2)+1
        rep_idx(i,:)=idx(i:length(idx2)+i-1);
    end
    diff_rep=diff(rep_idx,1,2)==diff(rep_idx2,1,2);
    % find first 3 matched rank and sliding alignment(n_slide)
    if ~isempty(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1  & diff_rep(:,3:end)==1,1))
        idx_slide=mod(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)-1,(length(idx)-length(idx2)+1));
        starting_rank=ceil(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)/(length(idx)-length(idx2)+1));
    else
        disp('Did not find a sliding number that makes alignment possible');
    end
elseif length(idx)<length(idx2)
    rep_idx=repmat(idx,length(idx2)-length(idx)+1,1);
    rep_idx2=rep_idx.*NaN;
    for i=1:length(idx2)-length(idx)+1
        rep_idx2(i,:)=idx2(i:length(idx)+i-1);
    end
    diff_rep=diff(rep_idx,1,2)==diff(rep_idx2,1,2);
    % find first 3 matched rank and sliding alignment(n_slide)
    if ~isempty(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1  & diff_rep(:,3:end)==1,1))
        idx2_slide=mod(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)-1,(length(idx2)-length(idx)+1));
        starting_rank=ceil(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)/(length(idx2)-length(idx)+1));
    else
        disp('Did not find a sliding number that makes alignment possible');
    end
end
% check if the index difference of the first three (largest) analog LV IRI
% (inter-reward interval) rank are then same in Phtometry time and Exper
% time. If there is Spurious triggers, largest rank can be skipped by
% starting_rank.
end_rank=starting_rank+2;
if prod(diff(idx(starting_rank:end_rank))==diff(idx2(starting_rank:end_rank)))
    % now find max number of matches to align
    for i=end_rank+1:min(length(idx),length(idx2))
        if prod(diff(idx((starting_rank:i)+idx_slide))==diff(idx2((starting_rank:i)+idx2_slide)))
            end_rank=i;
        end
    end
    matched_idx=unique([idx((starting_rank:end_rank+idx_slide)) idx((starting_rank:end_rank+idx_slide))+1]);
    matched_idx2=unique([idx2((starting_rank:end_rank)) idx2((starting_rank:end_rank)+idx2_slide)+1]);

    % add other matching LV time points between matched_idices
    for i=2:length(matched_idx)
        if diff(matched_idx(i-1:i))>1
            diff_Expert_LV2_on_time=diff(Expert_LV2_on_time(matched_idx(i-1):matched_idx(i))'*1000);
            [sval,sidx]=sort(diff_Expert_LV2_on_time,'descend');
            diff_LV2_on_time=diff(LV2_on_time(matched_idx2(i-1):matched_idx2(i)));
            [sval2,sidx2]=sort(diff_LV2_on_time,'descend');
            end_rank=0;
            starting_rank=1;
            for ii=1:min(length(sidx),length(sidx2))
                if ~isempty(diff(sidx(starting_rank:ii))==diff(sidx2(starting_rank:ii))) && prod(diff(sidx(starting_rank:ii))==diff(sidx2(starting_rank:ii)))
                    end_rank=ii;
                elseif sum(diff(sidx(1:ii))==diff(sidx2(1:ii)))==1
                    starting_rank=ii-1;
                end
            end
            smatched_idx=unique([sidx(starting_rank:end_rank) sidx(starting_rank:end_rank)+1])+matched_idx(i-1)-1;
            smatched_idx2=unique([sidx2(starting_rank:end_rank) sidx2(starting_rank:end_rank)+1])+matched_idx2(i-1)-1;
            matched_idx=[matched_idx smatched_idx];
            matched_idx2=[matched_idx2 smatched_idx2];
        end
    end
    % add other matching LV time points after last matched_idx
    if diff([matched_idx(i) length(Expert_LV2_on_time)])>1
        diff_Expert_LV2_on_time=diff(Expert_LV2_on_time(matched_idx(i):end)'*1000);
        [sval,sidx]=sort(diff_Expert_LV2_on_time,'descend');
        diff_LV2_on_time=diff(LV2_on_time(matched_idx2(i):end));
        [sval2,sidx2]=sort(diff_LV2_on_time,'descend');
        end_rank=0;
        starting_rank=1;
        for ii=1:min(length(sidx),length(sidx2))
            if ~isempty(diff(sidx(starting_rank:ii))==diff(sidx2(starting_rank:ii))) && prod(diff(sidx(starting_rank:ii))==diff(sidx2(starting_rank:ii)))
                end_rank=ii;
            elseif sum(diff(sidx(1:ii))==diff(sidx2(1:ii)))==1
                starting_rank=ii-1;
            end
        end
        smatched_idx=unique([sidx(starting_rank:end_rank) sidx(starting_rank:end_rank)+1])+matched_idx(i)-1;
        smatched_idx2=unique([sidx2(starting_rank:end_rank) sidx2(starting_rank:end_rank)+1])+matched_idx2(i)-1;
        matched_idx=[matched_idx smatched_idx];
        matched_idx2=[matched_idx2 smatched_idx2];
    end
    % add other matching LV time points before first matched_idx
    if matched_idx(1)>1
        diff_Expert_LV2_on_time=diff(Expert_LV2_on_time(1:matched_idx(1))'*1000);
        [sval,sidx]=sort(diff_Expert_LV2_on_time,'descend');
        diff_LV2_on_time=diff(LV2_on_time(1:matched_idx2(1)));
        [sval2,sidx2]=sort(diff_LV2_on_time,'descend');
        idx_slide=0;
        starting_rank=1;
        if length(sidx)==length(sidx2)
            idx_slide=0;
            starting_rank=1;
        elseif length(sidx)>length(sidx2)
            rep_idx2=repmat(sidx2,length(sidx)-length(sidx2)+1,1);
            rep_idx=rep_idx2.*NaN;
            for i=1:length(sidx)-length(sidx2)+1
                rep_idx(i,:)=sidx(i:length(sidx2)+i-1);
            end
            diff_rep=diff(rep_idx,1,2)==diff(rep_idx2,1,2);
            % find first 3 matched rank and sliding alignment(n_slide)
            if ~isempty(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1  & diff_rep(:,3:end)==1,1))
                idx_slide=mod(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)-1,(length(sidx)-length(sidx2)+1));
                starting_rank=ceil(find(diff_rep(:,1:end-2)==1 & diff_rep(:,2:end-1)==1 & diff_rep(:,3:end)==1,1)/(length(sidx)-length(sidx2)+1));
                end_rank=0;
                for ii=1:min(length(sidx),length(sidx2))
                    if ~isempty(diff(sidx((starting_rank:ii)+idx_slide))==diff(sidx2(starting_rank:ii))) && prod(diff(sidx((starting_rank:ii)+idx_slide))==diff(sidx2(starting_rank:ii)))
                        end_rank=ii;
                    end
                end
                smatched_idx=unique([sidx((starting_rank:end_rank)+idx_slide) sidx((starting_rank:end_rank)+idx_slide)+1]);
                smatched_idx2=unique([sidx2(starting_rank:end_rank) sidx2(starting_rank:end_rank)+1]);
                matched_idx=[matched_idx smatched_idx];
                matched_idx2=[matched_idx2 smatched_idx2];
            else
                disp('Did not find a sliding number that makes alignment possible');
            end
        elseif length(idx)<length(idx2)
        end
    end
    matched_idx=unique(matched_idx);
    matched_idx2=unique(matched_idx2);
    % now fill in the rest that matches the number of LV2 in between
    if matched_idx(1)==matched_idx2(1) && matched_idx(1)>1
        matched_idx=[matched_idx 1:matched_idx(1)-1];
        matched_idx2=[matched_idx2 1:matched_idx2(1)-1];
    end
    for i=2:length(matched_idx)
        if diff(matched_idx(i-1:i))==diff(matched_idx2(i-1:i)) && diff(matched_idx(i-1:i))>1
            matched_idx=[matched_idx matched_idx(i-1):matched_idx(i)];
            matched_idx2=[matched_idx2 matched_idx2(i-1):matched_idx2(i)];
        end
    end
    matched_idx=unique(matched_idx);
    matched_idx2=unique(matched_idx2);
    matched_Expert_LV2_on_time=Expert_LV2_on_time(matched_idx);
    matched_LV2_on_time=LV2_on_time(matched_idx2);

else
    % Method 2: old method, keep for checking new code
    if length(LV2_on_time)>length(Expert_LV2_on_time) % extra LV, check data
        matched_Expert_LV2_on_time = Expert_LV2_on_time;
        matched_LV2_on_time = LV2_on_time;
        matched_LV2_on_time = matched_LV2_on_time(length(matched_LV2_on_time)-length(Expert_LV2_on_time)+1:end);
        disp('Extra LV_on_time detected. Assuming these are valve test before the behavior session. Please double check!!!');
    elseif length(LV2_on_time)<length(Expert_LV2_on_time) % fewer LV, check data
        % Method 2: only works if recording all the way towards the end
        LV_trials=find(exper.odor_2afc.param.result.value==1.2 &exper.odor_2afc.param.port_side.value==2);
        ignore_trials=1:LV_trials(length(Expert_LV2_on_time)-length(LV2_on_time));
        matched_LV2_on_time = LV2_on_time;
        matched_Expert_LV2_on_time = Expert_LV2_on_time;
        matched_Expert_LV2_on_time = matched_Expert_LV2_on_time(length(matched_Expert_LV2_on_time)-length(LV2_on_time)+1:end);
        disp('Extra Expert_LV_on_time detected. Assuming recording starts at the middle of behavior session. Please double check!!!');
    end
    disp('Used old method, Trouble shoot inter-reward-interval (IRI) rank alignment');
end
trial_event_FP_time2 = interp1(matched_Expert_LV2_on_time,matched_LV2_on_time,exper.rpbox.param.trial_events.value(:,2),'linear','extrap');
trial_event_FP_time = trial_event_FP_time2;
% correction for events outside LV_on_time(1) or LV_on_time(end)
idx=exper.rpbox.param.trial_events.value(:,2)<matched_Expert_LV2_on_time(1) | exper.rpbox.param.trial_events.value(:,2)>matched_Expert_LV2_on_time(end);
trial_event_FP_time(idx) = interp1(matched_Expert_LV2_on_time([1 end]),matched_LV2_on_time([1 end]),exper.rpbox.param.trial_events.value(idx,2),'linear','extrap');

subplot(3,10,1:9);
temp=(matched_LV2_on_time(1)-matched_Expert_LV2_on_time(1)'*1000);
plot(matched_LV2_on_time,matched_LV2_on_time-(matched_Expert_LV2_on_time'*1000+temp));shg;hold on
title('FP time- exper time');
ylabel('mSec');
% plot un-matched events
if sum(~ismember(Expert_LV2_on_time,matched_Expert_LV2_on_time))>0
    plot(Expert_LV2_on_time(~ismember(Expert_LV2_on_time,matched_Expert_LV2_on_time))*1000+temp,20,'ko');shg
end
if sum(~ismember(LV2_on_time,matched_LV2_on_time))>0
    plot(LV2_on_time(~ismember(LV2_on_time,matched_LV2_on_time)),0,'mo');shg
end
% plot(LV2_on_time,0,'rd');shg
% plot(Expert_LV2_on_time*1000+temp,1,'kd');shg
%%
figure(783);
subplot(3,10,11:20);hold on;
% read FP file
% cut off FP signal 40 sec after behavior session stopped
FPTS=FPTS(FPTS< (trial_event_FP_time(end)+1000*40));
FP=FP(FPTS< (trial_event_FP_time(end)+1000*40),:);
% screen timestamp for overnight change
if ~isempty(find(diff(FPTS)<=-86400000*.95,1))
    FPTS(find(diff(FPTS)<=-86400000*.95,1)+1:end)=FPTS(find(diff(FPTS)<=-86400000*.95,1)+1:end)+86400000;
end
% find most common frame_duration
Edges=[10:0.1:50];
[N,Edges,Bin]=histcounts(diff(FPTS),Edges);
plot(Edges(1:end-1)+0.05,N);
xlabel('FP frame duration (mSec)');
plot(Edges(find(N==max(N)))+0.05,N(find(N==max(N))),'ro');shg;
frame_duration=mean(Edges(find(N==max(N))))+0.05; % mSec

%%
figure(784);clf
subplot(3,10,1:8);hold on;
%470 signal (cyan)
flag_470=6;
FP_470_signal=FP(FP(:,3)==flag_470,[green_ROI_column red_ROI_column]);
FP_470_time=FPTS(FP(:,3)==flag_470);
jump_idx=[];
if ~isempty(green_ROI_column)
    plot(FP_470_time,FP_470_signal(:,1:length(green_ROI_column)),'g-');
    % down sample to 4 seconds stretch to detect sudden change in signal
    frames_per_Xsec=find(FP_470_time>FP_470_time(1)+1000*2,1);
    downsampled_FP_470_signal=FP_470_signal(1:frames_per_Xsec:end);
    downsampled_FP_470_time=FP_470_time(1:frames_per_Xsec:end);
    [n,x]=hist(gca,diff(downsampled_FP_470_signal),10);
    if prod(n(7:9)==0)==1 && n(10)>0
        disp('upward jump found');
        jump_idx=find(diff(downsampled_FP_470_signal)>x(9));
        plot(downsampled_FP_470_time(jump_idx),downsampled_FP_470_signal(jump_idx),'rd');
        plot(downsampled_FP_470_time(jump_idx+1),downsampled_FP_470_signal(jump_idx+1),'kd');
    elseif prod(n(2:4)==0)==1 && n(1)>0
        disp('downward jump found');
        jump_idx=find(diff(downsampled_FP_470_signal)<x(2));
        plot(downsampled_FP_470_time(jump_idx),downsampled_FP_470_signal(jump_idx),'rd');
        plot(downsampled_FP_470_time(jump_idx+1),downsampled_FP_470_signal(jump_idx+1),'kd');
    end
end
if ~isempty(red_ROI_column)
    plot(FP_470_time,FP_470_signal(:,[1:length(red_ROI_column)]+length(green_ROI_column)),'r-');
end
FP_470_signal_baseline=nan(size(FP_470_signal));
FP_470_signal_dff=nan(size(FP_470_signal));
for c=1:size(FP_470_signal,2)
    if isempty(jump_idx)
        FP_470_signal_fit = fit(FP_470_time,FP_470_signal(:,c),'exp2');
        FP_470_signal_baseline(:,c)= FP_470_signal_fit(FP_470_time);
        FP_470_signal_dff(:,c)=(FP_470_signal(:,c)-FP_470_signal_baseline(:,c))./FP_470_signal_baseline(:,c);
    elseif length(jump_idx)==1
        if jump_idx>1
        jump_segment_idx=1:(jump_idx-1)*frames_per_Xsec;
        FP_470_signal_fit = fit(FP_470_time(jump_segment_idx),FP_470_signal(jump_segment_idx,c),'exp2');
        FP_470_signal_baseline(jump_segment_idx,c)= FP_470_signal_fit(FP_470_time(jump_segment_idx));
        FP_470_signal_dff(jump_segment_idx,c)=(FP_470_signal(jump_segment_idx,c)-FP_470_signal_baseline(jump_segment_idx,c))./FP_470_signal_baseline(jump_segment_idx,c);
        end
        jump_segment_idx=(jump_idx)*frames_per_Xsec+1:length(FP_470_signal);
        FP_470_signal_fit = fit(FP_470_time(jump_segment_idx),FP_470_signal(jump_segment_idx,c),'exp2');
        FP_470_signal_baseline(jump_segment_idx,c)= FP_470_signal_fit(FP_470_time(jump_segment_idx));
        FP_470_signal_dff(jump_segment_idx,c)=(FP_470_signal(jump_segment_idx,c)-FP_470_signal_baseline(jump_segment_idx,c))./FP_470_signal_baseline(jump_segment_idx,c);
    elseif length(jump_idx)>1
        if jump_idx>1
        jump_segment_idx=1:(jump_idx-1)*frames_per_Xsec;
        FP_470_signal_fit = fit(FP_470_time(jump_segment_idx),FP_470_signal(jump_segment_idx,c),'exp2');
        FP_470_signal_baseline(jump_segment_idx,c)= FP_470_signal_fit(FP_470_time(jump_segment_idx));
        FP_470_signal_dff(jump_segment_idx,c)=(FP_470_signal(jump_segment_idx,c)-FP_470_signal_baseline(jump_segment_idx,c))./FP_470_signal_baseline(jump_segment_idx,c);
        end
        for s=1:length(jump_idx)-1
            jump_segment_idx=(jump_idx(s))*frames_per_Xsec+1:(jump_idx(s+1)-1)*frames_per_Xsec;
            FP_470_signal_fit = fit(FP_470_time(jump_segment_idx),FP_470_signal(jump_segment_idx,c),'exp2');
            FP_470_signal_baseline(jump_segment_idx,c)= FP_470_signal_fit(FP_470_time(jump_segment_idx));
            FP_470_signal_dff(jump_segment_idx,c)=(FP_470_signal(jump_segment_idx,c)-FP_470_signal_baseline(jump_segment_idx,c))./FP_470_signal_baseline(jump_segment_idx,c);
        end
        jump_segment_idx=(jump_idx(s+1))*frames_per_Xsec+1:length(FP_470_signal);
        FP_470_signal_fit = fit(FP_470_time(jump_segment_idx),FP_470_signal(jump_segment_idx,c),'exp2');
        FP_470_signal_baseline(jump_segment_idx,c)= FP_470_signal_fit(FP_470_time(jump_segment_idx));
        FP_470_signal_dff(jump_segment_idx,c)=(FP_470_signal(jump_segment_idx,c)-FP_470_signal_baseline(jump_segment_idx,c))./FP_470_signal_baseline(jump_segment_idx,c);
    end
end
plot(FP_470_time,FP_470_signal_baseline,'k-');
th=text(FP_470_time(round(length(FP_470_time)*.7)),FP_470_signal_baseline(round(length(FP_470_time)*.7)),'470 nm exp2 baseline');
set(th,'color',[0 0 0]);
%415 signal (magenta)
flag_415=1;
FP_415_signal=FP(FP(:,3)==flag_415,[green_ROI_column red_ROI_column]);
FP_415_time=FPTS(FP(:,3)==flag_415);
if ~isempty(green_ROI_column)
    plot(FP_415_time,FP_415_signal(:,1:length(green_ROI_column)),'y-');
        % down sample to 4 seconds stretch to detect sudden change in signal
    frames_per_Xsec=find(FP_415_time>FP_415_time(1)+1000*2,1);
    downsampled_FP_415_signal=FP_415_signal(1:frames_per_Xsec:end);
    downsampled_FP_415_time=FP_415_time(1:frames_per_Xsec:end);
    [n,x]=hist(gca,diff(downsampled_FP_415_signal),10);
    if prod(n(7:9)==0)==1 && n(10)>0
        disp('upward jump found');
        jump_idx=find(diff(downsampled_FP_415_signal)>x(9));
        plot(downsampled_FP_415_time(jump_idx),downsampled_FP_415_signal(jump_idx),'rd');
        plot(downsampled_FP_415_time(jump_idx+1),downsampled_FP_415_signal(jump_idx+1),'kd');
    elseif prod(n(2:4)==0)==1 && n(1)>0
        disp('downward jump found');
        jump_idx=find(diff(downsampled_FP_415_signal)<x(2));
        plot(downsampled_FP_415_time(jump_idx),downsampled_FP_415_signal(jump_idx),'rd');
        plot(downsampled_FP_415_time(jump_idx+1),downsampled_FP_415_signal(jump_idx+1),'kd');
    end
end
if ~isempty(red_ROI_column)
    plot(FP_415_time,FP_415_signal(:,[1:length(red_ROI_column)]+length(green_ROI_column)),'m-');
end
FP_415_signal_baseline=nan(size(FP_415_signal));
FP_415_signal_dff=nan(size(FP_415_signal));
for c=1:size(FP_415_signal,2)
    if isempty(jump_idx)
        FP_415_signal_fit = fit(FP_415_time,FP_415_signal(:,c),'exp2');
        FP_415_signal_baseline(:,c)= FP_415_signal_fit(FP_415_time);
        FP_415_signal_dff(:,c)=(FP_415_signal(:,c)-FP_415_signal_baseline(:,c))./FP_415_signal_baseline(:,c);
    elseif length(jump_idx)==1
        if jump_idx>1
        jump_segment_idx=1:(jump_idx-1)*frames_per_Xsec;
        FP_415_signal_fit = fit(FP_415_time(jump_segment_idx),FP_415_signal(jump_segment_idx,c),'exp2');
        FP_415_signal_baseline(jump_segment_idx,c)= FP_415_signal_fit(FP_415_time(jump_segment_idx));
        FP_415_signal_dff(jump_segment_idx,c)=(FP_415_signal(jump_segment_idx,c)-FP_415_signal_baseline(jump_segment_idx,c))./FP_415_signal_baseline(jump_segment_idx,c);
        end
        jump_segment_idx=(jump_idx)*frames_per_Xsec+1:length(FP_415_signal);
        FP_415_signal_fit = fit(FP_415_time(jump_segment_idx),FP_415_signal(jump_segment_idx,c),'exp2');
        FP_415_signal_baseline(jump_segment_idx,c)= FP_415_signal_fit(FP_415_time(jump_segment_idx));
        FP_415_signal_dff(jump_segment_idx,c)=(FP_415_signal(jump_segment_idx,c)-FP_415_signal_baseline(jump_segment_idx,c))./FP_415_signal_baseline(jump_segment_idx,c);
    elseif length(jump_idx)>1
        if jump_idx>1
        jump_segment_idx=1:(jump_idx-1)*frames_per_Xsec;
        FP_415_signal_fit = fit(FP_415_time(jump_segment_idx),FP_415_signal(jump_segment_idx,c),'exp2');
        FP_415_signal_baseline(jump_segment_idx,c)= FP_415_signal_fit(FP_415_time(jump_segment_idx));
        FP_415_signal_dff(jump_segment_idx,c)=(FP_415_signal(jump_segment_idx,c)-FP_415_signal_baseline(jump_segment_idx,c))./FP_415_signal_baseline(jump_segment_idx,c);
        end
        for s=1:length(jump_idx)-1
            jump_segment_idx=(jump_idx(s))*frames_per_Xsec+1:(jump_idx(s+1)-1)*frames_per_Xsec;
            FP_415_signal_fit = fit(FP_415_time(jump_segment_idx),FP_415_signal(jump_segment_idx,c),'exp2');
            FP_415_signal_baseline(jump_segment_idx,c)= FP_415_signal_fit(FP_415_time(jump_segment_idx));
            FP_415_signal_dff(jump_segment_idx,c)=(FP_415_signal(jump_segment_idx,c)-FP_415_signal_baseline(jump_segment_idx,c))./FP_415_signal_baseline(jump_segment_idx,c);
        end
        jump_segment_idx=(jump_idx(s+1))*frames_per_Xsec+1:length(FP_415_signal);
        FP_415_signal_fit = fit(FP_415_time(jump_segment_idx),FP_415_signal(jump_segment_idx,c),'exp2');
        FP_415_signal_baseline(jump_segment_idx,c)= FP_415_signal_fit(FP_415_time(jump_segment_idx));
        FP_415_signal_dff(jump_segment_idx,c)=(FP_415_signal(jump_segment_idx,c)-FP_415_signal_baseline(jump_segment_idx,c))./FP_415_signal_baseline(jump_segment_idx,c);
    end
end
plot(FP_415_time,FP_415_signal_baseline,'k-');
th=text(FP_415_time(round(length(FP_415_time)*.7)),FP_415_signal_baseline(round(length(FP_415_time)*.7)),'415 nm exp2 baseline');
set(th,'color',[0 0 0]);
ylabel('FP ROI intensity');
xlabel('time (mSec)');

%plot df/f*1000 in green
subplot(3,1,2);hold on;
if ~isempty(green_ROI_column)
    plot(FP_470_time,FP_470_signal_dff(:,1:length(green_ROI_column))*1000,'g-');
end
if ~isempty(red_ROI_column)
    plot(FP_470_time,FP_470_signal_dff(:,(1:length(red_ROI_column))+length(green_ROI_column))*1000,'r-');
end
th=text(FP_470_time(round(length(FP_470_time)*.7)),max(FP_470_signal_dff(:))*500,'470 nm df/f *1000');
set(th,'color',[0 .6 0]);
ax=[ax_window(1) ax_window(2) -nanstd([FP_470_signal_dff(:);FP_415_signal_dff(:)])*plot_scale nanstd([FP_470_signal_dff(:);FP_415_signal_dff(:)])*plot_scale ];

if ~isempty(green_ROI_column)
    h=plot(FP_415_time,FP_415_signal_dff(:,1:length(green_ROI_column))*1000,'-');
    set(h,'color',bg_fill_color);
end
if ~isempty(red_ROI_column)
    h=plot(FP_415_time,FP_415_signal_dff(:,[1:length(red_ROI_column)]+length(green_ROI_column))*1000,'-');
end
th=text(FP_415_time(round(length(FP_415_time)*.9)),max(FP_415_signal_dff(:))*500,'415 nm df/f *1000');
set(th,'color',[.03 .03 .03]);
%%
% the C1poke that triggers the trial
C1poke_first_in_event_LeftTrial=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([12 8 13],n_trial_events,1))==0,2));
C1poke_first_in_event_LRnoRwd_R_Trial=find(prod((exper.rpbox.param.trial_events.value(:,3:5)-repmat([2 8 3],n_trial_events,1))==0,2));
C1poke_first_in_time_idx=sort([C1poke_first_in_event_LeftTrial;C1poke_first_in_event_LRnoRwd_R_Trial]);

C1poke_first_in_time=trial_event_FP_time(sort([C1poke_first_in_event_LeftTrial;C1poke_first_in_event_LRnoRwd_R_Trial]))-exper.odor_2afc.param.minodordelay.value;
ignore_trials=find(C1poke_first_in_time<FPTS(find(~isnan(FP_470_signal_dff),1,'first')));

counted_trial=exper.odor_2afc.param.countedtrial.value;
full_schedule=exper.odor_2afc.param.schedule.value(1:counted_trial(end));
full_result=exper.odor_2afc.param.result.value(1:counted_trial(end));
full_odorpokedur=exper.odor_2afc.param.odorpokedur.value(1:counted_trial(end));
full_portside=exper.odor_2afc.param.port_side.value(1:counted_trial(end));
full_portside(ignore_trials)=NaN;
result=full_result;
result(result==1)=0;
if sum(result==1)>0,disp('non-rewarded correct trial found');end
result=floor(result);
presumed_Port_Side=full_portside*0;
for in=1:length(full_portside)
    if full_portside(in)>0
        presumed_Port_Side(in)=full_portside(in);
    elseif full_portside(in)==0 && in>1
        presumed_Port_Side(in)=presumed_Port_Side(in-1);
    end
end
choosen_Port_Side=full_portside*0;
choosen_Port_Side(result==1)=presumed_Port_Side(result==1);
choosen_Port_Side(result==2)=3-presumed_Port_Side(result==2); %animal chose wrong side,port_side[1,2]==>choosen_port_side[2,1]
choosen_Port_Side(ignore_trials)=0;
prev_choosen_Port_Side=[0 choosen_Port_Side(1:end-1)];

C1poke_first_in_time_LeftTrial=trial_event_FP_time(C1poke_first_in_event_LeftTrial)-exper.odor_2afc.param.minodordelay.value;
C1poke_first_in_time_LRnoRwd_R_Trial=trial_event_FP_time(C1poke_first_in_event_LRnoRwd_R_Trial)-exper.odor_2afc.param.minodordelay.value;
C1poke_first_in_LRnoRwd_R_Trial_idx=find(ismember(full_portside,[0 1]));
C1poke_first_in_event_RightTrial=C1poke_first_in_event_LRnoRwd_R_Trial(full_portside(C1poke_first_in_LRnoRwd_R_Trial_idx)==1);
C1poke_first_in_time_RightTrial=trial_event_FP_time(C1poke_first_in_event_RightTrial)-exper.odor_2afc.param.minodordelay.value;
C1poke_first_in_event_LnoRwd=C1poke_first_in_event_LRnoRwd_R_Trial(full_portside(C1poke_first_in_LRnoRwd_R_Trial_idx)==0 & full_schedule(C1poke_first_in_LRnoRwd_R_Trial_idx)==9);
C1poke_first_in_time_LnoRwd=trial_event_FP_time(C1poke_first_in_event_LnoRwd)-exper.odor_2afc.param.minodordelay.value;
C1poke_first_in_event_RnoRwd=C1poke_first_in_event_LRnoRwd_R_Trial(full_portside(C1poke_first_in_LRnoRwd_R_Trial_idx)==0 & full_schedule(C1poke_first_in_LRnoRwd_R_Trial_idx)==11);
C1poke_first_in_time_RnoRwd=trial_event_FP_time(C1poke_first_in_event_RnoRwd)-exper.odor_2afc.param.minodordelay.value;

% all C1pokes
C1poke_in_time=trial_event_FP_time((exper.rpbox.param.trial_events.value(:,4)-repmat([1],n_trial_events,1))==0);
L1poke_in_time=trial_event_FP_time((exper.rpbox.param.trial_events.value(:,4)-repmat([3],n_trial_events,1))==0);
L1poke_out_time=trial_event_FP_time((exper.rpbox.param.trial_events.value(:,4)-repmat([4],n_trial_events,1))==0);
R1poke_in_time=trial_event_FP_time((exper.rpbox.param.trial_events.value(:,4)-repmat([5],n_trial_events,1))==0);
R1poke_out_time=trial_event_FP_time((exper.rpbox.param.trial_events.value(:,4)-repmat([6],n_trial_events,1))==0);

% check drift in time between computers
matched_LV1of2_on_time=LV1of2_on_time(ismember(LV2_on_time,matched_LV2_on_time));
matched_Expert_LV1_on_time=Expert_LV1_on_time(ismember(Expert_LV2_on_time,matched_Expert_LV2_on_time));
plot(matched_LV1of2_on_time,matched_LV1of2_on_time-matched_Expert_LV1_on_time'*1000-temp(1),'bd');hold on;
plot(C1poke_first_in_time_LeftTrial,C1poke_first_in_time_LeftTrial-exper.rpbox.param.trial_events.value(C1poke_first_in_event_LeftTrial,2)*1000-temp(1),'go');hold on;shg;

plot(C1poke_first_in_time_LeftTrial,-25,'go');hold on;
plot(Expert_LV1_on_time'*1000+temp(1),-25,'bd');
plot(C1poke_first_in_time_LRnoRwd_R_Trial,-25,'co');hold on;
RV1_on_time=trial_event_FP_time(Expert_RV1_on_event);
plot(RV1_on_time,-25,'cd');
% plot(CLed_on_time-temp(1),-20,'ys');shg

% deal with discarded final few trials if counted_trial < total trial
last_counted_trial_event_time=trial_event_FP_time(find(exper.rpbox.param.trial_events.value(:,2)==exper.odor_2afc.param.trial_events.trial{counted_trial}(1,3)))+1000;
LV1of2_on_time2=trial_event_FP_time(Expert_LV1_on_event);
LV1of2_on_time2=LV1of2_on_time2(LV1of2_on_time2<last_counted_trial_event_time & LV1of2_on_time2>FPTS(1));
RV1_on_time=RV1_on_time(RV1_on_time<last_counted_trial_event_time & RV1_on_time>FPTS(1));

% sanity check LV1_on_time (in ms) is same length as rewarded_L_trial
Left_trial=(full_portside(1:counted_trial(end))==2);
rewarded_L_trial=full_result==1.2 & Left_trial;
unrewarded_L_trial=full_result~=1.2 & choosen_Port_Side==2;
H2O_Valve_on_time_RwdTrial=zeros(size(Left_trial)).*NaN;
if sum(rewarded_L_trial)~=length(LV1of2_on_time2)
    disp('rewarded left trials does not match LV_on_time, check there are ignored trials');
else
    H2O_Valve_on_time_RwdTrial(rewarded_L_trial)=LV1of2_on_time2;
end

Right_trial=(full_portside(1:counted_trial(end))==1);
rewarded_R_trial=full_result==1.2 & Right_trial;
unrewarded_R_trial=full_result~=1.2 & choosen_Port_Side==1;
if sum(rewarded_R_trial)~=length(RV1_on_time)
    disp('rewarded right trials does not match LV_on_time, check there are ignored trials');
else
    H2O_Valve_on_time_RwdTrial(rewarded_R_trial)=RV1_on_time;
end

%%
fig=figure(8346);clf
set(fig,'Name',thisfile,'filename',[thisfile '.fig']);
for s=1:8 %go through rewarded/un-rewarded trials amd LL/LR/RL/RR choices history.
    if s==1
        previous_trial_idx=prev_choosen_Port_Side==2;
        previous_Sideport_out_time=L1poke_out_time;
        current_trial_idx=rewarded_L_trial;
        Sideport1_str='L';
        Sideport2_str='L';
        subplot_shift=0;
        outcome_event_str='Water Valve on';
    elseif s==2
        previous_trial_idx=prev_choosen_Port_Side==1;
        previous_Sideport_out_time=R1poke_out_time;
        current_trial_idx=rewarded_R_trial;
        Sideport1_str='R';
        Sideport2_str='R';
        subplot_shift=3;
        outcome_event_str='Water Valve on';
    elseif s==3
        previous_trial_idx=prev_choosen_Port_Side==2; %L
        current_trial_idx=rewarded_R_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='R';
        subplot_shift=6;
        outcome_event_str='Water Valve on';
    elseif s==4
        previous_trial_idx=prev_choosen_Port_Side==1;
        previous_Sideport_out_time=R1poke_out_time;
        current_trial_idx=rewarded_L_trial;
        Sideport1_str='R';
        Sideport2_str='L';
        subplot_shift=9;
        outcome_event_str='Water Valve on';
    elseif s==5
        previous_trial_idx=prev_choosen_Port_Side==2;
        previous_Sideport_out_time=L1poke_out_time;
        current_trial_idx=unrewarded_L_trial;
        Sideport1_str='L';
        Sideport2_str='L';
        subplot_shift=12;
        outcome_event_str='Time-out LED on';
    elseif s==6
        previous_trial_idx=prev_choosen_Port_Side==1;
        previous_Sideport_out_time=R1poke_out_time;
        current_trial_idx=unrewarded_R_trial;
        Sideport1_str='R';
        Sideport2_str='R';
        subplot_shift=15;
        outcome_event_str='Time-out LED on';
    elseif s==7
        previous_trial_idx=prev_choosen_Port_Side==2;
        previous_Sideport_out_time=L1poke_out_time;
        current_trial_idx=unrewarded_R_trial;
        Sideport1_str='L';
        Sideport2_str='R';
        subplot_shift=18;
        outcome_event_str='Time-out LED on';
    elseif s==8
        previous_trial_idx=prev_choosen_Port_Side==1;
        previous_Sideport_out_time=R1poke_out_time;
        current_trial_idx=unrewarded_L_trial;
        Sideport1_str='R';
        Sideport2_str='L';
        subplot_shift=21;
        outcome_event_str='Time-out LED on';
    end

    subplot(4,6,1+subplot_shift);cla;hold on;
    plot_trial_idx=find(previous_trial_idx & current_trial_idx); %find rewarded Left (correct) choice previously also went left
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time=C1poke_first_in_time(plot_trial_idx(i));
        % find last side_out_time before Center in
        Sideport_out_time_window=previous_Sideport_out_time(previous_Sideport_out_time>(aligned_event_time+time_window(1)) & previous_Sideport_out_time<(aligned_event_time));
        if  ~isempty(Sideport_out_time_window)
            aligned_event_time=Sideport_out_time_window(end);
            L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
            L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(L1poke_in_time_window)&& ~isempty(L1poke_out_time_window)
                if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                end
            end
            R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
            R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(R1poke_in_time_window) && ~isempty(R1poke_out_time_window)
                if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                end
            end
            C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
            C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
            if ~isempty(C1poke_first_in_time_window)
                plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
            end
            plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
            FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
            FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
            if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
                resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
                resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
            else
                disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
            end
            plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
        end
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
    set(h,'EdgeColor','none');
    FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
    set(h,'EdgeColor',[1 1 1]);
    plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
    axis(ax);
    if subplot_shift==0
        title_header=[Hemi_str ', total: ' num2str(counted_trial(end)) 'trials, '];
    else
        title_header=[];
    end
    title([title_header Sideport1_str '-port Out' sprintf('\n') ' ' Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
    xlabel('mSec');
    ylabel('FP signal (a.u.)');

    % find rewarded Left choice previously also went left (correct)
    subplot(4,6,2+subplot_shift);cla;hold on;
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time=C1poke_first_in_time(plot_trial_idx(i));
        L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
        L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
        if ~isempty(L1poke_in_time_window)&& ~isempty(L1poke_out_time_window)
            if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
            elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
            end
        end
        R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
        R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
        if ~isempty(R1poke_in_time_window)&& ~isempty(R1poke_out_time_window)
            if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
            elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
            end
        end
        C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
        C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
        if ~isempty(C1poke_first_in_time_window)
            plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
        end
        plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
        FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
        FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
        if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
            resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
            resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
        else
            disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
        end
        plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
    set(h,'EdgeColor','none');
    FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
    set(h,'EdgeColor',[1 1 1]);
    plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
    axis(ax);
    title(['Center port IN' sprintf('\n') Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
    xlabel('mSec');
    ylabel('FP signal (a.u.)');

    % find rewarded Left choice previously also went left (correct)
    subplot(4,6,3+subplot_shift);cla;hold on;
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time1=C1poke_first_in_time(plot_trial_idx(i));
        aligned_event_time=[];
        if full_result(plot_trial_idx(i))==1.2
            % find first H2O_Valve_on_time after Center in
            H2O_Valve_on_time_RwdTrial_window=H2O_Valve_on_time_RwdTrial(H2O_Valve_on_time_RwdTrial>aligned_event_time1 & H2O_Valve_on_time_RwdTrial<(aligned_event_time1+time_window(end)));
            if  ~isempty(H2O_Valve_on_time_RwdTrial_window)
                aligned_event_time=H2O_Valve_on_time_RwdTrial_window(1);
            end
        elseif full_result(plot_trial_idx(i))==2
            tt2=data.exper.odor_2afc.param.trial_events.trial{plot_trial_idx(i)}(:,3);
            if  ~isempty(tt2)
                aligned_event_time=trial_event_FP_time(exper.rpbox.param.trial_events.value(:,2)==tt2(1));
            end
        end
        if  ~isempty(aligned_event_time)
            L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
            L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(L1poke_in_time_window) && ~isempty(L1poke_out_time_window)
                if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                end
            end
            R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
            R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(R1poke_in_time_window) && ~isempty(R1poke_out_time_window)
                if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                end
            end
            C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
            C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
            if ~isempty(C1poke_first_in_time_window)
                plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
            end
            plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
            FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
            FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
            if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
                resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
                resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
            else
                disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
            end
            plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
        end
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    if ~isempty(resampled_FP_470_signal_dff_segmmtI)
        FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
        FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
        h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
        set(h,'EdgeColor','none');
        FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
        FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
        h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
        set(h,'EdgeColor',[1 1 1]);
        plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
        axis(ax);
        title([outcome_event_str sprintf('\n') Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
        xlabel('mSec');
        ylabel('FP signal (a.u.)');
    end
end

shg;

%%
prev_rewarded_L_trial=[0 rewarded_L_trial(1:end-1)];
prev_rewarded_L_trial_2back=[0 0 rewarded_L_trial(1:end-2)];
prev_rewarded_L_trial_3back=[0 0 0 rewarded_L_trial(1:end-3)];
prev_unrewarded_L_trial=[0 unrewarded_L_trial(1:end-1)];
prev_unrewarded_L_trial_2back=[0 0 unrewarded_L_trial(1:end-2)];
prev_unrewarded_L_trial_3back=[0 0 0 unrewarded_L_trial(1:end-3)];

prev_rewarded_R_trial=[0 rewarded_R_trial(1:end-1)];
prev_rewarded_R_trial_2back=[0 0 rewarded_R_trial(1:end-2)];
prev_rewarded_R_trial_3back=[0 0 0 rewarded_R_trial(1:end-3)];
prev_unrewarded_R_trial=[0 unrewarded_R_trial(1:end-1)];
prev_unrewarded_R_trial_2back=[0 0 unrewarded_R_trial(1:end-2)];
prev_unrewarded_R_trial_3back=[0 0 0 unrewarded_R_trial(1:end-3)];

%%
fig=figure(8347);clf
set(fig,'Name',thisfile,'filename' ,[thisfile '_' Hemi_str '.fig']);
for s=1:8 %go through rewarded/un-rewarded trials amd LL/LR/RL/RR choices history.
    if s==1
        previous_trial_idx=prev_rewarded_L_trial &prev_rewarded_L_trial_2back;
        current_trial_idx=rewarded_L_trial | unrewarded_L_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='L';
        subplot_shift=0;
        outcome_event_str='RR==>Both Outcome';
    elseif s==2
        previous_trial_idx=prev_rewarded_L_trial &prev_rewarded_L_trial_2back;
        current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='R';
        subplot_shift=3;
        outcome_event_str='RR==>Both Outcome';
    elseif s==3
        previous_trial_idx=prev_unrewarded_L_trial &prev_rewarded_L_trial_2back;
        current_trial_idx=rewarded_L_trial | unrewarded_L_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='L';
        subplot_shift=6;
        outcome_event_str='RU==>Both Outcome';
    elseif s==4
        previous_trial_idx=prev_unrewarded_L_trial &prev_rewarded_L_trial_2back;
        current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='R';
        subplot_shift=9;
        outcome_event_str='RU==>Both Outcome';
%         previous_trial_idx=prev_unrewarded_R_trial &prev_rewarded_R_trial_2back &prev_rewarded_R_trial_3back;
%         previous_Sideport_out_time=R1poke_out_time;
%         current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
%         Sideport1_str='R';
%         Sideport2_str='R';
%         subplot_shift=9;
%         outcome_event_str='RRU==>Both Outcome';
    elseif s==5
        previous_trial_idx=prev_unrewarded_L_trial &prev_unrewarded_L_trial_2back;
        current_trial_idx=rewarded_L_trial | unrewarded_L_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='L';
        subplot_shift=12;
        outcome_event_str='UU==>Both Outcome';
    elseif s==6
        previous_trial_idx=prev_unrewarded_L_trial &prev_unrewarded_L_trial_2back;
        current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
        previous_Sideport_out_time=L1poke_out_time;
        Sideport1_str='L';
        Sideport2_str='R';
        subplot_shift=15;
        outcome_event_str='UU==>Both Outcome';
%         previous_trial_idx=prev_unrewarded_R_trial &prev_unrewarded_R_trial_2back &prev_rewarded_R_trial_3back;
%         previous_Sideport_out_time=R1poke_out_time;
%         current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
%         Sideport1_str='R';
%         Sideport2_str='R';
%         subplot_shift=15;
%         outcome_event_str='RUU==>Both Outcome';
    elseif s==7
        previous_trial_idx=prev_rewarded_R_trial &prev_rewarded_R_trial_2back ;
        previous_Sideport_out_time=R1poke_out_time;
        current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
        Sideport1_str='R';
        Sideport2_str='R';
        subplot_shift=18;
        outcome_event_str='RR==>Both Outcome';
    elseif s==8
        previous_trial_idx=prev_unrewarded_R_trial &prev_rewarded_R_trial_2back;
        current_trial_idx=rewarded_R_trial | unrewarded_R_trial;
        previous_Sideport_out_time=R1poke_out_time;
        Sideport1_str='R';
        Sideport2_str='R';
        subplot_shift=21;
        outcome_event_str='RU==>Both Outcome';
    end

    subplot(4,6,1+subplot_shift);cla;hold on;
    plot_trial_idx=find(previous_trial_idx & current_trial_idx); %find rewarded Left (correct) choice previously also went left
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time=C1poke_first_in_time(plot_trial_idx(i));
        % find last side_out_time before Center in
        Sideport_out_time_window=previous_Sideport_out_time(previous_Sideport_out_time>(aligned_event_time+time_window(1)) & previous_Sideport_out_time<(aligned_event_time));
        if  ~isempty(Sideport_out_time_window)
            aligned_event_time=Sideport_out_time_window(end);
            L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
            L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(L1poke_in_time_window)&& ~isempty(L1poke_out_time_window)
                if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                end
            end
            R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
            R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(R1poke_in_time_window) && ~isempty(R1poke_out_time_window)
                if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                end
            end
            C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
            C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
            if ~isempty(C1poke_first_in_time_window)
                plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
            end
            plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
            FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
            FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
            if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
                resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
                resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
            else
                disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
            end
            plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
        end
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
    set(h,'EdgeColor','none');
    FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
    set(h,'EdgeColor',[1 1 1]);
    plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
    axis(ax);
    if subplot_shift==0
        title_header=[Hemi_str ', total: ' num2str(counted_trial(end)) 'trials, '];
    else
        title_header=[];
    end
    title([title_header Sideport1_str '-port Out' sprintf('\n') ' ' Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
    xlabel('mSec');
    ylabel('FP signal (a.u.)');

    % find rewarded Left choice previously also went left (correct)
    subplot(4,6,2+subplot_shift);cla;hold on;
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time=C1poke_first_in_time(plot_trial_idx(i));
        L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
        L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
        if ~isempty(L1poke_in_time_window)&& ~isempty(L1poke_out_time_window)
            if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
            elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
            end
        end
        R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
        R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
        if ~isempty(R1poke_in_time_window)&& ~isempty(R1poke_out_time_window)
            if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
            elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
            end
        end
        C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
        C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
        if ~isempty(C1poke_first_in_time_window)
            plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
        end
        plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
        FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
        FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
        if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
            resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
            resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
        else
            disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
        end
        plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
    set(h,'EdgeColor','none');
    FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
    h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
    set(h,'EdgeColor',[1 1 1]);
    plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
    axis(ax);
    title(['Center port IN' sprintf('\n') Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
    xlabel('mSec');
    ylabel('FP signal (a.u.)');

    % find rewarded Left choice previously also went left (correct)
    subplot(4,6,3+subplot_shift);cla;hold on;
    resampled_FP_470_signal_dff_segmmtI=[];
    resampled_FP_415_signal_dff_segmmtI=[];
    reply='';
    plot_each_trace=0;
    for i=1:length(plot_trial_idx)
        aligned_event_time1=C1poke_first_in_time(plot_trial_idx(i));
        aligned_event_time=[];
        if full_result(plot_trial_idx(i))==1.2
            % find first H2O_Valve_on_time after Center in
            H2O_Valve_on_time_RwdTrial_window=H2O_Valve_on_time_RwdTrial(H2O_Valve_on_time_RwdTrial>aligned_event_time1 & H2O_Valve_on_time_RwdTrial<(aligned_event_time1+time_window(end)));
            if  ~isempty(H2O_Valve_on_time_RwdTrial_window)
                aligned_event_time=H2O_Valve_on_time_RwdTrial_window(1);
            end
        elseif full_result(plot_trial_idx(i))==2
            tt2=data.exper.odor_2afc.param.trial_events.trial{plot_trial_idx(i)}(:,3);
            if  ~isempty(tt2)
                aligned_event_time=trial_event_FP_time(exper.rpbox.param.trial_events.value(:,2)==tt2(1));
            end
        end
        if  ~isempty(aligned_event_time)
            L1poke_in_time_window=L1poke_in_time(L1poke_in_time>(aligned_event_time+time_window(1)) & L1poke_in_time<(aligned_event_time+time_window(end)));
            L1poke_out_time_window=L1poke_out_time(L1poke_out_time>(aligned_event_time+time_window(1)) & L1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(L1poke_in_time_window) && ~isempty(L1poke_out_time_window)
                if L1poke_in_time_window(1)<L1poke_out_time_window(1)
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window))  L1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                elseif L1poke_in_time_window(1)>L1poke_out_time_window(1)&& length(L1poke_out_time_window)>1
                    plot([L1poke_in_time_window(1:length(L1poke_out_time_window)-1)  L1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'g-');
                end
            end
            R1poke_in_time_window=R1poke_in_time(R1poke_in_time>(aligned_event_time+time_window(1)) & R1poke_in_time<(aligned_event_time+time_window(end)));
            R1poke_out_time_window=R1poke_out_time(R1poke_out_time>(aligned_event_time+time_window(1)) & R1poke_out_time<(aligned_event_time+time_window(end)));
            if ~isempty(R1poke_in_time_window) && ~isempty(R1poke_out_time_window)
                if R1poke_in_time_window(1)<R1poke_out_time_window(1)
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window))  R1poke_out_time_window]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                elseif R1poke_in_time_window(1)>R1poke_out_time_window(1)&& length(R1poke_out_time_window)>1
                    plot([R1poke_in_time_window(1:length(R1poke_out_time_window)-1)  R1poke_out_time_window(2:end)]'-aligned_event_time,[i*ax(3)/length(plot_trial_idx) i*ax(3)/length(plot_trial_idx)],'m-');
                end
            end
            C1poke_first_in_time_window=C1poke_first_in_time(C1poke_first_in_time>(aligned_event_time+time_window(1)) & C1poke_first_in_time<(aligned_event_time+time_window(end)));
            C1poke_first_in_time_window=C1poke_first_in_time_window(~ismember(C1poke_first_in_time_window,C1poke_first_in_time(plot_trial_idx(i))));
            if ~isempty(C1poke_first_in_time_window)
                plot(C1poke_first_in_time_window-aligned_event_time,i*ax(3)/length(plot_trial_idx),'k.');
            end
            plot(C1poke_first_in_time(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'r.');
            FP_470_segmntI=find(FP_470_time>(aligned_event_time+time_window(1)-50) & FP_470_time<(aligned_event_time+time_window(end)+50));
            FP_415_segmntI=find(FP_415_time>(aligned_event_time+time_window(1)-50) & FP_415_time<(aligned_event_time+time_window(end)+50));
            if ~isempty(FP_470_segmntI) && length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))>0.425
                resampled_FP_470_signal_dff_segmmtI=[resampled_FP_470_signal_dff_segmmtI; interp1(FP_470_time(FP_470_segmntI)-aligned_event_time,FP_470_signal_dff((FP_470_segmntI)),time_window,'linear','extrap')];
                resampled_FP_415_signal_dff_segmmtI=[resampled_FP_415_signal_dff_segmmtI; interp1(FP_415_time(FP_415_segmntI)-aligned_event_time,FP_415_signal_dff((FP_415_segmntI)),time_window,'linear','extrap')];
            else
                disp(['dropped ' num2str(round((1-length(FP_470_segmntI)*frame_duration/diff(time_window([1 end]))*2)*1000)/10) '% frame in subplot ' num2str(3+subplot_shift) ' trace ' num2str(i)]);
            end
            plot(H2O_Valve_on_time_RwdTrial(plot_trial_idx(i))-aligned_event_time,i*ax(3)/length(plot_trial_idx),'b.');
        end
    end
    plot([0 0],[-1 4],'k:');
    % errorbar(time_window,mean(resampled_FP_470_signal_dff_segmmtI,1),std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(correct_LL_trial)-1),'b-');
    if ~isempty(resampled_FP_470_signal_dff_segmmtI)
        FP_Y1=mean(resampled_FP_415_signal_dff_segmmtI,1)+std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
        FP_Y2=mean(resampled_FP_415_signal_dff_segmmtI,1)-std(resampled_FP_415_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_415_signal_dff_segmmtI(:,1:10),1));
        h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], bg_fill_color);hold on;
        set(h,'EdgeColor','none');
        FP_Y1=mean(resampled_FP_470_signal_dff_segmmtI,1)+std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
        FP_Y2=mean(resampled_FP_470_signal_dff_segmmtI,1)-std(resampled_FP_470_signal_dff_segmmtI,1)./sqrt(length(plot_trial_idx)-1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1));
        h=fill( [time_window fliplr(time_window)],  [FP_Y1 fliplr(FP_Y2)], 'c');hold on;
        set(h,'EdgeColor',[1 1 1]);
        plot(time_window,(mean(resampled_FP_470_signal_dff_segmmtI,1)-mean(mean(resampled_FP_470_signal_dff_segmmtI(:,1:10),1))),'b-');
        axis(ax);
        title([outcome_event_str sprintf('\n') Sideport1_str '==>C1 in==>' Sideport2_str ',n=' num2str(size(resampled_FP_470_signal_dff_segmmtI,1)) ' trials' ]);
        xlabel('mSec');
        ylabel('FP signal (a.u.)');
    end
end

shg;