
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
