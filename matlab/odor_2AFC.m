function out = odor_2AFC(varargin)

global exper fake_rp_box inputevents output_routing OneChSound_ID OneChSound_side vp_sound
global right1led left1led right1water left1water

if nargin > 0
    action = lower(varargin{1});
else
    action = lower(get(gcbo,'tag'));
end
warning off 'MATLAB:divideByZero'

out=1;
switch action

    case 'restore_event'
        if nargin >1
            restore_event(varargin{2});
        else
            restore_event;
        end

    case 'init'
        ModuleNeeds(me,{'rpbox'});

        SetParam(me,'priority','value',GetParam('rpbox','priority')+1);
        fig = ModuleFigure(me,'visible','off');

        hs = 100;
        h = 5;
        vs = 20;
        n = 0;

        param_string={'Odor Name','Dout Channel','left reward ratio','right reward ratio','VP LED cue','stimulus duration','stimulus probability','stimulus name'};
        StimParam={'A', '6',  '1',  '0',  '0','0.1','0.5','IDS_A';...
            'B', '7',  '0',  '1',  '0','0.1','0.5','IDS_B';...
            'C', '8',  '1',  '0',  '0','0.1',  '0','IDS_C';...
            'D', '9',  '0',  '1',  '0','0.1',  '0','IDS_D';...
            'A', '6',  '0',  '1',  '0','0.1',  '0','IDR_A';...
            'B', '7',  '1',  '0',  '0','0.1',  '0','IDR_B';...
            'C', '8',  '0',  '1',  '0','0.1',  '0','IDR_C';...
            'D', '9',  '1',  '0',  '0','0.1',  '0','IDR_D';...
            'L', '0',  '1',  '0',  '0',  '0',  '0','L_no_odor';...
            'L', '0',  '1',  '0',  '1',  '0',  '0','L_no_odor';...
            'R', '0',  '0',  '1',  '0',  '0',  '0','R_no_odor';...
            'R', '0',  '0',  '1',  '1',  '0',  '0','R_no_odor';};
        InitParam(me,'StimParam','value',StimParam,'user',param_string);
        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';

        JntGrp={[ 1  3 5;   % For ploting joint performance in the same column, ie, mean of tone 1 and 10 is ploted in a group.
            2  4 6],[ 9;10 ] };    % Additional group lists can be added in the cell array.

        JntGrpXTitle={'IDS_AB' 'IDS_CD' 'IDS_EF' 'IDR_CD'};
        ScriptList={'none','LR_100%Cue_LR_&_switch','RL_100%Cue_LR_&_switch','LR_100%Cue_switch_only','RL_100%Cue_switch_only',...
            'LR_switch_rewards_blocks','RL_switch_rewards_blocks','LR_6-8_switch_No_H2O','RL_6-8_switch_No_H2O','LR_switch_LnRn_to_L1R1',...
            'RL_switch_LnRn_to_L1R1','LR_switch_L1R1_to_LnRn','RL_switch_L1R1_to_LnRn','LR_20switch_3vs2H2O','RL_20switch_3vs2H2O','extinction@400',...
            'BG_switch_rewards_blocks'};
        InitParam(me,'update_plot_flag','value',0);
        InitParam(me,'n_LR_switch','value',0);
        InitParam(me,'Hit_Streak','value',0);
        InitParam(me,'MaxTrial','value',2501);

        n=n+.2;
        InitParam(me,'TotalScore','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'TotalScore','label','Total Score','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'LeftScore','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftScore','label','Left Score','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightScore','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightScore','label','Right Score','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'rWaterValveDur','ui','edit','value',.076,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'rWaterValveDur','label','R_WaterV_Dur','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'BlockLength','ui','edit','value',50,'pos',[h+hs*5.13 n*vs hs*.2 vs]);
        SetParamUI(me,'BlockLength','label','Block+-','labelpos',[0 0 hs*.2 0]);
        InitParam(me,'BlockLength_jitter','ui','edit','value',0,'pos',[h+hs*5.85 n*vs hs*.2 vs]);
        SetParamUI(me,'BlockLength_jitter','label','length','labelpos',[0 0 hs*.12 0]);
        InitParam(me,'DirectDelivery','ui','edit','value',0,'pos',[h+hs*6.73 n*vs hs*.36 vs]);
        SetParamUI(me,'DirectDelivery','label','Direct Delivery Trials','labelpos',[0 0 hs*.74 0]);
        InitParam(me,'n_Short_Block','ui','edit','value',200,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'n_Short_Block','label','n_Short Block','labelpos',[0 0 hs*.45 0]);

        n=n+1;
        InitParam(me,'RecentScore','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'RecentScore','label','Recent Score =>','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'rLeftScore','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'rLeftScore','label','rLeftScore','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'rRightScore','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'rRightScore','label','rRightScore','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'lWaterValveDur','ui','edit','value',.082,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'lWaterValveDur','label','L_WaterV_Dur','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'DelayOdor','ui','checkbox','value',1,'pref',0,'pos',[h+hs*5.13 n*vs hs*.75 vs]);
        SetParamUI(me,'DelayOdor','label','','string','Delay Odor','labelpos',[-10 0 -hs*.74 0]);
        InitParam(me,'Deliver_Full_Stim','ui','checkbox','value',0,'pref',0,'pos',[h+hs*5.86 n*vs hs*.79 vs]);
        SetParamUI(me,'Deliver_Full_Stim','label','','string','Full odor dur','labelpos',[-10 0 -hs*.78 0]);
        InitParam(me,'GracePokeOutTime','ui','edit','value',150,'pos',[h+hs*6.73 n*vs hs*.36 vs]);
        SetParamUI(me,'GracePokeOutTime','label','Grace PO time (ms)','labelpos',[0 0 hs*.74 0]);
        InitParam(me,'n_Long_Block','ui','edit','value',10,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'n_Long_Block','label','n_Long Block','labelpos',[0 0 hs*.45 0]);

        n=n+1;
        InitParam(me,'ValidScore','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'ValidScore','label','Valid Score','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'LeftHit','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftHit','label','Left Hit','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightHit','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightHit','label','Right Hit','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'ValidOdorPokeDur','ui','edit','value',0.05,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'ValidOdorPokeDur','label','ValidPokeDur(s)','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'OdorDelaySchedule','ui','disp','value',0,'pref',0,'user',[],'pos',[h+hs*5.13 n*vs hs*.36 vs]);
        SetParamUI(me,'OdorDelaySchedule','label','Odor Delay (ms)','labelpos',[0 0 hs*.64 0]);
        InitParam(me,'RewardDelaySchedule','ui','disp','value',0,'pref',0,'user',[],'pos',[h+hs*6.73 n*vs hs*.36 vs]);
        SetParamUI(me,'RewardDelaySchedule','label','Reward Delay (ms)','labelpos',[0 0 hs*.74 0]);
        InitParam(me,'n_Cued_Short_Block','ui','edit','value',10,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'n_Cued_Short_Block','label','n_Cued short block','labelpos',[0 0 hs*.67 0]);

        n=n+1;
        InitParam(me,'LastOdorPokeDur','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'LastOdorPokeDur','label','Last TnPokeDur','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'LeftMiss','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftMiss','label','Left Miss','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightMiss','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightMiss','label','Right Miss','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'WaterAvailDur','ui','edit','value',3,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'WaterAvailDur','label','WaterAvailDur(s','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'maxOdorDelay','ui','edit','value',100,'pos',[h+hs*5.13 n*vs hs*.36 vs]);
        SetParamUI(me,'maxOdorDelay','label','maxOdorDelay (ms)','labelpos',[0 0 hs*.64 0]);
        InitParam(me,'maxRewardDelay','ui','edit','value',150,'pos',[h+hs*6.73 n*vs hs*.36 vs]);
        SetParamUI(me,'maxRewardDelay','label','maxRewardDelay(ms)','labelpos',[0 0 hs*.74 0]);
        InitParam(me,'n_Cued_LongBlock_Trial','ui','edit','value',10,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'n_Cued_LongBlock_Trial','label','n_Cued LB trial','labelpos',[0 0 hs*.65 0]);

        n=n+1;
        InitParam(me,'FirstOdorPokeDur','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'FirstOdorPokeDur','label','C1PokeDur (ms)','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'LeftFalse','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftFalse','label','Left False','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightFalse','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightFalse','label','Right False','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RecentHistory','ui','edit','value',20,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'RecentHistory','label','Recent History','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'minOdorDelay','ui','edit','value',50,'pos',[h+hs*5.13 n*vs hs*.36 vs]);
        SetParamUI(me,'minOdorDelay','label','minOdorDelay (ms)','labelpos',[0 0 hs*.64 0]);
        InitParam(me,'minRewardDelay','ui','edit','value',50,'pos',[h+hs*6.73 n*vs hs*.36 vs]);
        SetParamUI(me,'minRewardDelay','label','minRewardDelay (ms)','labelpos',[0 0 hs*.74 0]);
        InitParam(me,'p_Cued_Switch','ui','edit','value',100,'user',0,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'p_Cued_Switch','label','% Cued Switch','labelpos',[0 0 hs*.65 0]);

        n=n+1;
        InitParam(me,'CountedTrial','ui','disp','value',0,'pref',0,'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'CountedTrial','label','Counted Trial','labelpos',[-5 0 hs*.48 0]);
        InitParam(me,'LeftAbort','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftAbort','label','Left Abort','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightAbort','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightAbort','label','Right Abort','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'ITI','ui','edit','value',.01,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'ITI','label','ITI (sec)','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'UseMinOdorDelay','ui','checkbox','value',1,'pref',0,'pos',[h+hs*5.13 n*vs hs*1.43 vs]);
        SetParamUI(me,'UseMinOdorDelay','label','','string','OdorDelay=minOdorDelay','labelpos',[-10 0 -hs*1.42 0]);
        InitParam(me,'UseMinRewardDelay','ui','checkbox','value',1,'pref',0,'pos',[h+hs*6.73 n*vs hs*1.57 vs]);
        SetParamUI(me,'UseMinRewardDelay','label','','string','RwdDelay=minRewardDelay','labelpos',[-10 0 -hs*1.56 0]);
        InitParam(me,'SameSideLimit','ui','edit','value',3,'pos',[h+hs*8.35 n*vs hs*.25 vs]);
        SetParamUI(me,'SameSideLimit','label','SameSideLimit','labelpos',[0 0 hs*.5 0]);

        n=n+1;
        InitParam(me,'Trial_limit','ui','edit','value',500,'range',[.5 inf],'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'Trial_limit','label','Stop Exp@trial #','background',[1 .7 .7],'labelbackground',[1 .7 .7],'labelpos',[-3 0 hs*.48 0]);
        InitParam(me,'LeftEarlyWithdraw','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'LeftEarlyWithdraw','label','lErlyWthdrw','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'RightEarlyWithdraw','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'RightEarlyWithdraw','label','rErlyWthdrw','labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'IRI','ui','edit','value',1,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'IRI','label','I-Rwd-I (sec)','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'L_H2O_port','ui','popupmenu','list',[0 2 4],'value',2,'pref',1,'pos',[h+hs*5.13 n*vs hs*.33 vs]);
        SetParamUI(me,'L_H2O_port','label','L-H2OPort-R','labelpos',[0 0 hs*.31 0],'labelbackground',[0 1 1]);
        InitParam(me,'R_H2O_port','ui','popupmenu','list',[0 2 4],'value',3,'pref',1,'pos',[h+hs*6.23 n*vs hs*.33 vs]);
        SetParamUI(me,'R_H2O_port','label','','labelpos',[0 0 hs*-.32 0]);
        InitParam(me,'TimeOut','ui','edit','value',4,'pos',[h+hs*6.73 n*vs hs*.35 vs]);
        SetParamUI(me,'TimeOut','label','Timeout(s)','BackgroundColor',[1 1 .5],'labelbackground',[1 1 .5],'labelpos',[0 0 hs*.4 0]);
        InitParam(me,'UseTimeOutLED','ui','checkbox','value',1,'pref',0,'pos',[h+hs*7.79 (n+0.05)*vs hs*1.2 vs]);
        SetParamUI(me,'UseTimeOutLED','label','','string','Timeout LED','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*1.19 0]);
        InitParam(me,'Miss_TimeOutLED','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*8.65 (n+0.05)*vs hs*.425 vs]);
        SetParamUI(me,'Miss_TimeOutLED','label','','string','miss','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.415 0]);
        InitParam(me,'False_TimeOutLED','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*9.06 (n+0.05)*vs hs*.45 vs]);
        SetParamUI(me,'False_TimeOutLED','label','','string','false','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.44 0]);
        InitParam(me,'Abort_TimeOutLED','ui','radiobutton','value',0,'pref',0,'pos',[h+hs*9.5 (n+0.05)*vs hs*0.45 vs]);
        SetParamUI(me,'Abort_TimeOutLED','label','','string','abort','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.44 0]);

        n=n+1;
        InitParam(me,'H2O_limit','ui','edit','value',600,'range',[.5 inf],'pos',[h n*vs hs*.38 vs]);
        SetParamUI(me,'H2O_limit','label','<H2O-->Stop Exp','background',[1 .7 .7],'labelbackground',[1 .7 .7],'labelpos',[-3 0 hs*.48 0]);
        InitParam(me,'leftH2OReward','ui','disp','value',0,'pref',0,'pos',[h+hs*1.37 n*vs hs*.38 vs]);
        SetParamUI(me,'leftH2OReward','label','l_H2O','background',[0 1 1],'labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'DeliverLeftH2O','ui','pushbutton','value',0,'Enable','on','pref',0,'pos',[h+hs*2.12 n*vs hs*.3 vs]);
        SetParamUI(me,'DeliverLeftH2O','String','0','label','','background',[0 .5 1],'labelpos',[-10 0 -hs*.29 0]);
        InitParam(me,'rightH2OReward','ui','disp','value',0,'pref',0,'pos',[h+hs*2.54 n*vs hs*.38 vs]);
        SetParamUI(me,'rightH2OReward','label','r_H2O','background',[0 1 1],'labelpos',[-5 0 hs*.26 0]);
        InitParam(me,'DeliverRightH2O','ui','pushbutton','value',0,'Enable','on','pref',0,'pos',[h+hs*3.28 n*vs hs*.3 vs]);
        SetParamUI(me,'DeliverRightH2O','String','0','label','','background',[0 .5 1],'labelpos',[-10 0 -hs*.29 0]);
        InitParam(me,'DefaultRewardSize','ui','popupmenu','list',[1 2 3],'value',2,'pref',1,'pos',[h+hs*3.7 n*vs hs*.38 vs]);
        SetParamUI(me,'DefaultRewardSize','label','DefaultRwd Size','labelpos',[0 0 hs*.44 0]);
        InitParam(me,'Use_Hit_Streak','ui','checkbox','value',0,'pref',0,'pos',[h+hs*5.13 n*vs hs*.9 vs]);
        SetParamUI(me,'Use_Hit_Streak','label','','string','Use_HitStreak','labelpos',[-10 0 -hs*.89 0]);
        InitParam(me,'Hit_Streak_3x_H2O','ui','checkbox','value',0,'pref',0,'pos',[h+hs*6 n*vs hs*.58 vs]);
        SetParamUI(me,'Hit_Streak_3x_H2O','label','','string','3x_H2O','labelpos',[-10 0 -hs*.57 0]);
        InitParam(me,'AdaptiveTimeOut','ui','checkbox','value',0,'pref',0,'pos',[h+hs*6.73 (n+0.05)*vs hs*1.2 vs]);
        SetParamUI(me,'AdaptiveTimeOut','label','','string','Adaptive Timeout','BackgroundColor',[1 1 .5]);
        InitParam(me,'UseShortTimeOut','ui','checkbox','value',1,'pref',0,'pos',[h+hs*7.79 (n+0.05)*vs hs*1.2 vs]);
        SetParamUI(me,'UseShortTimeOut','label','','string','Short Timeout','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*1.19 0]);
        InitParam(me,'Miss_ShortTimeOut','ui','radiobutton','value',0,'pref',0,'pos',[h+hs*8.65 (n+0.05)*vs hs*.425 vs]);
        SetParamUI(me,'Miss_ShortTimeOut','label','','string','miss','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.415 0]);
        InitParam(me,'False_ShortTimeOut','ui','radiobutton','value',0,'pref',0,'pos',[h+hs*9.06 (n+0.05)*vs hs*.45 vs]);
        SetParamUI(me,'False_ShortTimeOut','label','','string','false','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.44 0]);
        InitParam(me,'Abort_ShortTimeOut','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*9.5 (n+0.05)*vs hs*0.45 vs]);
        SetParamUI(me,'Abort_ShortTimeOut','label','','string','abort','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*.44 0]);

        n=n+1.2;
        InitParam(me,'datapath','value',[fileparts(fileparts(which(me))) filesep 'data']);
        InitParam(me,'LoadMouseSetting','ui','popupmenu','list',Load_Seting_Strs,'value',1,'user',1,'pref',0,'pos',[h n*vs hs*.9 vs]);
        SetParamUI(me,'LoadMouseSetting','label',' load mouse setting','labelpos',[-10 0 hs*.21 0]);
        InitParam(me,'SaveMouseSetting','ui','pushbutton','value',0,'Enable','on','pref',0,'pos',[h+hs*1.86 n*vs hs*.7 vs]);
        SetParamUI(me,'SaveMouseSetting','String','save setting','label','','background',[1 .5 1],'labelpos',[-10 0 -hs*.69 0]);
        InitParam(me,'Script','ui','popupmenu','list',ScriptList,'value',1,'pos',[h+hs*2.63 n*vs hs*1.22 vs]);
        SetParamUI(me,'Script','label',' Run custom Script','labelpos',[-10 0 -hs*.25 0]);
        InitParam(me,'Miss_Correction','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*4.87 n*vs hs*.425 vs]);
        SetParamUI(me,'Miss_Correction','label','','string','miss');
        InitParam(me,'False_Correction','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*5.3 n*vs hs*.45 vs]);
        SetParamUI(me,'False_Correction','label','','string','false');
        InitParam(me,'Abort_Correction','ui','radiobutton','value',1,'pref',0,'pos',[h+hs*5.75 n*vs hs*1.4 vs]);
        SetParamUI(me,'Abort_Correction','label','','string','abort Correction in next','labelpos',[-10 0 -hs*1 0]);
        InitParam(me,'CorrectionTrial','ui','edit','value',5,'pref',0,'range',[1 10],'pos',[h+hs*7.03 n*vs hs*.25 vs]);
        SetParamUI(me,'CorrectionTrial','label','trials','labelpos',[-5 -2 0 0]);
        InitParam(me,'ClearScore','ui','checkbox','value',1,'pref',0,'pos',[h+hs*7.65 n*vs hs*1.4 vs]);
        SetParamUI(me,'ClearScore','label','','string','Clear Score when Reset','labelpos',[-10 0 -hs*1.39 0]);
        InitParam(me,'TrackAbort','ui','checkbox','value',0,'pref',0,'pos',[h+hs*9.2 n*vs hs*0.75 vs]);
        SetParamUI(me,'TrackAbort','label','','string','TrackAbort','BackgroundColor',[1 1 .5],'labelpos',[-10 0 -hs*0.74 0]);

        n=n+1.2;
        InitParam(me,'UseScriptTrialLimit','ui','checkbox','value',1,'pref',0,'pos',[h+hs*8.3 n*vs hs*1.37 vs]);
        SetParamUI(me,'UseScriptTrialLimit','label','','string','Use_TrialLimit_in_script','background',[1 .7 .7],'labelpos',[-10 0 -hs*1.36 0]);
        % message box
        uicontrol(fig,'tag','message','style','edit','enable','inact','horiz','left','pos',[h n*vs hs*2.5 vs]);
        InitParam(me,'ChangeSchedule','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*5 n*vs hs*.75 vs]);
        SetParamUI(me,'ChangeSchedule','label','','string','New Schedule','BackgroundColor',[1 1 0],'labelpos',[-10 0 -hs*.74 0]);
        InitParam(me,'StimEditGUI','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*6 n*vs hs*.75 vs]);
        SetParamUI(me,'StimEditGUI','label','','string','Edit Stimuli','BackgroundColor',[0 1 0],'labelpos',[-10 0 -hs*.74 0]);

        n=n+1.2;
        InitParam(me,'Stim_Disp','ui','disp','user',Str,'value',Str{1},'pref',0,'pos',[h n*vs hs*1.25 vs]);
        SetParamUI(me,'Stim_Disp','label','','HorizontalAlignment','Left','labelpos',[-10 0 -hs*1.24 0]);
        InitParam(me,'PlotAxes_Back','value',0,'user',0);
        InitParam(me,'PlotAxes_Forward','value',0,'user',0);
        InitParam(me,'SetPlotAxes_Back2Start','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*1.3 n*vs hs*.5 vs]);
        SetParamUI(me,'SetPlotAxes_Back2Start','label','','string','|<<','labelpos',[-10 0 -hs*.49 0]);
        InitParam(me,'SetPlotAxes_Back','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*1.8 n*vs hs*.5 vs]);
        SetParamUI(me,'SetPlotAxes_Back','label','','string','<','labelpos',[-10 0 -hs*.49 0]);
        InitParam(me,'SetPlotAxes_Default','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*6 n*vs hs*.5 vs]);
        SetParamUI(me,'SetPlotAxes_Default','label','','string','< reset >','labelpos',[-10 0 -hs*.49 0]);
        InitParam(me,'SetPlotAxes_Forward','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*8.5 n*vs hs*.5 vs]);
        SetParamUI(me,'SetPlotAxes_Forward','label','','string','>','labelpos',[-10 0 -hs*.49 0]);
        InitParam(me,'SetPlotAxes_Forward2End','ui','pushbutton','value',0,'pref',0,'pos',[h+hs*9 n*vs hs*.5 vs]);
        SetParamUI(me,'SetPlotAxes_Forward2End','label','','string','>>|','labelpos',[-10 0 -hs*.49 0]);

        InitParam(me,'Trial_Events','value',[],'trial',[]);

        BlankSchedule=zeros(1,GetParam(me,'MaxTrial'));
        InitParam(me,'OdorChannel','value',BlankSchedule);
        InitParam(me,'OdorName','value',BlankSchedule);
        InitParam(me,'Schedule','value',BlankSchedule);
        InitParam(me,'OdorSchedule','value',BlankSchedule);
        InitParam(me,'OdorDelaySchedule','user',BlankSchedule);
        InitParam(me,'RewardDelaySchedule','user',BlankSchedule);
        InitParam(me,'Port_Side','value',BlankSchedule);
        InitParam(me,'Cue_Port_Side','value',BlankSchedule);
        InitParam(me,'VP_LED','value',BlankSchedule);
        InitParam(me,'Result','value',BlankSchedule);
        InitParam(me,'OdorPokeDur','value',BlankSchedule);
        InitParam(me,'nTonePoke','value',BlankSchedule);
        InitParam(me,'JntGroup','value',JntGrp);
        InitParam(me,'JntGrpXTitle','value',JntGrpXTitle);
        InitParam(me,'WPort_in2LastOut','value',BlankSchedule,'user',0);
        InitParam(me,'WPort_in2_2ndLastOut','value',BlankSchedule);

        RP = rpbox('getstatemachine');
        RP = SetInputEvents(RP, [inputevents{GetParam('RPBox','port_number')}(1:6) 0 ], 'ai'); %'0' for virtual inputs (sched wave inputs)
        inputevents{GetParam('RPBox','port_number')}=GetInputEvents(RP);
        RP = SetOutputRouting(RP, [output_routing(1:2);struct('type', 'sched_wave', 'data', '')]);
        output_routing=GetOutputRouting(RP);
        %SetScheduledWaves(machine, [trig_id(2^id for use) wave_in_col wave_out_col(-1:none) dioline(-1:none) sound_trig(-1:none) pre on refrac])
        %         RP = SetScheduledWaves(RP,[0 6  -1 -1 0 rdd .001 0.00]);  %sched_wave for fixed trial length
        RP = SetStateMatrix(RP, [0 0 0 0 0 0 0 0 180 0 0 0]);
        RP = rpbox('setstatemachine', RP);
        eval([me '(''changeschedule'');']);
        eval([me '(''script'');']);
        figure_property=get(fig);
        if isfield(figure_property,'PaperPositionMode')
            set(fig,'PaperPositionMode','auto');
        end
        set(fig,'pos',[140 100 hs*10 (n+26)*vs],'visible','on');
        update_plot;

        vp_sound=InitVP_Sound;
        rpbox('initrpsound');
        rpbox('loadrpsound',vp_sound,1+OneChSound_ID,OneChSound_side);

    case 'trialend'

    case 'stimeditgui'
        StimEditGUI;

    case 'setstimparams'
        StimParam=SetStimParams;
        eval([me '(''changeschedule'');']);
        param_string=GetParam(me,'StimParam','user');
        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';
        SetParam(me,'Stim_Disp','user',Str);
        figure(findobj('Type','figure','Name','Stimulus Parameters','NumberTitle','off','Menu','None','File',me));

    case 'reset'
        Message('control','wait for RP (RP2/RM1) reseting');
        if Getparam(me,'ClearScore')
            clear_score;
        end
        SetParam(me,'Trial_Events','value',[],'trial',[]);
        BlankSchedule=zeros(1,GetParam(me,'MaxTrial'));
        SetParam(me,'Schedule','value',BlankSchedule);
        SetParam(me,'OdorChannel','value',BlankSchedule);
        SetParam(me,'OdorName','value',BlankSchedule);
        SetParam(me,'Port_Side','value',BlankSchedule);
        SetParam(me,'Cue_Port_Side','value',BlankSchedule);
        SetParam(me,'VP_LED','value',BlankSchedule);
        SetParam(me,'Result','value',BlankSchedule);
        SetParam(me,'OdorPokeDur','value',BlankSchedule);
        SetParam(me,'nTonePoke','value',BlankSchedule);
        SetParam(me,'Hit_Streak',0);
        SetParam(me,'n_LR_switch',0);
        SetParamUI(me,'DeliverLeftH2O','String','0');
        SetParamUI(me,'DeliverRightH2O','String','0');
        RP = rpbox('getstatemachine');
        RP = SetInputEvents(RP, [inputevents{GetParam('RPBox','port_number')}(1:6) 0 ], 'ai'); %'0' for virtual inputs (sched wave inputs)
        inputevents{GetParam('RPBox','port_number')}=GetInputEvents(RP);
        RP = SetOutputRouting(RP, [output_routing(1:2);struct('type', 'sched_wave', 'data', '')]);
        output_routing=[output_routing(1:2);struct('type', 'sched_wave', 'data', '')];
        %SetScheduledWaves(machine, [trig_id(2^id for use) alin_col alout_col(-1:none) dioline(-1:none) sound_trig(-1:none) pre on refrac])
        %         RP = SetScheduledWaves(RP,[0 6 -1 5 0 1 1 0.00]);  %sched_wave for fixed trial length
        RP = SetStateMatrix(RP, [0 0 0 0 0 0 0 0 180 0 0 0]);
        eval([me '(''changeschedule'');']);
        eval([me '(''script'');']);
        Message('control','');

    case {'lwatervalvedur' 'rwatervalvedur'}
        RP = rpbox('getstatemachine');
        [state_matrix rdd vpd stm_dur]=state_transition_matrix;
        RP = SetScheduledWaves(RP,[0 6  -1 -1 0     rdd .01 .01;
                                   1 6  -1 -1 0     vpd .01 .01;
                                   2 6  -1 -1 0 stm_dur .01 .01]);  %sched_wave for fixed stimulus duration
        RP = rpbox('setstatemachine', RP);
        RPbox('send_matrix',state_matrix);

    case {'init_schedule','changeschedule'}
        change_schedule;
        if ~GetParam('rpbox','run')
            RP = rpbox('getstatemachine');
            [state_matrix rdd vpd stm_dur]=state_transition_matrix;
            RP = SetScheduledWaves(RP,[0 6  -1 -1 0     rdd .01 .01;
                                       1 6  -1 -1 0     vpd .01 .01;
                                       2 6  -1 -1 0 stm_dur .01 .01]);  %sched_wave for fixed stimulus duration
            RP = rpbox('setstatemachine', RP);
            RPbox('send_matrix',state_matrix);
        end
        update_plot;

    case 'setplotaxes_back2start'
        SetParam(me,'PlotAxes_Back',GetParam(me,'MaxTrial'));
        update_plot;
    case 'setplotaxes_back'
        SetParam(me,'PlotAxes_Back',GetParam(me,'PlotAxes_Back')+50);
        update_plot;
    case 'setplotaxes_default'
        SetParam(me,'PlotAxes_Back',0);
        SetParam(me,'PlotAxes_Forward',0);
        update_plot;
    case 'setplotaxes_forward'
        SetParam(me,'PlotAxes_Forward',GetParam(me,'PlotAxes_Forward')+50);
        update_plot;
    case 'setplotaxes_forward2end'
        SetParam(me,'PlotAxes_Forward',GetParam(me,'MaxTrial'));
        update_plot;

    case 'update'
        update_event;
        if GetParam(me,'update_plot_flag')
            update_plot;
            Setparam(me,'update_plot_flag',0);
        end

    case 'state512'
        update_event;
        CountedTrial    =GetParam(me,'CountedTrial')+1;
        Result          =GetParam(me,'Result');
        New_Events      =GetParam(me,'Trial_Events','value');
        Trial_Events    =GetParam(me,'Trial_Events','trial');
        RightHit=GetParam(me,'RightHit');
        LeftHit=GetParam(me,'LeftHit');
        RightAbort=GetParam(me,'RightAbort');
        LeftAbort=GetParam(me,'LeftAbort');
        RightMiss=GetParam(me,'RightMiss');
        LeftMiss=GetParam(me,'LeftMiss');
        RightFalse=GetParam(me,'RightFalse');
        LeftFalse=GetParam(me,'LeftFalse');
        Port_Side       =GetParam(me,'Port_Side');
        Cue_Port_Side   =GetParam(me,'Cue_Port_Side');
        pts             =Port_Side(CountedTrial);         % water port_side 1:Right, 2:Left, 3:Both
        if CountedTrial>6
            Hit_Streak=(floor(Result(CountedTrial:-1:CountedTrial-5))==1)*1;
            if prod(Hit_Streak(1:6))
                SetParam(me,'Hit_Streak',6);
            elseif prod(Hit_Streak(1:5))
                SetParam(me,'Hit_Streak',5);
            elseif prod(Hit_Streak(1:4))
                SetParam(me,'Hit_Streak',4);
            elseif prod(Hit_Streak(1:3))
                SetParam(me,'Hit_Streak',3);
            elseif prod(Hit_Streak(1:2))
                SetParam(me,'Hit_Streak',2);
            elseif Hit_Streak(1)
                SetParam(me,'Hit_Streak',1);
            else
                SetParam(me,'Hit_Streak',0);
            end
        end

        SetParam(me,'TotalScore',str2double(sprintf('%0.4g',(RightHit+LeftHit)/(RightHit+LeftHit+RightMiss+LeftMiss+RightFalse+LeftFalse+RightAbort+LeftAbort))));
        SetParam(me,'ValidScore',str2double(sprintf('%0.4g',(RightHit+LeftHit)/(RightHit+LeftHit+RightMiss+LeftMiss+RightFalse+LeftFalse))));
        SetParam(me,'RightScore',str2double(sprintf('%0.4g',RightHit/(RightHit+RightMiss+LeftFalse))));
        SetParam(me,'LeftScore',str2double(sprintf('%0.4g',LeftHit/(LeftHit+LeftMiss+RightFalse))));

        RcntHis=GetParam(me,'RecentHistory');
        rcnt_trial=max(1,CountedTrial-RcntHis):CountedTrial;
        rRightHit   =size(find(floor(Result(rcnt_trial))==1 & Port_Side(rcnt_trial)==1),2);
        rLeftHit    =size(find(floor(Result(rcnt_trial))==1 & Port_Side(rcnt_trial)==2),2);
        rRightScore =rRightHit/size(find(Port_Side(rcnt_trial)==1),2);
        rLeftScore  =rLeftHit/size(find(Port_Side(rcnt_trial)==2),2);
        SetParam(me,'rRightScore',sprintf('%0.4g',rRightScore));
        SetParam(me,'rLeftScore',sprintf('%0.4g',rLeftScore));
        SetParam(me,'RecentScore',sprintf('%0.4g',(rRightHit+ rLeftHit)/length(rcnt_trial)));

        eval([me '(''script'');']);
        SetParam(me,'CountedTrial',CountedTrial);
        SetParam(me,'Trial_Events','trial',[Trial_Events {New_Events} ]);
        SetParam(me,'Trial_Events','value',[]);

        OdorSchedule=GetParam(me,'OdorSchedule');
        Schedule=GetParam(me,'Schedule');
        OdorChannel = GetParam(me,'OdorChannel');
        OdorName = GetParam(me,'OdorName');
        VP_LED = GetParam(me,'VP_LED');
        if (GetParam(me,'False_Correction')&& Result(CountedTrial)==2)||(GetParam(me,'Miss_Correction')&& Result(CountedTrial)==3)||...
                (GetParam(me,'Abort_Correction')&& Result(CountedTrial)==4)
            Delay_correction=ceil(rand*GetParam(me,'CorrectionTrial'));
            Schedule(CountedTrial+Delay_correction)=Schedule(CountedTrial);
            Port_Side(CountedTrial+Delay_correction)=pts;
            Cue_Port_Side(CountedTrial+Delay_correction)=Cue_Port_Side(CountedTrial);
            OdorChannel(CountedTrial+Delay_correction)=OdorChannel(CountedTrial);
            OdorName(CountedTrial+Delay_correction) =OdorName(CountedTrial);
            OdorSchedule(CountedTrial+Delay_correction)=OdorSchedule(CountedTrial);
            VP_LED(CountedTrial+Delay_correction)=VP_LED(CountedTrial);
            SetParam(me,'OdorSchedule',OdorSchedule);
            SetParam(me,'Schedule',Schedule);
            SetParam(me,'Port_Side',Port_Side);
            SetParam(me,'Cue_Port_Side',Cue_Port_Side);
            SetParam(me,'VP_LED',VP_LED);
            SetParam(me,'OdorChannel',OdorChannel);
            SetParam(me,'OdorName',OdorName);
        end

        % adaptive timeout
        if GetParam(me,'AdaptiveTimeOut') && CountedTrial>20
            performance_index=[-1 .5  .6  .65 .7 .75  .8 .85 .9];
            suggested_timeout=[ 2 2.5  3  3.5  4  4.5  5  6   7];
            Setparam(me,'TimeOut',suggested_timeout(find(GetParam(me,'ValidScore')>performance_index, 1, 'last' )));
        end

        RP = rpbox('getstatemachine');
        [state_matrix rdd vpd stm_dur]=state_transition_matrix;
        %       ID IN_EVENT_COL OUT_EVENT_COL DIO_LINE SOUND_TRIG PREAMBLE SUSTAIN REFRACTION
        RP = SetScheduledWaves(RP,[0 6  -1 -1 0     rdd .01 .01;
                                   1 6  -1 -1 0     vpd .01 .01;
                                   2 6  -1 -1 0 stm_dur .01 .01]);  %sched_wave for fixed stimulus duration
        RP = rpbox('setstatemachine', RP);
        RPbox('send_matrix',state_matrix);

        datapath=GetParam('control','datapath');
        str=GetParam(me,'LoadMouseSetting','list');
        str=str{GetParam(me,'LoadMouseSetting')};
        if ~strcmp(str,'none') && mod(CountedTrial,20)==1
            ds=str_sep(datestr(date,26),'/');
            ds=[ds{1} ds{2} ds{3}];
            save([datapath '\' str '_' ds  '_' me '_autosave.mat'],'exper');
        end

        % check trial_limit & H2O_limit
        if (RightHit+LeftHit+RightFalse+LeftFalse) > GetParam(me,'trial_limit') ||...
                (GetParam(me,'lefth2oreward')+GetParam(me,'righth2oreward')) > GetParam(me,'H2O_limit')
            SetParam('rpbox','Run',0);
            Message(me,'Reached Trial/H2O Limt ==> Stop Exp','cyan');
            %             for i=1:10
            %                 beep;
            %                 pause(2);
            %             end
        end
        Setparam(me,'update_plot_flag',1);

    case 'script'
        CountedTrial = GetParam(me,'CountedTrial')+GetParam('rpbox','run');
        change_schedule_flag=0;
        cScrpt=GetParam(me,'Script');   %current script
        if ismember(cScrpt,[2 3 4 5 6 7 8 9 10 11 12 13 14 15])
            Setparam(me,'false_correction',0);
            Setparam(me,'miss_correction',0);
            Setparam(me,'abort_correction',0);
            if GetParam(me,'UseScriptTrialLimit')
                SetParam(me,'trial_limit',2299);
                SetParam(me,'H2O_limit',1599);
            end
                %             Setparam(me,'use_hit_streak',0);
            blocklength=GetParam(me,'blocklength');
            StimParam=GetParam(me,'stimparam');
            param_string=GetParam(me,'StimParam','user');
            stim_prob_idx=find(strcmp(param_string,'stimulus probability'));
            VDL_cue_idx=(strcmp(param_string,'VP LED cue'));
            Result=GetParam(me,'Result');
            Result=Result(CountedTrial:-1:1);
            PortSide=GetParam(me,'Port_Side');
            PortSide=PortSide(CountedTrial:-1:1);
            %                 first_same_side_idx=find(PortSide(1:CountedTrial)~=PortSide(1),1)-1;
            first_reward_ind=find(PortSide,1);
            if isempty(first_reward_ind)
                first_same_side_idx=CountedTrial;
            else
                first_same_side_idx=find(PortSide(1:CountedTrial)~=PortSide(first_reward_ind)&PortSide(1:CountedTrial)~=0,1)-1;
                if isempty(first_same_side_idx)
                    first_same_side_idx=CountedTrial;
                end
            end
            n_LR_switch=GetParam(me,'n_LR_switch');
            yymmddhhmmss=clock;
            rand('twister',sum(yymmddhhmmss(1)*10000+yymmddhhmmss(2)*100+yymmddhhmmss(3))*n_LR_switch);
            rnum=(rand(1,1)-.5);
            recent_side_reward=sum((floor(Result(1:first_same_side_idx))==1).*(PortSide(1:first_same_side_idx)>0));

            if CountedTrial==0
                % automatically convert stim type to VP led and Both LED cue
                str=GetParam(me,'LoadMouseSetting','list');
                if (~prod(single(strcmp(StimParam([9 11],VDL_cue_idx),'2'))) || ~prod(single(strcmp(StimParam([10 12],VDL_cue_idx),'1'))) )...
                    && isempty(strfind(str{GetParam(me,'LoadMouseSetting')},'DelayGo'))
                    StimParam([9 11],VDL_cue_idx)={'2'};            %use both LED cue
                    StimParam([10 12],VDL_cue_idx)={'1'};           %use VP LED cue
                    change_schedule_flag=1;
                end
                if ismember(cScrpt,[2 4])
                    StimParam(1:end,stim_prob_idx)={'0'};           %all other 'stimulus probability'=0;
                    StimParam(10,stim_prob_idx)={'1'};              %L_no_odor, VP side LED, 'stimulus probability'=1;
                    SetParam(me,'p_Cued_Switch','user',cScrpt==4);
                    Setparam(me,'TimeOut',4);
                    change_schedule_flag=1;
                elseif ismember(cScrpt,[3 5])
                    StimParam(1:end,stim_prob_idx)={'0'};           %all other 'stimulus probability'=0;
                    StimParam(12,stim_prob_idx)={'1'};              %R_no_odor, VP side LED, 'stimulus probability'=1;
                    SetParam(me,'p_Cued_Switch','user',cScrpt==5);
                    Setparam(me,'TimeOut',4);
                    change_schedule_flag=1;
                elseif ismember(cScrpt,[6 8 10 14]) && ~strcmp(StimParam{9,stim_prob_idx},'1');
                    StimParam(1:end,stim_prob_idx)={'0'};            %all other 'stimulus probability'=0;
                    StimParam(9,stim_prob_idx)={'1'};               %L_no_odor 'stimulus probability'=1;
                    change_schedule_flag=1;
                elseif ismember(cScrpt,[7 9 11 15]) && ~strcmp(StimParam{11,stim_prob_idx},'1');
                    StimParam(1:end,stim_prob_idx)={'0'};            %all other 'stimulus probability'=0;
                    StimParam(11,stim_prob_idx)={'1'};          %R_no_odor 'stimulus probability'=1;
                    change_schedule_flag=1;
                elseif ismember(cScrpt,[12 13])
                    if size(StimParam,1)<13 || GetParam(me,'BlockLength')~=1
                        StimParam(13:14,:)=StimParam([9 11],:);
                        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';
                        SetParam(me,'Stim_Disp','user',Str);
                        SetParam(me,'BlockLength',1);
                        SetParam(me,'BlockLength_jitter',0);
                    end
                    if ismember(cScrpt,12) && ~strcmp(StimParam{13,stim_prob_idx},'1');
                        StimParam(1:end,stim_prob_idx)={'0'};            %all other 'stimulus probability'=0;
                        StimParam(13,stim_prob_idx)={'1'};               %L_no_odor 'stimulus probability'=1;
                        change_schedule_flag=1;
                    elseif ismember(cScrpt,13) && ~strcmp(StimParam{14,stim_prob_idx},'1');
                        StimParam(1:end,stim_prob_idx)={'0'};            %all other 'stimulus probability'=0;
                        StimParam(14,stim_prob_idx)={'1'};               %R_no_odor 'stimulus probability'=1;
                        change_schedule_flag=1;
                    end
                end
            elseif recent_side_reward<blocklength+round(rnum*GetParam(me,'BlockLength_jitter')*2)
                if ismember(cScrpt,[2 3 4 5 6 7 8 9 10 11 12 13 14 15])
                    n_Long_Block=GetParam(me,'n_Long_Block');
                    n_Short_Block=GetParam(me,'n_Short_Block');
                    n_Cued_Short_Block=GetParam(me,'n_Cued_Short_Block');
                    n_Cued_LB_trial=GetParam(me,'n_Cued_LongBlock_Trial');
                    n_total_Block=n_Long_Block+n_Short_Block+n_Cued_Short_Block;
                    if PortSide(first_reward_ind)==1                    % right side
                        if  ~strcmp(StimParam{12,stim_prob_idx},'1') && (ismember(cScrpt,[2 3])||(~recent_side_reward&&GetParam(me,'p_Cued_Switch','user'))...
                            ||(ismember(cScrpt,[10 11])&&n_Cued_LB_trial>recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),0:n_Long_Block-1))...
                            ||(ismember(cScrpt,[12 13])&&n_Cued_LB_trial>recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block+n_Short_Block:n_Cued_Short_Block+n_Short_Block+n_Long_Block-1)))
                            StimParam(1:end,stim_prob_idx)={'0'};           % all other 'stimulus probability'=0;
                            StimParam(12,stim_prob_idx)={'1'};          % R_no_odor ,VP side LED, 'stimulus probability'=1;
                            Setparam(me,'TimeOut',4);
                            change_schedule_flag=1;
                        elseif  ~strcmp(StimParam{11,stim_prob_idx},'1') &&(ismember(cScrpt,[6 7 8 9 14 15])||(recent_side_reward&&GetParam(me,'p_Cued_Switch','user'))...
                            ||(ismember(cScrpt,[10 11])&&n_Cued_LB_trial<=recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),0:n_Long_Block-1))...
                            ||(ismember(cScrpt,[12 13])&&n_Cued_LB_trial<=recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block+n_Short_Block:n_Cued_Short_Block+n_Short_Block+n_Long_Block-1)))
                            StimParam(1:end,stim_prob_idx)={'0'};           % all other 'stimulus probability'=0;
                            StimParam(11,stim_prob_idx)={'1'};          % R_no_odor ,Both side LED, 'stimulus probability'=1;
                            Setparam(me,'TimeOut',1);
                            change_schedule_flag=1;
                        end
                    elseif PortSide(first_reward_ind)==2                % left side
                        if  ~strcmp(StimParam{10,stim_prob_idx},'1') && (ismember(cScrpt,[2 3])||(~recent_side_reward&&GetParam(me,'p_Cued_Switch','user'))...
                            ||(ismember(cScrpt,[10 11])&&n_Cued_LB_trial>recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),0:n_Long_Block-1))...
                            ||(ismember(cScrpt,[12 13])&&n_Cued_LB_trial>recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block+n_Short_Block:n_Cued_Short_Block+n_Short_Block+n_Long_Block-1)))
                            StimParam(1:end,stim_prob_idx)={'0'};           % all other 'stimulus probability'=0;
                            StimParam(10,stim_prob_idx)={'1'};          % L_no_odor ,VP side LED, 'stimulus probability'=1;
                            Setparam(me,'TimeOut',4);
                            change_schedule_flag=1;
                        elseif  ~strcmp(StimParam{9,stim_prob_idx},'1') &&(ismember(cScrpt,[6 7 8 9 14 15])||(recent_side_reward&&GetParam(me,'p_Cued_Switch','user'))...
                            ||(ismember(cScrpt,[10 11])&&n_Cued_LB_trial<=recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),0:n_Long_Block-1))...
                            ||(ismember(cScrpt,[12 13])&&n_Cued_LB_trial<=recent_side_reward&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block+n_Short_Block:n_Cued_Short_Block+n_Short_Block+n_Long_Block-1)))
                            StimParam(1:end,stim_prob_idx)={'0'};           % all other 'stimulus probability'=0;
                            StimParam(9,stim_prob_idx)={'1'};           %L_no_odor ,Both side LED, 'stimulus probability'=1;
                            Setparam(me,'TimeOut',1);
                            change_schedule_flag=1;
                        end
                    end
                end
            elseif recent_side_reward>=blocklength+round(rnum*GetParam(me,'BlockLength_jitter')*2)
                change_schedule_flag=1;
                n_LR_switch=n_LR_switch+1;
                SetParam(me,'n_LR_switch',n_LR_switch);
                if ismember(cScrpt,[2 3]) || (ismember(cScrpt,[4 5])&& (GetParam(me,'p_Cued_Switch')/100)>rand)
                    % 100% cued switch trial just after switch
                    if PortSide(first_reward_ind)==1                 %right side
                        StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                        StimParam(10,stim_prob_idx)={'1'};          % L_no_odor,VP side LED,  'stimulus probability'=1;
                    elseif PortSide(first_reward_ind)==2             %left side
                        StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                        StimParam(12,stim_prob_idx)={'1'};          % R_no_odor,VP side LED,  'stimulus probability'=1;
                    end
                    SetParam(me,'p_Cued_Switch','user',ismember(cScrpt,[4 5]));
                    Setparam(me,'TimeOut',4);
                elseif ismember(cScrpt,[4 5 6 7 8 9 14 15])
                    if PortSide(first_reward_ind)==1                 %right side
                        StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                        StimParam(9,stim_prob_idx)={'1'};           % L_no_odor 'stimulus probability'=1;
                        if GetParam(me,'n_LR_switch')>20
                            if ismember(cScrpt,[14])
                                SetParam(me,'DefaultRewardSize',3);
                            else
                                SetParam(me,'DefaultRewardSize',2);
                            end
                        end
                    elseif PortSide(first_reward_ind)==2             %left side
                        StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                        StimParam(11,stim_prob_idx)={'1'};          % R_no_odor 'stimulus probability'=1;
                        if GetParam(me,'n_LR_switch')>20
                            if ismember(cScrpt,[15])
                                SetParam(me,'DefaultRewardSize',3);
                            else
                                SetParam(me,'DefaultRewardSize',2);
                            end
                        end
                    end
                    SetParam(me,'p_Cued_Switch','user',0);
                    Setparam(me,'TimeOut',1);
                elseif ismember(cScrpt,[10 11 12 13])
                    SetParam(me,'p_Cued_Switch','user',0);
                    Setparam(me,'TimeOut',3);
                    n_Long_Block=GetParam(me,'n_Long_Block');
                    n_Short_Block=GetParam(me,'n_Short_Block');
                    n_Cued_Short_Block=GetParam(me,'n_Cued_Short_Block');
                    n_Cued_LB_trial=GetParam(me,'n_Cued_LongBlock_Trial');
                    n_total_Block=n_Long_Block+n_Short_Block+n_Cued_Short_Block;
                    if (ismember(cScrpt,[10 11])&&ismember(mod(n_LR_switch,n_total_Block),0:n_Long_Block-1)) ||...
                       (ismember(cScrpt,[12 13])&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block+n_Short_Block:n_Cued_Short_Block+n_Short_Block+n_Long_Block-1))
                        % Determine whether to use LED cue or not based on n_Cued_LB_trial
                        SetParam(me,'BlockLength',20);
                        SetParam(me,'BlockLength_jitter',8);
                        StimParam(:,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                        if PortSide(first_reward_ind)==1                 %right side
                            if n_Cued_LB_trial>0
                                StimParam(10,stim_prob_idx)={'1'};           % L_no_odor 'stimulus probability'=1;
                            else
                                StimParam(9,stim_prob_idx)={'1'};           % L_no_odor 'stimulus probability'=1;
                            end
                        elseif PortSide(first_reward_ind)==2             %left side
                            if n_Cued_LB_trial>0
                                StimParam(12,stim_prob_idx)={'1'};          % R_no_odor 'stimulus probability'=1;
                            else
                                StimParam(11,stim_prob_idx)={'1'};           % R_no_odor 'stimulus probability'=1;
                            end
                        end
                    elseif (ismember(cScrpt,[10 11])&&ismember(mod(n_LR_switch,n_total_Block),n_Long_Block:n_Long_Block+n_Cued_Short_Block-1))||...
                       (ismember(cScrpt,[12 13])&&ismember(mod(n_LR_switch,n_total_Block),0:n_Cued_Short_Block-1))
                        SetParam(me,'BlockLength',1);
                        SetParam(me,'BlockLength_jitter',0);
                        % 100% cued switch trial
                        if PortSide(first_reward_ind)==1                 %right side
                            StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                            StimParam(10,stim_prob_idx)={'1'};          % L_no_odor,VP side LED,  'stimulus probability'=1;
                        elseif PortSide(first_reward_ind)==2             %left side
                            StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                            StimParam(12,stim_prob_idx)={'1'};          % R_no_odor,VP side LED,  'stimulus probability'=1;
                        end
                    elseif (ismember(cScrpt,[10 11])&&ismember(mod(n_LR_switch,n_total_Block),n_Long_Block+n_Cued_Short_Block:n_Long_Block+n_Cued_Short_Block+n_Short_Block-1)) ||...
                       (ismember(cScrpt,[12 13])&&ismember(mod(n_LR_switch,n_total_Block),n_Cued_Short_Block:n_Cued_Short_Block+n_Short_Block-1))
                        StimParam(13:14,:)=StimParam([9 11],:);
                        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';
                        SetParam(me,'Stim_Disp','user',Str);
                        SetParam(me,'BlockLength',1);
                        SetParam(me,'BlockLength_jitter',0);
                        if PortSide(first_reward_ind)==1                 %right side
                            StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                            StimParam(13,stim_prob_idx)={'1'};           % L_no_odor 'stimulus probability'=1;
                        elseif PortSide(first_reward_ind)==2             %left side
                            StimParam(1:end,stim_prob_idx)={'0'};        % all other 'stimulus probability'=0;
                            StimParam(14,stim_prob_idx)={'1'};          % R_no_odor 'stimulus probability'=1;
                        end
                    end
                end
                if ismember(cScrpt,[8 9]) % frustration test
                    Setparam(me,'TimeOut',1);
                    if GetParam(me,'n_LR_switch')>7
                        SetParam(me,'lwatervalvedur',GetParam(me,'lwatervalvedur','user'));
                        SetParam(me,'rwatervalvedur',GetParam(me,'rwatervalvedur','user'));
                        SetParam(me,'False_TimeOutLED',1);
                        SetParam(me,'False_ShortTimeOut',0);
                    elseif GetParam(me,'n_LR_switch')>5
                        if isempty(GetParam(me,'lwatervalvedur','user'))
                            SetParam(me,'lwatervalvedur','user',GetParam(me,'lwatervalvedur'));
                            SetParam(me,'rwatervalvedur','user',GetParam(me,'rwatervalvedur'));
                        end
                        SetParam(me,'False_TimeOutLED',0);
                        SetParam(me,'False_ShortTimeOut',1);
                        SetParam(me,'lwatervalvedur',0.001);
                        SetParam(me,'rwatervalvedur',0.001);
                    end
                end
            end
            rand('twister',sum(100*clock));
            Setparam(me,'StimParam',StimParam);
        elseif ismember(cScrpt,16) % extinction
            if GetParam(me,'countedtrial')>=400
                SetParam(me,'lwatervalvedur',0.001);
                SetParam(me,'rwatervalvedur',0.001);
                Setparam(me,'TimeOut',0.01);
                SetParam(me,'False_TimeOutLED',0);
                SetParam(me,'False_ShortTimeOut',1);
                SetParam(me,'Miss_TimeOutLED',0);
                SetParam(me,'Miss_ShortTimeOut',1);
            end
        elseif ismember(cScrpt,17) % Blue Green switching
%             Setparam(me,'false_correction',0);
%             Setparam(me,'miss_correction',0);
%             Setparam(me,'abort_correction',0);
            if GetParam(me,'UseScriptTrialLimit')
                SetParam(me,'trial_limit',2599);
                SetParam(me,'H2O_limit',1999);
            end
            Setparam(me,'use_hit_streak',0);
            blocklength=GetParam(me,'blocklength');
            Setparam(me,'recenthistory',round(blocklength/2));
            StimParam=GetParam(me,'stimparam');
            param_string=GetParam(me,'StimParam','user');
            stim_prob_idx=find(strcmp(param_string,'stimulus probability'));
            VDL_cue_idx=(strcmp(param_string,'VP LED cue'));
            Result=GetParam(me,'Result');
            Result=Result(CountedTrial:-1:1);
            PortSide=GetParam(me,'Port_Side');
            PortSide=PortSide(CountedTrial:-1:1);
            Schedule=GetParam(me,'Schedule');
            Schedule=Schedule(CountedTrial:-1:1);
            stim_name_idx=find(strcmp(param_string,'stimulus name'));
            stim_name=StimParam(:,stim_name_idx);
            first_reward_ind=find(PortSide,1);
            if isempty(first_reward_ind)
                first_same_color_idx=CountedTrial;
            else
                first_same_color_idx=find(~ismember(Schedule,strmatch(stim_name{Schedule(first_reward_ind)}(1:7),stim_name))&PortSide(1:CountedTrial)~=0,1)-1;
                if isempty(first_same_color_idx)
                    first_same_color_idx=CountedTrial;
                end
            end
            n_LR_switch=GetParam(me,'n_LR_switch');
            yymmddhhmmss=clock;
            rand('twister',sum(yymmddhhmmss(1)*10000+yymmddhhmmss(2)*100+yymmddhhmmss(3))*n_LR_switch);
            rnum=(rand(1,1)-.5);
            recent_side_reward=sum((floor(Result(1:first_same_color_idx))==1).*(PortSide(1:first_same_color_idx)>0));
            if CountedTrial==0
                % automatically convert stim type to VP led and Both LED cue
                if ~prod(single(strcmp(StimParam([9 10 11 12],VDL_cue_idx),'1')))
                    StimParam([9:12],VDL_cue_idx)={'1'};           %use VP LED cue
                    change_schedule_flag=1;
                end
            elseif recent_side_reward<blocklength+round(rnum*GetParam(me,'BlockLength_jitter')*2)
            elseif recent_side_reward>=blocklength+round(rnum*GetParam(me,'BlockLength_jitter')*2) &&...
                    mean([str2num(GetParam(me,'rrightscore')),str2num(GetParam(me,'rleftscore'))])>0.72
                change_schedule_flag=1;
                n_LR_switch=n_LR_switch+1;
                SetParam(me,'n_LR_switch',n_LR_switch);
                    if strfind(stim_name{Schedule(first_reward_ind)},'B_VPLED')            %target:Blue LED
                        StimParam(1:end,stim_prob_idx)={'0'};               % all other 'stimulus probability'=0;
                        StimParam(11:12,stim_prob_idx)={'1'};                   % B_VPLED_G 'stimulus probability'=1;
                    elseif strfind(stim_name{Schedule(first_reward_ind)},'G_VPLED')        %target:Green LED
                        StimParam(1:end,stim_prob_idx)={'0'};           % all other 'stimulus probability'=0;
                        StimParam(7:8,stim_prob_idx)={'1'};           % G_VPLED_B 'stimulus probability'=1;
                    end
                    Setparam(me,'TimeOut',1);
            end
            rand('twister',sum(100*clock));
            Setparam(me,'StimParam',StimParam);
        end

        if change_schedule_flag
            change_schedule;
            PortSide=GetParam(me,'Port_Side');
            CuePortSide=GetParam(me,'Cue_Port_Side');
            if PortSide(1)==0
                PortSide(1)=PortSide(find(PortSide,1));
                CuePortSide(1)=CuePortSide(find(PortSide,1));
            end
            if PortSide(CountedTrial+1)==0
                PortSide(CountedTrial+1)=PortSide(CountedTrial+find(PortSide(CountedTrial+1:end),1));
                CuePortSide(CountedTrial+1)=CuePortSide(CountedTrial+find(PortSide(CountedTrial+1:end),1));
            end
            SetParam(me,'Port_Side',PortSide);
            SetParam(me,'Cue_Port_Side',CuePortSide);
            if ~GetParam('rpbox','run')
                RP = rpbox('getstatemachine');
                [state_matrix rdd vpd stm_dur]=state_transition_matrix;
                RP = SetScheduledWaves(RP,[0 6  -1 -1 0     rdd .01 .01;
                                           1 6  -1 -1 0     vpd .01 .01;
                                           2 6  -1 -1 0 stm_dur .01 .01]);  %sched_wave for fixed stimulus duration
                RP = rpbox('setstatemachine', RP);
                RPbox('send_matrix',state_matrix);
            end
            update_plot;
        end

    case 'load'
        n_missing_results=sum(exper.odor_2afc.param.result.value(1:GetParam(me,'countedtrial'))==0);
        if n_missing_results
            str=[num2str(n_missing_results) ' missing results, Do you want to re-analyze the data?'];
            ButtonName = questdlg(str,'re-analyze trial_events data','Yes','No','No');
            if strcmp(ButtonName,'Yes')
                eval([me '(''restore_event'');']);
            end
        end
        LoadParams(me);
        update_plot;

    case {'savemousesetting' 'saveexpersetting'}
        my_params=GetParam(me);
        my_gui_params=[];
        for i=1:length(my_params)
            if ismember(GetParam(me,my_params{i},'ui'),{'edit','radiobutton','checkbox','popupmenu'})
                my_gui_params = [my_gui_params;my_params(i)];
            end
        end
        extra_param={'stimparam'};
        var = [my_gui_params;extra_param];
        mouse_settings=struct;
        for i=1:length(var)
            mouse_settings.(var{i})=GetParam(me,var{i},'value');
        end
        str=GetParam(me,'LoadMouseSetting','list');
        str=str{GetParam(me,'LoadMouseSetting')};
        if ~strcmp(str,'none')
            save([GetParam(me,'datapath') filesep me '_load_' str '_settings.mat'],'mouse_settings');
        else
            prompt={'Enter a name for setting file'};
            name='Select setting file name';
            numlines=1;
            defaultanswer={'mousename'};
            answer=inputdlg(prompt,name,numlines,defaultanswer);
            if ~isempty(answer)
                save([GetParam(me,'datapath') filesep me '_load_' answer{1} '_settings.mat'],'mouse_settings');
                SetParam(me,'LoadMouseSetting','list',Load_Seting_Strs);
            else
                return;
            end
        end
        out=mouse_settings;

    case 'restore'
        my_params=GetParam(me);
        my_gui_params=[];
        for i=1:length(my_params)
            if ~isempty(GetParam(me,my_params{i},'ui'))
                my_gui_params = [my_gui_params;my_params(i)];
            end
        end
        extra_param={'stimparam'};
        my_gui_params = [my_gui_params;extra_param];
        for i=1:length(my_gui_params)
            fig=findobj('tag',me,'type','figure');
            h=findobj(fig,'tag',my_gui_params{i});
            h_pb=findobj(fig,'tag',my_gui_params{i},'style','pushbutton');
            h=h(h~=h_pb);
            Setparam(me,my_gui_params{i},'h',h);
            if isfield(getfield(exper,me,'param'),my_gui_params{i})
                Setparam(me,my_gui_params{i},'value',GetParam(me,my_gui_params{i}));
            end
        end
        StimParam=GetParam(me,'stimparam');
        eval([me '(''changeschedule'');']);
        param_string=GetParam(me,'StimParam','user');
        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';
        SetParam(me,'Stim_Disp','user',Str);

    case 'loadmousesetting'
        datapath=GetParam(me,'datapath');
        str=GetParam(me,'LoadMouseSetting','list');
        str=str{GetParam(me,'LoadMouseSetting')};
        if ~strcmp(str,'none')
            load([datapath filesep me '_load_' str '_settings.mat']);
            fn=fieldnames(mouse_settings);
            for i=1:length(fn)
                if ~strcmpi(fn{i},'loadmousesetting')
                    Setparam(me,fn{i},'value',mouse_settings.(fn{i}));
                end
            end
            out=mouse_settings;
        end
        eval([me '(''changeschedule'');']);
        StimParam=GetParam(me,'stimparam');
        param_string=GetParam(me,'StimParam','user');
        Str        = StimParam(:,strcmp(param_string,'stimulus name'))';
        SetParam(me,'Stim_Disp','user',Str);

    case 'deliverlefth2o'
        wvd=GetParam(me,'lWaterValveDur');
        L_H2O_port_list=GetParam(me,'L_H2O_port','list');
        l1w=L_H2O_port_list(GetParam(me,'L_H2O_port'));
        rpbox('bit',l1w,1);
        pause(wvd);
        rpbox('bit',l1w,0);
        Message(me,['Left Valve Opened ' num2str(wvd) ' Sec'],'cyan');
        SetParamUI(me,'DeliverLeftH2O','String',num2str(str2double(GetParamUI(me,'DeliverLeftH2O','String'))+1));

    case 'deliverrighth2o'
        wvd=GetParam(me,'rWaterValveDur');
        R_H2O_port_list=GetParam(me,'R_H2O_port','list');
        r1w=R_H2O_port_list(GetParam(me,'R_H2O_port'));
        rpbox('bit',r1w,1);
        pause(wvd);
        rpbox('bit',r1w,0);
        Message(me,['Right Valve Opened ' num2str(wvd) ' Sec'],'cyan');
        SetParamUI(me,'DeliverRightH2O','String',num2str(str2double(GetParamUI(me,'DeliverRightH2O','String'))+1));

    case 'close'
        toclose=findobj('Type','figure','Name','Stimulus Parameters','NumberTitle','off','Menu','None','File',me);
        close(toclose);
        RP = rpbox('getstatemachine');
        %SetScheduledWaves(machine, [trig_id(2^id for use) alin_col alout_col(-1:none) dioline(-1:none) pre on refrac])
        RP = SetInputEvents(RP, inputevents{GetParam('RPBox','port_number')}(1:6), 'ai'); %'0' for virtual inputs (sched wave inputs)
        inputevents{GetParam('RPBox','port_number')}=GetInputEvents(RP);
        output_routing=[output_routing(1:2)];
        RP = SetOutputRouting(RP, output_routing);
        if fake_rp_box~=3
            RP = ClearScheduledWaves(RP);
        end
        RP = SetStateMatrix(RP, [0 0 0 0 0 0 0 180 0 0 ]);
        rpbox('setstatemachine', RP);
        SetParam('rpbox','protocols',1);
    otherwise
        out=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [state_matrix rdd vpd stm_dur]=state_transition_matrix
% varargin={vpd,wad,pts,tnd,tmo}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the columns of the transition matrix represent inputs
% Cin,Cout,Lin,Lout,Rin, Rout, Times-up
% The rows are the states (from Staet 0 upto 32)
% The timer is in unit of seconds, # of columns >= # of states
% DIO output in "word" format, 1=DIO-0_ON, 8=DIO-3_ON (DIO-0~8)
% AO output in "word" format, 1=AO-1_ON, 3=AO-1,2_ON,  (AO-1,2)
global center1led right1led left1led right2led left2led right1water left1water OneChSound_ID
R_H2O_port_list=GetParam(me,'R_H2O_port','list');
r1w=2^R_H2O_port_list(GetParam(me,'R_H2O_port'));
L_H2O_port_list=GetParam(me,'L_H2O_port','list');
l1w=2^L_H2O_port_list(GetParam(me,'L_H2O_port'));
c1l=center1led*0;
CountedTrial = GetParam(me,'CountedTrial')+1;
dd=(GetParam(me,'DirectDelivery')>CountedTrial)*5;
vpd=GetParam(me,'ValidOdorPokeDur');
if vpd < 0.001 % vpd has to be larger than the sampling reate of RPDevice
    vpd=0.001;  % sec
end
wad=GetParam(me,'WaterAvailDur');
Port_Side=GetParam(me,'Port_Side');
pts=Port_Side(CountedTrial);         % water port_side 1:Right, 2:Left, 3:Both
Cue_Port_Side=GetParam(me,'Cue_Port_Side');
cps=Cue_Port_Side(CountedTrial);     % cue port_side 1:Right, 2:Left, 3:Both
OdorSchedule=GetParam(me,'OdorSchedule');
OdorDelaySchedule=GetParam(me,'OdorDelaySchedule','user');
RewardDelaySchedule=GetParam(me,'RewardDelaySchedule','user');
StimParam   =GetParam(me,'StimParam','value');
param_string=GetParam(me,'StimParam','user');
LeftRewardP  =str2double(StimParam(:,strcmp(param_string,'left reward ratio')));
RightRewardP =str2double(StimParam(:,strcmp(param_string,'right reward ratio')));
LeftRewardP=LeftRewardP(OdorSchedule((CountedTrial)));
RightRewardP=RightRewardP(OdorSchedule((CountedTrial)));
if LeftRewardP==-1 && RightRewardP==-1
    pts=-1;         % water port_side -1:No-Go 1:Right, 2:Left, 3:Both
end
stim_name=StimParam(:,strcmp(param_string,'stimulus name'));
stim_name=stim_name{OdorSchedule((CountedTrial))};
C1asRwdPort=~isempty(strfind(stim_name,'Rwd@C1in'));

% C1Poke_TTL=~isempty(strfind(stim_name,'TTL@C1Poke'));
% C1Poke_TTLL=~isempty(strfind(stim_name,'TTLL@C1Poke'));
% H2O_TTL=~isempty(strfind(stim_name,'TTL@H2O'));
% H2O_TTLL=~isempty(strfind(stim_name,'TTLL@H2O'));
% ctw=C1Poke_TTL*2^3 + C1Poke_TTLL*2^4;
% htw=H2O_TTL*2^3+H2O_TTLL*2^4;
OdorCh=str2double(StimParam(:,strcmp(param_string,'Dout Channel')));
och=OdorCh(OdorSchedule((CountedTrial)));
odr=(2^och + 2^0)*(och>0);
dor=GetParam(me,'DelayOdor');
dlo=odr*(~dor);

VPD_LED_cue=str2double(StimParam(:,strcmp(param_string,'VP LED cue')));
VPD_LED_cue=VPD_LED_cue(OdorSchedule((CountedTrial)));
if VPD_LED_cue==1 %use VP LED cue
    if strmatch(stim_name,'B_VPLED_N')
        vpl=[0 0 right1led left1led right1led+left1led];
    elseif strmatch(stim_name,'B_VPLED_G')
        vpl=[0 0 right1led+left2led left1led+right2led right1led+left1led];
    elseif strmatch(stim_name,'G_VPLED_N')
        vpl=[0 0 right2led left2led right2led+left2led];
    elseif strmatch(stim_name,'G_VPLED_B')
        vpl=[0 0 right2led+left1led left2led+right1led right2led+left2led];
    else
        vpl=[0 0 right1led left1led right1led+left1led];
    end
    vpl=vpl(pts+2+cps*(pts==0));
elseif VPD_LED_cue==2 % both LED cue
    vpl=right1led+left1led;
else
    vpl=0;
end

uto=GetParam(me,'UseTimeOutLED');
% toLED=(center1led+right1led+left1led)*uto;
toLED=(center1led)*uto;
mtL=GetParam(me,'Miss_TimeOutLED')*toLED;
ftL=GetParam(me,'False_TimeOutLED')*toLED;
atL=GetParam(me,'Abort_TimeOutLED')*toLED;

iti=GetParam(me,'ITI');
iri=GetParam(me,'IRI');
tmo=GetParam(me,'TimeOut');

usto=GetParam(me,'UseShortTimeOut');
sto=tmo*0.1*usto;
mto=tmo*~(usto*GetParam(me,'Miss_ShortTimeOut')) +sto*GetParam(me,'Miss_ShortTimeOut');
fto=tmo*~(usto*GetParam(me,'False_ShortTimeOut'))+sto*GetParam(me,'False_ShortTimeOut');
ato=tmo*~(usto*GetParam(me,'Abort_ShortTimeOut'))+sto*GetParam(me,'Abort_ShortTimeOut');

lvd=GetParam(me,'lWaterValveDur');
rvd=GetParam(me,'rWaterValveDur');

stm_dur=str2double(StimParam(:,strcmp(param_string,'stimulus duration')));
stm_dur=stm_dur(OdorSchedule((CountedTrial)));
stw=2^2;        % ScheduleWave_for_stimulus_delivery
if stm_dur==0
    stw=0;
    stm_dur=0.001;  % sec
elseif stm_dur < 0.001  % ltd has to be larger than the sampling reate of RPDevice
    stm_dur=0.001;  % sec
end
% ltd=(stm_dur-vpd);  %leftover tone duration
% ltd=ltd*(ltd>0);
% if  0 < ltd & ltd < 0.001  % ltd has to be larger than the sampling reate of RPDevice
%     ltd=0.001;  % sec
% end
ltd=0;

if GetParam(me,'UseMinOdorDelay')
    dly=GetParam(me,'minOdorDelay')/1000;
else
    dly=(GetParam(me,'minOdorDelay')+exp(-rand)*(GetParam(me,'maxOdorDelay')-GetParam(me,'minOdorDelay'))/(exp(1)-1))/1000;
end
if dly < 0.001
    dly=0.001;
end
OdorDelaySchedule(CountedTrial)=dly;
SetParam(me,'OdorDelaySchedule','value',round(dly*1000*10)/10,'user',OdorDelaySchedule);

if GetParam(me,'UseMinRewardDelay')
    rdd=GetParam(me,'minRewardDelay')/1000;
else
    DelayTau = 2.5;
    DelayMin = GetParam(me,'minRewardDelay')/1000;
    DelayMax = GetParam(me,'maxRewardDelay')/1000;
    x=DelayMin+DelayMax;
    while x>(DelayMax-DelayMin)
        x=exprnd(DelayTau,1,1);
    end
    rdd = DelayMin + x;
end
if rdd < 0.001
    rdd=0.001;
end
RewardDelaySchedule(CountedTrial)=rdd;
SetParam(me,'RewardDelaySchedule','value',rdd*1000,'user',RewardDelaySchedule);

gpo=GetParam(me,'GracePokeOutTime')/1000;
dfs=GetParam(me,'Deliver_Full_Stim')*odr;

rwd=[45 44 43];
stk1_rwd=rwd(GetParam(me,'DefaultRewardSize')*~GetParam(me,'Use_Hit_Streak')+GetParam(me,'Use_Hit_Streak'));
Hit_Streak=GetParam(me,'Hit_Streak')*GetParam(me,'Use_Hit_Streak');
stk_list=[stk1_rwd rwd(2) rwd(2) rwd(2) rwd(2) rwd(2) rwd(2+GetParam(me,'Hit_Streak_3x_H2O'))];
stk=stk_list(Hit_Streak+1);

vps=1+OneChSound_ID;

if pts+dd+C1asRwdPort*10==-1
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=59;ab3=59;
    elseif dor %delay odor
        abw=0;abt=1;ab2=3;ab3=58;
    else        %no delay odor
        abw=0;abt=2;ab2=3;ab3=58;
    end
    state_transition_matrix=[ ...                                           % No-Go: no Reward
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp       Timer DIO   AO SChWv
        0    0   0    0   0    0    0     1           iti   0     0    0;   % State  0 "ITI-State"
        2    1  51    1  51    1    1     1           180  c1l    0    0;   % State  1 "Pre-State"
        2   abt  2    2   2    2    2     3           dly dlo+c1l 0    0;   % State  2 "Center Poke in, before tone on / Start Odor delivery"
        3   ab2  3    3   3    3    4    58          .001 odr+c1l 0  abw;   % State  3 "Odor/stim on"
        4    5   9    4   9    4    6     7           wad odr+vpl vps stw;   % State  4 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        5    5   9    5   9    5    6     7           wad dfs+vpl 0    0;   % State  5 "Valid Poke Signal, nose out"
        6    6   9    6   9    6    6     7           wad  vpl    0    0;   % State  6 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        7    7   7    7   7    7    7   stk          .001   0     0    0;   % State  7 "Valid Poke ==> Water!!! :)"
        8    8   8    8   8    8    8   512           ato  atL    0    0;   % State  8 "ShortPoke => Abort => House Light "
        9    9   9    9   9    9    9   512           fto  ftL    0    0;   % State  9 "FalsePoke, wrong side => House Light "
       10   10  10   10  10   10   10   512           mto  mtL    0    0;   % State 10 "ValidTonePoke but missed reward => House Light "
        zeros(21,12);
        32    1  32   32  32   32   32     2           dly   0     0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0     0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0     0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
       512  512 512  512 512  512  512     0           999   0     0    0;   % State 35 "End Of Trial"
        36   36  36   37   9    9    7     7           rdd   0     0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        10   10  36   37   9    9   38    10           gpo   0     0    0;   % State 37 Reward poke out "Grace period"
        10   10   7   38   9    9   38    10           gpo   0     0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7           rdd   0     0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0     0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0     0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0     0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43   512          .001   0     0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44   512          .001   0     0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45   512          .001   0     0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           rvd   0     0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0     0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           rvd   0     0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0     0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           rvd   0     0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51     1          .001  c1l    0    0;
        zeros(6,12);
        58  ab3  58   58  58   58    4     4           vpd odr+c1l 0    0;   % State 58 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        58   59  59   59  59   59   60     8           gpo odr+c1l 0    0;   % State 59 Center Poke out "Grace period"
         4   60  60   60  60   60   60     8           gpo odr+c1l 0    0;]; % State 60 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==0
    rdd=2*rdd;
    lps=33;
    rps=34;
    if  RightRewardP==0&& LeftRewardP==0
    elseif LeftRewardP==0
        lps=9;
    elseif RightRewardP==0
        rps=19;
    end
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=59;ab3=59;
    elseif dor %delay odor
        abw=0;abt=1;ab2=3;ab3=58;
    else        %no delay odor
        abw=0;abt=2;ab2=3;ab3=58;
    end
    state_transition_matrix=[ ...                               % No Reward WAD=inf
        %  Cin Cout Lin Lout Rin Rout SchWv TimeUp       Timer DIO   AO SChWv
        0    0   0    0   0    0    0     1           iti   0     0    0;   % State  0 "ITI-State"
        2    1  51    1  51    1    1     1           180  c1l    0    0;   % State  1 "Pre-State"
        2   abt  2    2   2    2    2     3           dly dlo+c1l 0    0;   % State  2 "Center Poke in, before tone on / Start Odor delivery"
        3   ab2  3    3   3    3    4    58          .001 odr+c1l 0  abw;   % State  3 "Odor/stim on"
        4    5  lps   4  rps   4    6    10           wad odr+vpl vps stw;   % State  4 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        5    5  lps   5  rps   5    6    10           wad dfs+vpl 0    0;   % State  5 "Valid Poke Signal, nose out"
        6    6  lps   6  rps   6    6    10           wad  vpl    0    0;   % State  6 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        7    7   7    7   7    7    7     9          .001   0     0    0;   % State  7 "Valid Poke ==> Water!!! :)"
        8    8   8    8   8    8    8   512           ato  atL    0    0;   % State  8 "ShortPoke => Abort => House Light "
        9    9   9    9   9    9    9   512           fto  ftL    0    0;   % State  9 "FalsePoke, wrong side => House Light "
        10   10  10   10  10   10   10   512           mto  mtL    0    0;   % State 10 "ValidTonePoke but missed reward => House Light "
        zeros(8,12);
        19   19  19   19  19   19   19   512           fto  ftL    0    0;   % State 19 "FalsePoke, wrong side => House Light "
        zeros(12,12);
        32    1  32   32  32   32   32     2           dly   0     0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0     0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0     0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
       512  512 512  512 512  512  512     0           999   0     0    0;   % State 35 "End Of Trial"
        36   36  36   37   9    9    7     7           rdd   0     0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        10   10  36   37   9    9   38    10           gpo   0     0    0;   % State 37 Reward poke out "Grace period"
        10   10   7   38   9    9   38    10           gpo   0     0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7           rdd   0     0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0     0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0     0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0     0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43   512          .001   0     0    0;   % State 43 Hit_Streak>=4, no 3 drops of water
        44   44  44   44  44   44   44   512          .001   0     0    0;   % State 44 Hit_Streak>=1, no 2 drops of water
        45   45  45   45  45   45   45   512          .001   0     0    0;   % State 45 Hit_Streak =0, no 1 drops of water
        46   46  46   46  46   46   46    47           rvd   0     0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0     0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           rvd   0     0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0     0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           rvd   0     0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51     1          .001  c1l    0    0;
        zeros(6,12);
        58  ab3  58   58  58   58    4     4           vpd odr+c1l 0    0;   % State 58 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        58   59  59   59  59   59   60     8           gpo odr+c1l 0    0;   % State 59 Center Poke out "Grace period"
         4   60  60   60  60   60   60     8           gpo odr+c1l 0    0;]; % State 60 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==1
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;abt=1;ab2=59;ab3=59;
    elseif dor %delay odor
        abw=0;abt=1;ab2=3;ab3=58;
    else        %no delay odor
        abw=0;abt=2;ab2=3;ab3=58;
    end
    state_transition_matrix=[ ...                               % Right Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0    0     1           iti   0     0    0;   % State  0 "ITI-State"
        2    1  51   51  51   51    1     1           180  c1l    0    0;   % State  1 "Pre-State"
        2   abt  2    2   2    2    2     3           dly dlo+c1l 0    0;   % State  2 "Center Poke in, before tone on / Start Odor delivery"
        3   ab2  3    3   3    3    4    58          .001 odr+c1l 0  abw;   % State  3 "Odor/stim on"
        4    5   9    4  34    4    6    10           wad odr+vpl vps stw;   % State  4 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        5    5   9    5  34    5    6    10           wad dfs+vpl 0    0;   % State  5 "Valid Poke Signal, nose out"
        6    6   9    6  34    6    6    10           wad  vpl    0    0;   % State  6 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        7    7   7    7   7    7    7   stk          .001   0     0    0;   % State  7 "Valid Poke ==> Water!!! :)"
        8    8   8    8   8    8    8   512           ato   atL   0    0;   % State  8 "ShortPoke => Abort => House Light "
        9    9   9    9   9    9    9   512           fto   ftL   0    0;   % State  9 "FalsePoke, wrong side => House Light "
        10   10  10   10  10   10   10  512           mto   mtL   0    0;   % State 10 "ValidTonePoke but missed reward => House Light "
        zeros(21,12);
        32    1  32   32  32   32   32     2           dly  dlo   0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37   9    9    7     7         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        10   10  36   37   9    9   38    10           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        10   10   7   38   9    9   38    10           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           rvd  r1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           rvd  r1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           rvd  r1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51     1          .001   0    0    0;
        zeros(6,12);
        58  ab3  58   58  58   58    4     4           vpd odr+c1l 0    0;   % State 58 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        58   59  59   59  59   59   60     8           gpo odr+c1l 0    0;   % State 59 Center Poke out "Grace period"
         4   60  60   60  60   60   60     8           gpo odr+c1l 0    0;]; % State 60 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==2
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;abt=11;ab2=69;ab3=69;
    elseif dor %delay odor
        abw=0;abt=11;ab2=13;ab3=68;
    else        %no delay odor
        abw=0;abt=12;ab2=13;ab3=68;
    end
    state_transition_matrix=[ ...                               % Left Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0     0    11           iti   0     0    0;   % State 0 "ITI-State"
        zeros(10,12);
        12   11  51   51  51   51   11    11           180   c1l   0    0;   % State 11 "Pre-State"
        12  abt  12   12  12   12   12    13           dly dlo+c1l 0    0;   % State 12 "Center Poke in, before tone on / Start Odor delivery"
        13  ab2  13   13  13   13   14    68          .001 odr+c1l 0  abw;   % State 13 "Odor/stim on"
        14   15  33   14  19   14   16    20           wad odr+vpl vps stw;   % State 14 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        15   15  33   15  19   15   16    20           wad dfs+vpl 0    0;   % State 15 "Valid Poke Signal, nose out"
        16   16  33   16  19   16   16    20           wad  vpl    0    0;   % State 16 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        17   17  17   17  17   17   17   stk          .001   0     0    0;   % State 17 "Valid Poke ==> Water!!! :)"
        18   18  18   18  18   18   18   512           ato  atL    0    0;   % State 18 "ShortPoke => Abort => House Light "
        19   19  19   19  19   19   19   512           fto  ftL    0    0;   % State 19 "FalsePoke, wrong side => House Light "
        20   20  20   20  20   20   20   512           mto  mtL    0    0;   % State 20 "ValidTonePoke but missed reward => House Light "
        zeros(11,12);
        32   11  32   32  32   32   32    12           dly   dlo  0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37  19   19   17    17         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        20   20  36   37  19   19   38    20           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        20   20  17   38  19   19   38    20           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           lvd  l1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           lvd  l1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           lvd  l1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51    11          .001   0    0    0;
        zeros(16,12);
        68  ab3  68   68  68   68   14    14           vpd odr+c1l 0    0;  % State 68 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        68   69  69   69  69   69   70    18           gpo odr+c1l 0    0;  % State 69 Center Poke out "Grace period"
        14   70  70   70  70   70   70    18           gpo odr+c1l 0    0;];% State 70 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==3
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;abt=21;ab2=79;ab3=79;
    elseif dor %delay odor
        abw=0;abt=21;ab2=23;ab3=78;
    else        %no delay odor
        abw=0;abt=22;ab2=23;ab3=78;
    end
    state_transition_matrix=[ ...                               % Both Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0     0    21           iti   0     0    0;   % State 0 "ITI-State"
        zeros(20,12);
        22   21  51   51  51   51   21    21           180   c1l   0    0;   % State 21 "Pre-State"
        22  abt  22   22  22   22   22    23           dly dlo+c1l 0    0;   % State 22 "Center Poke in, before tone on / Start Odor delivery"
        23  ab2  23   23  23   23   24    78          .001 odr+c1l 0  abw;   % State 23 "Odor/stim on"
        24   25  33   24  34   24   26    30           wad odr+vpl vps stw;   % State 24 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        25   25  33   25  34   25   26    30           wad dfs+vpl 0    0;   % State 25 "Valid Poke Signal, nose out"
        26   26  33   26  34   26   26    30           wad  vpl    0    0;   % State 26 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        27   27  27   27  27   27   27   stk          .001   0     0    0;   % State 27 "Valid Poke ==> Water!!! :)"
        28   28  28   28  28   28   28   512           ato  atL    0    0;   % State 28 "ShortPoke => Abort => House Light "
        29   29  29   29  29   29   29   512           fto  ftL    0    0;   % State 29 "FalsePoke, wrong side => House Light "
        30   30  30   30  30   30   30   512           mto  mtL    0    0;   % State 30 "ValidTonePoke but missed reward => House Light "
        zeros(1,12);
        32   11  32   32  32   32   32    12           dly   dlo  0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37  19   19   17    17         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        20   20  36   37  19   19   38    20           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        20   20  17   38  19   19   38    20           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           lvd  l1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           lvd  l1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           lvd  l1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51    11          .001  c1l   0    0;
        zeros(26,12);
        78  ab3  78   78  78   78   24    24           vpd odr+c1l 0   0;   % State 78 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        78   79  79   79  79   79   80    28           gpo odr+c1l 0   0;   % State 79 Center Poke out "Grace period"
        24   80  80   80  80   80   80    28           gpo odr+c1l 0   0;]; % State 80 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==5
    state_transition_matrix=[ ...                               % No Water Port direct delivery
   %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0    0   0     0    1           iti   0    0    0;   % State 0 "ITI-State"
        2    1   1    1    1   1     1    1           180   0    0    0;   % State 1 "Pre-State"
        2    1   2    2    2   2     2    3           dly   dlo  0    0;   % State 2 "Center Poke in, before tone on"
        3    1   3    3    3   3     3    4           vpd   odr  0    0;   % State 3 "tone on"
        4    4   4    4    4   4     5    4           999   odr  0   stw;  % State 4 "pre- Center Poke out"
        5    5   5    5    5   5     5    6           rdd   0    0    0;   % State 5 "Reward Delay period"
        6    6   6    6    6   6     6   37           rvd   0    0    0;   % State 6 "Valid Poke ==> Water!!! :)"
        7    7   9    9    9   9     7   10           wad   0    0    0;   % State 7 "mouse finds Water!!! :)"
        8    8   8    8    8   8     8    0          .001   0    0    0;   % State 8 "ShortPoke => Abort"
        9    9   9    9    9   9     9   512         .001   0    0    0;   % State 9 "FalsePoke, wrong side "
       10   10  10   10   10  10    10   512         .001   0    0    0;   % State10 "ValidTonePoke but missed reward"
        zeros(24,12);
       35   35  35   35   35  35    35   512         .001   0    0    0;   % State 35 "mouse finds Water!!! :)"
        zeros(1,12);
       37   37   9    37   9  37    37   10           wad   0    0    0;]; % State 37 "mouse finds Water!!! :)"
elseif pts+dd+C1asRwdPort*10==6
    state_transition_matrix=[ ...                               % Right Side Water Port direct delivery
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0    0   0     0    1           iti   0    0    0;   % State 0 "ITI-State"
        2    1   1    1    1   1     1    1           180   0    0    0;   % State 1 "Pre-State"
        2    1   2    2    2   2     2    3           dly   dlo  0    0;   % State 2 "Center Poke in, before tone on"
        3    1   3    3    3   3     3    4           vpd   odr  0    0;   % State 3 "tone on"
        4    4   4    4    4   4     5    4           999   odr  0   stw;  % State 4 "pre- Center Poke out"
        5    5   5    5    5   5     5    6           rdd   0    0    0;   % State 5 "Reward Delay period"
        6    6   6    6    6   6     6    7           rvd   r1w  0    0;   % State 6 "Valid Poke ==> Water!!! :)"
        7    7   9    9   35  35     7   10           wad   0    0    0;   % State 7 "mouse finds Water!!! :)"
        8    8   8    8    8   8     8    0          .001   0    0    0;   % State 8 "ShortPoke => Abort"
        9    9   9    9    9   9     9   512         .001   0    0    0;   % State 9 "FalsePoke, wrong side "
       10   10  10   10   10  10    10   512         .001   0    0    0;   % State10 "ValidTonePoke but missed reward"
        zeros(24,12);
       35   35  35   35   35  35   35    512         .001   0    0    0;]; % State 35 "mouse finds Water!!! :)"
elseif pts+dd+C1asRwdPort*10==7
    state_transition_matrix=[ ...                                % Left Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0     0   0    0   0    0    0    11           iti   0    0    0;   % State 0 "ITI-State"
        zeros(10,12);
        12   11  11   11  11   11   11    11           180   0    0    0;   % State 11 "Pre-State"
        12   11  12   12  12   12   12    13           dly   dlo  0    0;   % State 12 "Center Poke in, before tone on"
        13   11  13   13  13   13   13    14           vpd   odr  0    0;   % State 13 "tone on"
        14   14  14   14  14   14   15    14           999   odr  0   stw;  % State 14 "pre- Center Poke out"
        15   15  15   15   15  15   15    16           rdd   0    0    0;   % State 15 "Reward Delay period"
        16   16  16   16   16  16   16    17           lvd   l1w  0    0;   % State 16 "Valid Poke ==> Water!!! :)"
        17   17  36   36   17  19   19    20           wad   0    0    0;   % State 17 "mouse finds Water!!! :)"
        18   18  18   18   18  18   18     0          .001   0    0    0;   % State 18 "ShortPoke => Abort"
        19   19  19   19   19  19   19    512         .001   0    0    0;   % State 19 "FalsePoke, wrong side "
        10   10  10   10   10  10   10    512         .001   0    0    0;   % State 20 "ValidTonePoke but missed reward"
        zeros(15,12);
        36   36  36   36   36  36   36    512         .001   0    0    0;]; % State 36 "mouse finds Water!!! :)"
elseif pts+dd+C1asRwdPort*10==8
    state_transition_matrix=[ ...                                % Both Side Water Port
        %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0    0     21          iti    0    0    0;   % State 0 "ITI-State"
        zeros(20,12);
        22   21  21   21  21   21   21    21          180    0    0    0;   % State 21 "Pre-State"
        22   27  22   22  22   22   22    23          dly   dlo   0    0;   % State 22 "Center Poke in, before tone on"
        23   27  23   23  23   23   23    24+(ltd==0) vpd   odr   0    0;   % State 23 "tone on"
        24   25  25    0  25    0   24    25          ltd   odr   0    0;   % State 24 "pre- Center Poke out"
        25   25  25   25  25   25   25    26          rdd   odr   0    0;   % State 25 "Reward Delay period"
        26   26  26   26  26   26   26    27   (rvd+lvd)/2 l1w+r1w 0    0;   % State 26 "Valid Poke ==> Water!!! :)"
        27   27 512  512 512  512   27    30          wad   0     0    0;   % State 27 "mouse finds Water!!! :)"
        28   28  28   28  28   28   28     0         .001   0     0     0;   % State 28 "ShortPoke => Abort => House Light "
        29   29  29   29  29   29   29     0         .001   0     0     0;   % State 29 "FalsePoke, wrong side => House Light "
        30   30  30   30  30   30   30    512        .001   0     0     0;   % State 30 "ValidTonePoke but missed reward => House Light "
        zeros(4,12);
        35   35  35   35   35  35   35    512         .001   0    0    0;]; % State 35 "mouse finds Water!!! :)"
elseif pts+dd+C1asRwdPort*10==10
    rdd=2*rdd;
    lps=54;
    rps=55;
    if  RightRewardP==0&& LeftRewardP==0
    elseif LeftRewardP==0
        lps=56;
    elseif RightRewardP==0
        rps=57;
    end
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=59;ab3=59;
    else
        abw=0;ab2=3;ab3=58;
    end
    state_transition_matrix=[ ...                               % No Reward WAD=inf
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp       Timer DIO   AO SChWv
        0    0   0    0   0    0    0     1           iti   0     0    0;   % State  0 "ITI-State"
        2    1  51    1  51    1    1     1           180  c1l    0    0;   % State  1 "Pre-State"
        2    1   2    2   2    2    2     3           dly dlo+c1l 0    0;   % State  2 "Center Poke in, before tone on / Start Odor delivery"
        3   ab2  3    3   3    3    4    58          .001 odr+c1l 0  abw;   % State  3 "Odor/stim on"
        4    5  lps   4  rps   4    6    10           wad odr+vpl vps stw;   % State  4 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        5    5  lps   5  rps   5    6    10           wad dfs+vpl 0    0;   % State  5 "Valid Poke Signal, nose out"
        6    6  lps   6  rps   6    6    10           wad  vpl    0    0;   % State  6 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        7    7   7    7   7    7    7     9          .001   0     0    0;   % State  7 "Valid Poke ==> Water!!! :)"
        8    8   8    8   8    8    8   512           ato  atL    0    0;   % State  8 "ShortPoke => Abort => House Light "
        9    9   9    9   9    9    9   512           fto  ftL    0    0;   % State  9 "FalsePoke, wrong side => House Light "
       10   10  10   10  10   10   10   512           mto  mtL    0    0;   % State 10 "ValidTonePoke but missed reward => House Light "
       zeros(8,12);
       19   19  19   19  19   19   19   512           fto  ftL    0    0;   % State 19 "FalsePoke, wrong side => House Light "
       zeros(12,12);
       32    1  32   32  32   32   32     2           dly   0     0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
       33   33  33   33  33   33   33    36          .001   0     0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
       34   34  34   34  34   34   34    39          .001   0     0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
      512  512 512  512 512  512  512     0           999   0     0    0;   % State 35 "End Of Trial"
       36   36  36   37   9    9    7     7           rdd   0     0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
       10   10  36   37   9    9   38    10           gpo   0     0    0;   % State 37 Reward poke out "Grace period"
       10   10   7   38   9    9   38    10           gpo   0     0    0;   % State 38 Reward poke out "Grace period" after left reward delay
       39   39   9    9  39   40    7     7           rdd   0     0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
       10   10   9    9  39   40   41    10           gpo   0     0    0;   % State 40 Reward poke out "Grace period"
       10   10   9    9   7   41   41    10           gpo   0     0    0;   % State 41 Reward poke out "Grace period" after right reward delay
       42   42  42   42  42   42   42     0         0.001   0     0    0;   % State 42 poke-in in ITI state, reset ITI timer
       43   43  43   43  43   43   43   512          .001   0     0    0;   % State 43 Hit_Streak>=4, no 3 drops of water
       44   44  44   44  44   44   44   512          .001   0     0    0;   % State 44 Hit_Streak>=1, no 2 drops of water
       45   45  45   45  45   45   45   512          .001   0     0    0;   % State 45 Hit_Streak =0, no 1 drops of water
       46   46  46   46  46   46   46    47           rvd   0     0    0;   % State 46 Hit_Streak>=4, 3 drops of water
       47   47  47   47  47   47   47    48           iri   0     0    0;   % State 47 Hit_Streak>=4, 3 drops of water
       48   48  48   48  48   48   48    49           rvd   0     0    0;   % State 48 Hit_Streak>=2, 2 drops of water
       49   49  49   49  49   49   49    50           iri   0     0    0;   % State 49 Hit_Streak>=2, 2 drops of water
       50   50  50   50  50   50   50   512           rvd   0     0    0;   % State 50 Hit_Streak <2, 1 drops of water
       51   51  51   51  51   51   51     1          .001  c1l    0    0;
        zeros(2,12);
       33   54  54   54  54   54   54    20           wad   0     0    0;   % State 54 Valid Left Choice Poke, Reward Avaiable Dur
       34   55  55   55  55   55   55    10           wad   0     0    0;   % State 55 Valid Right Choice Poke, Reward Avaiable Dur
        9   56  56   56  56   56   56    10           wad   0     0    0;   % State 56 False Left Choice Poke,
       19   57  57   57  57   57   57    20           wad   0     0    0;   % State 57 False Right Choice Poke,
       58  ab3  58   58  58   58    4     4           vpd odr+c1l 0    0;   % State 58 Valid Center Poke, Odor/stim on ==> Wait for Go cue
       58   59  59   59  59   59   60     8           gpo odr+c1l 0    0;   % State 59 Center Poke out "Grace period"
        4   60  60   60  60   60   60     8           gpo odr+c1l 0    0;]; % State 60 Center Poke out "Grace period" after right reward delay
elseif pts+dd+C1asRwdPort*10==11
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=59;ab3=59;
    else
        abw=0;ab2=3;ab3=58;
    end
    state_transition_matrix=[ ...                               % Right Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0    0     1           iti   0     0    0;   % State  0 "ITI-State"
        2    1  51   51  51   51    1     1           180  c1l    0    0;   % State  1 "Pre-State"
        2    1   2    2   2    2    2     3           dly dlo+c1l 0    0;   % State  2 "Center Poke in, before tone on / Start Odor delivery"
        3   ab2  3    3   3    3    4    58          .001 odr+c1l 0  abw;   % State  3 "Odor/stim on"
        4    5  56    4  55    4    6    10           wad odr+vpl vps stw;   % State  4 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        5    5  56    5  55    5    6    10           wad dfs+vpl 0    0;   % State  5 "Valid Poke Signal, nose out"
        6    6  56    6  55    6    6    10           wad  vpl    0    0;   % State  6 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        7    7   7    7   7    7    7   stk          .001   0     0    0;   % State  7 "Valid Poke ==> Water!!! :)"
        8    8   8    8   8    8    8   512           ato   atL   0    0;   % State  8 "ShortPoke => Abort => House Light "
        9    9   9    9   9    9    9   512           fto   ftL   0    0;   % State  9 "FalsePoke, wrong side => House Light "
        10   10  10   10  10   10   10  512           mto   mtL   0    0;   % State 10 "ValidTonePoke but missed reward => House Light "
        zeros(21,12);
        32    1  32   32  32   32   32     2           dly  dlo   0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37   9    9    7     7         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        10   10  36   37   9    9   38    10           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        10   10   7   38   9    9   38    10           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           rvd  r1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           rvd  r1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           rvd  r1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51     1          .001  c1l   0    0;
        zeros(2,12);
        33   54  54   54  54   54   54    20           wad   0    0    0;   % State 54 Valid Left Choice Poke, Reward Avaiable Dur
        34   55  55   55  55   55   55    10           wad   0    0    0;   % State 55 Valid Right Choice Poke, Reward Avaiable Dur
         9   56  56   56  56   56   56    10           wad   0    0    0;   % State 56 False Left Choice Poke,
        19   57  57   57  57   57   57    20           wad   0    0    0;   % State 57 False Right Choice Poke,
        58  ab3  58   58  58   58    4     4           vpd odr+c1l 0   0;   % State 58 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        58   59  59   59  59   59   60     8           gpo odr+c1l 0   0;   % State 59 Center Poke out "Grace period"
         4   60  60   60  60   60   60     8           gpo odr+c1l 0   0;]; % State 60 Center Poke out "Grace period" after right reward delay
elseif pts+dd+C1asRwdPort*10==12
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=69;ab3=69;
    else
        abw=0;ab2=13;ab3=68;
    end
    state_transition_matrix=[ ...                               % Left Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0     0    11           iti   0     0    0;   % State 0 "ITI-State"
        zeros(10,12);
        12   11  51   51  51   51   11    11           180   c1l   0    0;   % State 11 "Pre-State"
        12   11  12   12  12   12   12    13           dly dlo+c1l 0    0;   % State 12 "Center Poke in, before tone on / Start Odor delivery"
        13  ab2  13   13  13   13   14    68          .001 odr+c1l 0  abw;   % State 13 "Odor/stim on"
        14   15  54   14  57   14   16    20           wad odr+vpl vps stw;   % State 14 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        15   15  54   15  57   15   16    20           wad dfs+vpl 0    0;   % State 15 "Valid Poke Signal, nose out"
        16   16  54   16  57   16   16    20           wad  vpl    0    0;   % State 16 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        17   17  17   17  17   17   17   stk          .001   0     0    0;   % State 17 "Valid Poke ==> Water!!! :)"
        18   18  18   18  18   18   18   512           ato  atL    0    0;   % State 18 "ShortPoke => Abort => House Light "
        19   19  19   19  19   19   19   512           fto  ftL    0    0;   % State 19 "FalsePoke, wrong side => House Light "
        20   20  20   20  20   20   20   512           mto  mtL    0    0;   % State 20 "ValidTonePoke but missed reward => House Light "
        zeros(11,12);
        32   11  32   32  32   32   32    12           dly   dlo  0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37  19   19   17    17         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        20   20  36   37  19   19   38    20           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        20   20  17   38  19   19   38    20           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           lvd  l1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           lvd  l1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           lvd  l1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51    11          .001  c1l   0    0;
        zeros(2,12);
        33   54  54   54  54   54   54    20           wad   0    0    0;   % State 54 Valid Left Choice Poke, Reward Avaiable Dur
        34   55  55   55  55   55   55    10           wad   0    0    0;   % State 55 Valid Right Choice Poke, Reward Avaiable Dur
         9   56  56   56  56   56   56    10           wad   0    0    0;   % State 56 False Left Choice Poke,
        19   57  57   57  57   57   57    20           wad   0    0    0;   % State 57 False Right Choice Poke,
        zeros(10,12);
        68  ab3  68   68  68   68   14    14           vpd odr+c1l 0   0;   % State 68 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        68   69  69   69  69   69   70    18           gpo odr+c1l 0   0;   % State 69 Center Poke out "Grace period"
        14   70  70   70  70   70   70    18           gpo odr+c1l 0   0;]; % State 70 Center Poke out "Grace period" after right reward delay

elseif pts+dd+C1asRwdPort*10==13
    if usto*GetParam(me,'TrackAbort')
        abw=2^1;ab2=79;ab3=79;
    else
        abw=0;ab2=23;ab3=78;
    end
    state_transition_matrix=[ ...                               % Both Side Water Port
    %  Cin Cout Lin Lout Rin Rout SchWv TimeUp        Timer DIO  AO
        0    0   0    0   0    0     0    21           iti   0     0    0;   % State 0 "ITI-State"
        zeros(20,12);
        22   21  51   51  51   51   21    21           180   c1l   0    0;   % State 21 "Pre-State"
        22   21  22   22  22   22   22    23           dly dlo+c1l 0    0;   % State 22 "Center Poke in, before tone on / Start Odor delivery"
        23  ab2  23   23  23   23   24    78          .001 odr+c1l 0  abw;   % State 23 "Odor/stim on"
        24   25  54   24  55   24   26    30           wad odr+vpl vps stw;   % State 24 "Valid Poke Signal ==> Set ScheduleWave: 2^1"
        25   25  54   25  54   25   26    30           wad dfs+vpl 0    0;   % State 25 "Valid Poke Signal, nose out"
        26   26  54   26  54   26   26    30           wad  vpl    0    0;   % State 26 "Valid Poke Signal, Odor/stim off, Reward Avaiable Dur"
        27   27  27   27  27   27   27   stk          .001   0     0    0;   % State 27 "Valid Poke ==> Water!!! :)"
        28   28  28   28  28   28   28   512           ato  atL    0    0;   % State 28 "ShortPoke => Abort => House Light "
        29   29  29   29  29   29   29   512           fto  ftL    0    0;   % State 29 "FalsePoke, wrong side => House Light "
        30   30  30   30  30   30   30   512           mto  mtL    0    0;   % State 30 "ValidTonePoke but missed reward => House Light "
        zeros(1,12);
        32   11  32   32  32   32   32    12           dly   dlo  0    0;   % State 32 "Center Poke in, before tone on / Delay Odor delivery"
        33   33  33   33  33   33   33    36          .001   0    0    1;   % State 33 "Left Valid Poke ==> Set ScheduleWave: 2^0 )"
        34   34  34   34  34   34   34    39          .001   0    0    1;   % State 34 "Right Valid Poke ==> Set ScheduleWave: 2^0 )"
        512  512 512  512 512  512  512    0           999   0    0    0;   % State 35 "End Of Trial"
        36   36  36   37  19   19   17    17         2*rdd   0    0    0;   % State 36 " Reward Delay time ==> Wait for water!!! :)"
        20   20  36   37  19   19   38    20           gpo   0    0    0;   % State 37 Reward poke out "Grace period"
        20   20  17   38  19   19   38    20           gpo   0    0    0;   % State 38 Reward poke out "Grace period" after left reward delay
        39   39   9    9  39   40    7     7         2*rdd   0    0    0;   % State 39 " Reward Delay time ==> Wait for water!!! :)"
        10   10   9    9  39   40   41    10           gpo   0    0    0;   % State 40 Reward poke out "Grace period"
        10   10   9    9   7   41   41    10           gpo   0    0    0;   % State 41 Reward poke out "Grace period" after right reward delay
        42   42  42   42  42   42   42     0         0.001   0    0    0;   % State 42 poke-in in ITI state, reset ITI timer
        43   43  43   43  43   43   43    46          .001   0    0    0;   % State 43 Hit_Streak>=4, 3 drops of water
        44   44  44   44  44   44   44    48          .001   0    0    0;   % State 44 Hit_Streak>=1, 2 drops of water
        45   45  45   45  45   45   45    50          .001   0    0    0;   % State 45 Hit_Streak =0, 1 drops of water
        46   46  46   46  46   46   46    47           lvd  l1w   0    0;   % State 46 Hit_Streak>=4, 3 drops of water
        47   47  47   47  47   47   47    48           iri   0    0    0;   % State 47 Hit_Streak>=4, 3 drops of water
        48   48  48   48  48   48   48    49           lvd  l1w   0    0;   % State 48 Hit_Streak>=2, 2 drops of water
        49   49  49   49  49   49   49    50           iri   0    0    0;   % State 49 Hit_Streak>=2, 2 drops of water
        50   50  50   50  50   50   50   512           lvd  l1w   0    0;   % State 50 Hit_Streak <2, 1 drops of water
        51   51  51   51  51   51   51    11          .001  c1l   0    0;
        zeros(2,12);
        33   54  54   54  54   54   54    20           wad   0    0    0;   % State 54 Valid Left Choice Poke, Reward Avaiable Dur
        34   55  55   55  55   55   55    10           wad   0    0    0;   % State 55 Valid Right Choice Poke, Reward Avaiable Dur
        zeros(22,12);
        78  ab3  78   78  78   78   24    24           vpd odr+c1l 0   0;   % State 78 Valid Center Poke, Odor/stim on ==> Wait for Go cue
        78   79  28   28  28   28   80    28           gpo odr+c1l 0   0;   % State 79 Center Poke out "Grace period"
        24   80  28   28  28   28   80    28           gpo odr+c1l 0   0;]; % State 80 Center Poke out "Grace period" after right reward delay
end
state_matrix=state_transition_matrix;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function clear_score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SetParam(me,'CountedTrial',0);
SetParam(me,'LastOdorPokeDur',0);

SetParam(me,'TotalScore',0);
SetParam(me,'RecentScore',0);
SetParam(me,'rLeftScore',0);
SetParam(me,'rRightScore',0);

SetParam(me,'LeftScore',0);
SetParam(me,'LeftHit',0);
SetParam(me,'LeftMiss',0);
SetParam(me,'LeftFalse',0);
SetParam(me,'LeftAbort',0);

SetParam(me,'RightScore',0);
SetParam(me,'RightHit',0);
SetParam(me,'RightMiss',0);
SetParam(me,'RightFalse',0);
SetParam(me,'RightAbort',0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_event
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CountedTrial    =GetParam(me,'CountedTrial')+1;
dd=(GetParam(me,'DirectDelivery')>CountedTrial);
Trial_Events    =GetParam(me,'Trial_Events','value');
Result          =GetParam(me,'Result');
OdorPokeDur     =GetParam(me,'OdorPokeDur');
nTonePoke       =GetParam(me,'nTonePoke');
WPort_in2LastOut    =GetParam(me,'WPort_in2LastOut');
WPort_in2_2ndLastOut=GetParam(me,'WPort_in2_2ndLastOut');
OdorSchedule    =GetParam(me,'OdorSchedule');
StimParam       =GetParam(me,'StimParam','value');
param_string    =GetParam(me,'StimParam','user');
LeftRewardP     =str2double(StimParam(:,strcmp(param_string,'left reward ratio')));
RightRewardP    =str2double(StimParam(:,strcmp(param_string,'right reward ratio')));
LeftRewardP     =LeftRewardP(OdorSchedule((CountedTrial)));
RightRewardP    =RightRewardP(OdorSchedule((CountedTrial)));
Port_Side       =GetParam(me,'Port_Side');
Cue_Port_Side   =GetParam(me,'Cue_Port_Side');
if dd
    rHit =[35 8];
    lHit =[36 8];
    rAbort =[0 0];
    lAbort =[0 0];
    bAbort =[0 0];
    lFalse =[7 3;37 3];
    rFalse =[17 5;37 5];
    rMiss =[20 8];
    lMiss =[10 8];
    bMiss =[0 0];
    rEW =[0 0];
    lEW =[0 0];
    H2Ox3=[0 0];
    H2Ox2=[0 0];
elseif LeftRewardP==-1 && RightRewardP==-1
    rHit =[7 8];
    lHit =[0 0];
    rAbort =[59 8;60 8];
    lAbort =[0 0];
    bAbort =[0 0];
    lFalse =[4 3;5 3;6 3];
    rFalse =[4 5;5 5;6 5];
    rMiss =[0 0];
    lMiss =[0 0];
    rEW =[0 0];
    lEW =[0 0];
    H2Ox3=[0 0];
    H2Ox2=[0 0];
elseif Port_Side(CountedTrial)==0
    if  RightRewardP==0&& LeftRewardP==0
        rHit =[4 5;5 5;6 5];
        lHit =[4 3;5 3;6 3];
        rAbort =[0 0];
        lAbort =[0 0];
        bAbort =[79 8;80 8];
        lFalse =[0 0];
        rFalse =[0 0];
        rMiss =[0 0];
        lMiss =[0 0];
        bMiss =[4 8; 5 8; 6 8;55 8];
        rEW =[0 0];
        lEW =[0 0];
        H2Ox3=[0 0];
        H2Ox2=[0 0];
    elseif LeftRewardP==0
        rHit =[7 8;40 8;41 8];
        lHit =[0 0];
        rAbort =[59 8;60 8];
        lAbort =[0 0];
        bAbort =[0 0];
        lFalse =[4 3;5 3;6 3];
        rFalse =[0 0];
        rMiss  =[4 8; 5 8; 6 8;55 8];
        lMiss  =[0 0];
        rEW =[0 0];
        lEW =[0 0];
        H2Ox3=[0 0];
        H2Ox2=[0 0];
    elseif RightRewardP==0
        rHit =[0 0];
        lHit =[7 8;37 8;38 8];
        rAbort =[0 0];
        lAbort =[59 8;60 8];
        bAbort =[0 0];
        lFalse =[0 0];
        rFalse =[4 5;5 5;6 5];
        rMiss  =[0 0];
        lMiss  =[4 8; 5 8; 6 8;55 8];
        rEW =[0 0];
        lEW =[0 0];
        H2Ox3=[0 0];
        H2Ox2=[0 0];
    elseif Cue_Port_Side(CountedTrial)==1 %right/left/both
        rHit =[7 8;40 8;41 8];
        lHit =[0 0];
        rAbort =[59 8;60 8];
        lAbort =[0 0];
        bAbort =[0 0];
        lFalse =[4 3;5 3;6 3];
        rFalse =[0 0];
        rMiss  =[4 8; 5 8; 6 8;55 8];
        lMiss  =[0 0];
        rEW =[0 0];
        lEW =[0 0];
        H2Ox3=[0 0];
        H2Ox2=[0 0];
    elseif Cue_Port_Side(CountedTrial)==2 %right/left/both
        rHit =[0 0];
        lHit =[7 8;37 8;38 8];
        rAbort =[0 0];
        lAbort =[59 8;60 8];
        bAbort =[0 0];
        lFalse =[0 0];
        rFalse =[4 5;5 5;6 5];
        rMiss  =[0 0];
        lMiss  =[4 8; 5 8; 6 8;55 8];
        rEW =[0 0];
        lEW =[0 0];
        H2Ox3=[0 0];
        H2Ox2=[0 0];
    end
else
    rHit =[7 8];
    lHit =[17 8];
    rAbort =[59 8;60 8];
    lAbort =[69 8;70 8];
    bAbort =[79 8;80 8];
    lFalse =[4 3;5 3;6 3];
    rFalse =[14 5;15 5;16 5];
    rMiss =[ 4 8; 5 8; 6 8;55 8];
    lMiss =[14 8;15 8;16 8;54 8];
    bMiss =[24 8;25 8;26 8];
    rEW =[40 8;41 8];
    lEW =[37 8;38 8];
    H2Ox3=[43 8];
    H2Ox2=[44 8];
end

Event=GetParam('rpbox','event','user'); % [state,chan,event time]
for i=1:size(Event,1)
    if Event(i,2)==1        %tone poke in
        if Event(i,1:2)==[0 1] & (CountedTrial-1)
            nTonePoke(CountedTrial-1)=nTonePoke(CountedTrial-1)+nTonePoke(CountedTrial);
            nTonePoke(CountedTrial)=1;
        else
            nTonePoke(CountedTrial)=nTonePoke(CountedTrial)+1;
        end
        SetParam(me,'LastOdorPokeDur','user1',Event(i,3));

    elseif Event(i,2)==2    %tone poke out
        lastpkdur=(Event(i,3)-GetParam(me,'LastOdorPokeDur','user1'))*1000;
        SetParam(me,'LastOdorPokeDur','user2',Event(i,3));
        SetParam(me,'LastOdorPokeDur',round(lastpkdur*10)/10);
        if nTonePoke(CountedTrial)==1
            SetParam(me,'FirstOdorPokeDur',round(lastpkdur*10)/10);
            OdorPokeDur(CountedTrial)=lastpkdur;
            SetParam(me,'OdorPokeDur',OdorPokeDur);
        end
    elseif Event(i,2)==3    %Left poke in
        if sum(prod(double(repmat(Event(i,1:2),size(lFalse,1),1)==lFalse),2))
            Result(CountedTrial) =2;  % FalsePoke, wrong side
            Message(me,['Wrong side, FalsePoke #' num2str(GetParam(me,'LeftFalse')+1)],'red');
            SetParam(me,'LeftFalse',GetParam(me,'LeftFalse')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        end
        if ~WPort_in2LastOut(CountedTrial)&& nTonePoke(CountedTrial)
            SetParam(me,'WPort_in2LastOut','user',Event(i,3));
        end
    elseif Event(i,2)==4    %Left poke out
        if ~WPort_in2LastOut(CountedTrial) && ~nTonePoke(CountedTrial) && CountedTrial>1
            WPort_in2LastOut(CountedTrial-1)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        elseif ~WPort_in2LastOut(CountedTrial) && nTonePoke(CountedTrial)
            WPort_in2LastOut(CountedTrial)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        elseif WPort_in2LastOut(CountedTrial) && ~nTonePoke(CountedTrial) && CountedTrial>1
            WPort_in2_2ndLastOut(CountedTrial-1)=WPort_in2LastOut(CountedTrial);
            WPort_in2LastOut(CountedTrial-1)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        else
            WPort_in2_2ndLastOut(CountedTrial)=WPort_in2LastOut(CountedTrial);
            WPort_in2LastOut(CountedTrial)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        end

    elseif Event(i,2)==5    %Right poke in
        if sum(prod(double(repmat(Event(i,1:2),size(rFalse,1),1)==rFalse),2))
            Result(CountedTrial) =2;  % FalsePoke, wrong side
            Message(me,['Wrong side, FalsePoke #' num2str(GetParam(me,'RightFalse')+1)],'red');
            SetParam(me,'RightFalse',GetParam(me,'RightFalse')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        end
        if ~WPort_in2LastOut(CountedTrial)&& nTonePoke(CountedTrial)
            SetParam(me,'WPort_in2LastOut','user',Event(i,3));
        end
    elseif Event(i,2)==6    %Right poke out
        if ~WPort_in2LastOut(CountedTrial) && ~nTonePoke(CountedTrial) && CountedTrial>1
            WPort_in2LastOut(CountedTrial-1)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        elseif ~WPort_in2LastOut(CountedTrial) && nTonePoke(CountedTrial)
            WPort_in2LastOut(CountedTrial)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        elseif WPort_in2LastOut(CountedTrial) && ~nTonePoke(CountedTrial) && CountedTrial>1
            WPort_in2_2ndLastOut(CountedTrial-1)=WPort_in2LastOut(CountedTrial);
            WPort_in2LastOut(CountedTrial-1)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        else
            WPort_in2_2ndLastOut(CountedTrial)=WPort_in2LastOut(CountedTrial);
            WPort_in2LastOut(CountedTrial)=Event(i,3)-GetParam(me,'WPort_in2LastOut','user');
        end

    elseif Event(i,2)==8    % time up
        if sum(prod(double(repmat(Event(i,1:2),size(rHit,1),1)==rHit),2))
            Result(CountedTrial) =1.1;  % Hit
            Message(me,['Right Hit #' num2str(GetParam(me,'RightHit')+1)],'cyan');
            SetParam(me,'RightHit',GetParam(me,'RightHit')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(lHit,1),1)==lHit),2))
            Result(CountedTrial) =1.1;  % Hit
            Message(me,['Left Hit #' num2str(GetParam(me,'LeftHit')+1)],'cyan');
            SetParam(me,'LeftHit',GetParam(me,'LeftHit')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];

        elseif sum(prod(double(repmat(Event(i,1:2),size(rMiss,1),1)==rMiss),2))
            Result(CountedTrial) =3;  % ValidTonePoke but missed reward
            Message(me,'Missed reward');
            SetParam(me,'RightMiss',GetParam(me,'RightMiss')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(lMiss,1),1)==lMiss),2))
            Result(CountedTrial) =3;  % ValidTonePoke but missed reward
            Message(me,'Missed reward');
            SetParam(me,'LeftMiss',GetParam(me,'LeftMiss')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(bAbort,1),1)==bAbort),2))
            Result(CountedTrial) =3;  % ValidTonePoke but missed reward
            Message(me,'Missed reward');
            SetParam(me,'RightMiss',GetParam(me,'RightMiss')+.5);
            SetParam(me,'LeftMiss',GetParam(me,'LeftMiss')+.5);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(rEW,1),1)==rEW),2))
            Result(CountedTrial) =1.0;  % Valid response but withdraw before reward
            Message(me,'withdraw before reward');
            SetParam(me,'RightEarlyWithdraw',GetParam(me,'RightEarlyWithdraw')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(lEW,1),1)==lEW),2))
            Result(CountedTrial) =1.0;  % Valid response but withdraw before reward
            Message(me,'withdraw before reward');
            SetParam(me,'LeftEarlyWithdraw',GetParam(me,'LeftEarlyWithdraw')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(H2Ox3,1),1)==H2Ox3),2))
            Result(CountedTrial) =1.3;  % Hit streak>=4, 3 drops of H2O
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(H2Ox2,1),1)==H2Ox2),2))
            Result(CountedTrial) =1.2;  % Hit streak>=2, 2 drops of H2O
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(rAbort,1),1)==rAbort),2))
            Result(CountedTrial) =4;  % ShortPoke => Abort
            Message(me,['ShortPoke => RightAbort #' num2str(GetParam(me,'RightAbort')+1)],'green');
            SetParam(me,'RightAbort',GetParam(me,'RightAbort')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(lAbort,1),1)==lAbort),2))
            Result(CountedTrial) =4;  % ShortPoke => Abort
            Message(me,['ShortPoke => LeftAbort #' num2str(GetParam(me,'LeftAbort')+1)],'green');
            SetParam(me,'LeftAbort',GetParam(me,'LeftAbort')+1);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        elseif sum(prod(double(repmat(Event(i,1:2),size(bAbort,1),1)==bAbort),2))
            Result(CountedTrial) =4;  % ShortPoke => Abort
            Message(me,['ShortPoke => Abort #' num2str(GetParam(me,'LeftAbort')+GetParam(me,'RightAbort'))],'green');
            SetParam(me,'RightAbort',GetParam(me,'RightAbort')+.5);
            SetParam(me,'LeftAbort',GetParam(me,'LeftAbort')+.5);
            Trial_Events=[Trial_Events;Event(i,1:3)];
        end
    elseif Event(i,2)==7    % SchWV


    end
end
SetParam(me,'Result',Result);
SetParam(me,'nTonePoke',nTonePoke);
SetParam(me,'Trial_Events','value',Trial_Events);
Setparam('rpbox','event','user',[]);    %clearing events so it won't get counted twice
SetParam(me,'WPort_in2LastOut',WPort_in2LastOut);
SetParam(me,'WPort_in2_2ndLastOut',WPort_in2_2ndLastOut);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_plot
global exper

fig = findobj('tag',me,'type','figure');
% figure(fig);
axh = findobj(fig,'tag','plot_schedule');
if ~isempty(axh)
    if length(axh)==1
        axes(axh);
        set(axh,'pos',[0.1 0.36 0.85 0.26]);
        axh_flag=1;
    elseif length(axh)>=2
        delete(axh(2:length(axh)));
        axes(axh(1));
        set(axh(1),'pos',[0.1 0.36 0.85 0.26]);
        axh_flag=2;
    end
else
    axh = axes('tag','plot_schedule','pos',[0.1 0.36 0.85 0.26]);
    axh_flag=3;
end
axh=axh(1);
OdorSchedule=GetParam(me,'OdorSchedule');
Schedule=GetParam(me,'Schedule');
Port_Side=GetParam(me,'Port_Side');
Cue_Port_Side=GetParam(me,'Cue_Port_Side');
Result=GetParam(me,'Result');
Rewarded_Result=Result.*(Port_Side>0);
NonRewarded_Result=Result.*(Port_Side==0);
CountedTrial = GetParam(me,'CountedTrial');
MaxTrial = GetParam(me,'MaxTrial');
WPort_in2LastOut    =GetParam(me,'WPort_in2LastOut');
WPort_in2_2ndLastOut=GetParam(me,'WPort_in2_2ndLastOut');

future_trial_idx=[CountedTrial+1:MaxTrial];
right_future_trial_idx=future_trial_idx(Port_Side(future_trial_idx)==1);
if ~isempty(right_future_trial_idx)
    marker_h=plot(axh,right_future_trial_idx,Schedule(right_future_trial_idx),'c>');hold(axh,'on');
    set(marker_h,'MarkerSize',5,'MarkerFaceColor',[0 .7 1],'Color',[0 .7 1]);
end
left_future_trial_idx =future_trial_idx(Port_Side(future_trial_idx)==2);
if ~isempty(left_future_trial_idx)
    marker_h=plot(axh,left_future_trial_idx,Schedule(left_future_trial_idx),'c<');hold(axh,'on');
    set(marker_h,'MarkerSize',5,'MarkerFaceColor',[0 1 1]);
end
null_right_cue_future_trial_idx=future_trial_idx(Cue_Port_Side(future_trial_idx)==1 & Port_Side(future_trial_idx)==0);
null_left_cue_future_trial_idx =future_trial_idx(Cue_Port_Side(future_trial_idx)==2 & Port_Side(future_trial_idx)==0);
null_future_trial_idx=future_trial_idx(Cue_Port_Side(future_trial_idx)==0 & Port_Side(future_trial_idx)==0);
if ~isempty(null_future_trial_idx)
    marker_h=plot(axh,null_future_trial_idx,Schedule(null_future_trial_idx),'ks');hold(axh,'on');
    set(marker_h,'MarkerSize',3,'MarkerFaceColor',[0 0 0]);
end
if ~isempty(null_right_cue_future_trial_idx)
    marker_h=plot(axh,null_right_cue_future_trial_idx,Schedule(null_right_cue_future_trial_idx),'k>');hold(axh,'on');
    set(marker_h,'MarkerSize',5,'MarkerFaceColor',[0 0 0]);
end
if ~isempty(null_left_cue_future_trial_idx)
    marker_h=plot(axh,null_left_cue_future_trial_idx,Schedule(null_left_cue_future_trial_idx),'k<');hold(axh,'on');
    set(marker_h,'MarkerSize',5,'MarkerFaceColor',[0 0 0]);
end

Hit3=find(Rewarded_Result==1.3);
if Hit3
    plot(axh,Hit3,Schedule(Hit3),'bd');
end
Hit2=find(Rewarded_Result==1.2);
if Hit2
    plot(axh,Hit2,Schedule(Hit2),'b+');
end
Hit=find(Rewarded_Result==1.1);
if Hit
    plot(axh,Hit,Schedule(Hit),'b.');
end
NonRewarded_Hit=find(NonRewarded_Result==1.1);
if NonRewarded_Hit
    marker_h=plot(axh,NonRewarded_Hit,Schedule(NonRewarded_Hit),'bs');
    set(marker_h,'MarkerSize',3,'MarkerFaceColor',[0 0 1]);
end
EW=find(Rewarded_Result==1.0);
if EW
    plot(axh,EW,Schedule(EW),'bh');
end
false=find(Rewarded_Result==2);
if false
    plot(axh,false,Schedule(false),'r.');
end
NonRewarded_false=find(NonRewarded_Result==2);
if NonRewarded_false
    marker_h=plot(axh,NonRewarded_false,Schedule(NonRewarded_false),'rs');
    set(marker_h,'MarkerSize',3,'MarkerFaceColor',[1 0 0]);
end

miss=find(Result==3);
if miss
    plot(axh,miss,Schedule(miss),'bo');
end
abort=find(Result==4);
if abort
    plot(axh,abort,Schedule(abort),'g.');
end
right_Hit_trials  =(Port_Side(1:CountedTrial)==1 & ismember(Rewarded_Result(1:CountedTrial),[1.1 1.2 1.3]));
SetParam(me,'rightH2OReward',sum((Rewarded_Result(right_Hit_trials)-1)*10));
left_Hit_trials  =(Port_Side(1:CountedTrial)==2 & ismember(Rewarded_Result(1:CountedTrial),[1.1 1.2 1.3]));
SetParam(me,'leftH2OReward',sum((Rewarded_Result(left_Hit_trials)-1)*10));

plot(axh,CountedTrial+1,Schedule(CountedTrial+1),'or');
Str=GetParam(me,'Stim_Disp','user');
SetParam(me,'Stim_Disp',Str{OdorSchedule(CountedTrial+1)});
ax = axis;
hold(axh,'off');

PlotAxes_Back   =GetParam(me,'PlotAxes_Back');
PlotAxes_Forward=GetParam(me,'PlotAxes_Forward');
axis(axh,[min(max(ceil((CountedTrial-90-PlotAxes_Back)/50)*50,0),max((MaxTrial-100-PlotAxes_Back),0)) ...
    min(max(ceil((CountedTrial+10+PlotAxes_Forward)/50)*50,100),MaxTrial) ax(3)-.3 ax(4)+.8]);
xlabel(axh,'Counted Trial');
ylabel(axh,'Stim. #');

StimParam=GetParam(me,'StimParam');
param_string=GetParam(me,'StimParam','user');
Ratio_cell = StimParam(:,strcmp(param_string,'stimulus probability'));
Ratio=zeros(1,length(Ratio_cell));
for i=1:length(Ratio_cell)
    Ratio(i)=str2double(Ratio_cell{i});
end
Ratio=Ratio/sum(Ratio);

YTickLabel=[];
for i=1:length(Ratio)
    YTickLabel{i}=sprintf('%d(%2.0f%%)',i,Ratio(i)*100 );
end
set(axh,'YTick', [1:i],'YTickLabel',YTickLabel);
set(axh,'tag','plot_schedule');

%%%%%%%%%%%%%%%%% plot performance %%%%%%%%%%%%%%%%
ax2h = findobj(fig,'tag','plot_performance');
if ~isempty(ax2h)
    if length(ax2h)==1
        axes(ax2h);
        set(ax2h,'pos',[0.10 0.72 0.4 0.25]);
        ax2h_flag=1;
    elseif length(ax2h)>=2
        delete(ax2h(2:length(ax2h)));
        axes(ax2h(1));
        set(ax2h(1),'pos',[0.10 0.72 0.4 0.25]);
        ax2h_flag=2;
    end
else
    ax2h = axes('tag','plot_performance','pos',[0.10 0.72 0.4 0.25]);
    ax2h_flag=3;
end
ax2h=ax2h(1);
Valid_Performance=[];
if CountedTrial
    Performance=zeros(size(Ratio));
    Valid_Performance=Performance;
    Miss_Performance=Performance;
    False_Performance=Performance;
    Abort_Performance=Performance;
    EarlyWithdraw    =Performance;
    n_Hit=Performance;
    n_EW =Performance;
    n_Fls=Performance;
    n_Mis=Performance;
    n_Abt=Performance;
    n_trial=Performance;
    for i=1:length(Ratio)
        trial_idx=find(Schedule(1:CountedTrial)==i);
        n_trial(i)=length(trial_idx);
        n_Hit(i)=size(find(floor(Result(trial_idx))==1),2);
        n_EW(i) =size(find(Result(trial_idx)==1.0),2);
        n_Fls(i)=size(find(Result(trial_idx)==2),2);
        n_Mis(i)=size(find(Result(trial_idx)==3),2);
        n_Abt(i)=size(find(Result(trial_idx)==4),2);
        if n_trial(i)==0
            Performance(i)=NaN;
            False_Performance(i)=NaN;
            Miss_Performance(i)=NaN;
            Abort_Performance(i)=NaN;
        else
            Performance(i)=n_Hit(i)/n_trial(i);
            False_Performance(i)=n_Fls(i)/ n_trial(i);
            Miss_Performance(i)=n_Mis(i)/ n_trial(i);
            Abort_Performance(i)=n_Abt(i)/ n_trial(i);
            EarlyWithdraw(i)=n_EW(i)/ n_trial(i);
        end
        if (n_trial(i)-n_Abt(i))==0
            Valid_Performance(i)=NaN;
        else
            Valid_Performance(i)=n_Hit(i)/ (n_Hit(i)+n_Fls(i)+n_Mis(i));
        end
    end
    x=1:length(Ratio);
    plot(ax2h,x,Performance,'b*',x,EarlyWithdraw,'bh',x,Valid_Performance,'c-',x,Miss_Performance,'bo',x,Abort_Performance,'g.',x,False_Performance,'r.');
end

axis(ax2h,[0.5 length(Ratio)+.5 0 1]);
set(ax2h,'XTick',[1:1:length(Ratio)],'XTickLabel',[1:1:length(Ratio)]);
xlabel(ax2h,[ sprintf('%6.2g',Valid_Performance)  sprintf('\n') 'Stim #']);
ylabel(ax2h,['Performance' sprintf('\n') 'Fraction correct']);
set(ax2h,'tag','plot_performance');

% %%%%%%%%%%%%%%%%% plot joint performance %%%%%%%%%%%%%%%%
ax3h = findobj(fig,'tag','plot_jnt_performance');
if ~isempty(ax3h)
    if length(ax3h)==1
        axes(ax3h);
        set(ax3h,'pos',[0.6 0.72 0.35 0.25]);
        ax3h_flag=1;
    elseif length(ax3h)>=2
        delete(ax3h(2:length(ax3h)));
        axes(ax3h(1));
        set(ax3h(1),'pos',[0.6 0.72 0.35 0.25]);
        ax3h_flag=2;
    end
else
    ax3h = axes('tag','plot_jnt_performance','pos',[0.6 0.72 0.35 0.25]);
    ax3h_flag=3;
end
ax3h=ax3h(1);cla(ax3h);
if CountedTrial>GetParam(me,'blocklength') && ismember(GetParam(me,'Script'),[2 3 4 5 6 7 8 9 14 15])
    Result=[NaN GetParam(me,'Result')];
    presumed_Port_Side=Port_Side(1:CountedTrial)*0;
    for i=1:CountedTrial
        if Port_Side(i)>0
            presumed_Port_Side(i)=Port_Side(i);
        elseif Port_Side(i)==0 && i>1
            presumed_Port_Side(i)=presumed_Port_Side(i-1);
        end
    end
    switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&presumed_Port_Side(1:CountedTrial-1)==1&ismember(Schedule(2:CountedTrial),[9 11]))+1;
%     cued_switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&presumed_Port_Side(1:CountedTrial-1)==1&ismember(Schedule(2:CountedTrial),[10 12]))+1;
    if ~isempty(switch_idx)
        aligned_sw_performance_idx=(meshgrid(-10:10,1:length(switch_idx))+repmat(switch_idx',1,21))+1;
        aligned_sw_performance_idx(aligned_sw_performance_idx<1)=1;
        switch_performance=floor(Result(aligned_sw_performance_idx));
        switch_performance(switch_performance==0)=NaN;
        switch_performance(switch_performance>1)=0;
        plot(ax3h,1:10,nanmean(switch_performance(:,1:10),1),'r>-.');hold(ax3h,'on');
        plot(ax3h,11:21,nanmean(switch_performance(:,11:21),1),'b<-.');
    end

    switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&presumed_Port_Side(1:CountedTrial-1)==2&ismember(Schedule(2:CountedTrial),[9 11]))+1;
%     cued_switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&presumed_Port_Side(1:CountedTrial-1)==2&ismember(Schedule(2:CountedTrial),[10 12]))+1;
    if ~isempty(switch_idx)
        aligned_sw_performance_idx=(meshgrid(-10:10,1:length(switch_idx))+repmat(switch_idx',1,21))+1;
        aligned_sw_performance_idx(aligned_sw_performance_idx<1)=1;
        switch_performance=floor(Result(aligned_sw_performance_idx));
        switch_performance(switch_performance==0)=NaN;
        switch_performance(switch_performance>1)=0;
        plot(ax3h,1:10,nanmean(switch_performance(:,1:10),1),'b<-.');hold(ax3h,'on');
        plot(ax3h,11:21,nanmean(switch_performance(:,11:21),1),'r>-.');
    end

    switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&ismember(Schedule(2:CountedTrial),[9 11]))+1;
    cued_switch_idx=find(diff(presumed_Port_Side(1:CountedTrial))&ismember(Schedule(2:CountedTrial),[10 12]))+1;
    if ~isempty(switch_idx)
        aligned_sw_performance_idx=(meshgrid(-10:10,1:length(switch_idx))+repmat(switch_idx',1,21))+1;
        aligned_sw_performance_idx(aligned_sw_performance_idx<1)=1;
        switch_performance=floor(Result(aligned_sw_performance_idx));
        switch_performance(switch_performance==0)=NaN;
        switch_performance(switch_performance>1)=0;
        plot(ax3h,nanmean(switch_performance,1),'k');hold(ax3h,'on');
        axis(ax3h,[1 21 0 1]);
        set(ax3h,'XTick',[1:21],'XTickLabel',[-10:1:10]);
        xlabel(ax3h,'Trial relative to imposed switch');
        ylabel(ax3h,'faction correct');
    end
    trial2sw=(1:length(switch_idx)-1)*nan;
    for k=1:length(trial2sw)
        sw_trial=find(floor(Result(switch_idx(k)+1:switch_idx(k+1)))==1,1,'first');
        if ~isempty(sw_trial)
            trial2sw(k)=sw_trial;
        else
            trial2sw(k)=switch_idx(k+1)-switch_idx(k);
        end
    end
    cued_trial2sw=(1:length(cued_switch_idx)-1)*nan;
    for k=1:length(cued_trial2sw)
        sw_trial=find(floor(Result(cued_switch_idx(k)+1:cued_switch_idx(k+1)))==1,1,'first');
        if ~isempty(sw_trial)
            cued_trial2sw(k)=sw_trial;
        else
            cued_trial2sw(k)=cued_switch_idx(k+1)-cued_switch_idx(k);
        end
    end
    text(2,.2,['trials to sw:' num2str(mean(trial2sw)-1) sprintf('\n') 'cued\_trials2sw:' num2str(mean(cued_trial2sw)-1)]);
    hold(ax3h,'off');
elseif CountedTrial>GetParam(me,'blocklength') && ismember(GetParam(me,'Script'),[10 11 12 13])
    naCountedTrial=CountedTrial-sum(Result(1:CountedTrial)==4);
    naResult=Result(Result~=4);
    naSchedule=Schedule(Result~=4);
    l1r1_idx=[0 find(ismember(naSchedule(1:naCountedTrial),[13 14]))];
    l1r1_swi=l1r1_idx([1==0 diff(l1r1_idx)>1]);
    if ~isempty(l1r1_swi)
        aligned_sw_performance_idx=(meshgrid(-10:10,1:length(l1r1_swi))+repmat(l1r1_swi',1,21));
        aligned_sw_performance_idx(aligned_sw_performance_idx<1)=1;
        switch_performance=floor(naResult(aligned_sw_performance_idx));
        switch_performance(switch_performance==0)=NaN;
        switch_performance(switch_performance>1)=0;
        plot(ax3h,1:11,nanmean(switch_performance(:,1:11),1),'r-');hold(ax3h,'on');
        plot(ax3h,11:21,nanmean(switch_performance(:,11:21),1),'b-');
    end

    lnrn_idx=[0 find(ismember(naSchedule(1:naCountedTrial),[9 11]))];
    lnrn_swi=lnrn_idx([1==0 diff(lnrn_idx)>1]);
    if ~isempty(lnrn_swi)
        aligned_sw_performance_idx=(meshgrid(-10:10,1:length(lnrn_swi))+repmat(lnrn_swi',1,21))+1;
        aligned_sw_performance_idx(aligned_sw_performance_idx<1)=1;
        switch_performance=floor(naResult(aligned_sw_performance_idx));
        switch_performance(switch_performance==0)=NaN;
        switch_performance(switch_performance>1)=0;
        plot(ax3h,1:11,nanmean(switch_performance(:,1:11),1),'b-');hold(ax3h,'on');
        plot(ax3h,11:21,nanmean(switch_performance(:,11:21),1),'r-');
    end
    axis(ax3h,[1 21 0 1]);
    set(ax3h,'XTick',[1:21],'XTickLabel',[-10:1:10]);
    if ~isempty(l1r1_swi)&~isempty(lnrn_swi)
        if l1r1_swi(end)<lnrn_swi(end)
            l1r1_swi=[l1r1_swi lnrn_swi(end)];
        elseif l1r1_swi(end)>lnrn_swi(end)
            lnrn_swi=[lnrn_swi l1r1_swi(end)];
        end
    end
    trial2sw2l1r1=(1:length(l1r1_swi)-1)*nan;
    for k=1:length(trial2sw2l1r1)
        sw_trial=find(floor(naResult(l1r1_swi(k)+1:l1r1_swi(k+1)))==1 &...
            [0==1 diff(floor(naResult(l1r1_swi(k)+1:l1r1_swi(k+1)))==1)==0],1,'first');
        if ~isempty(sw_trial)
            trial2sw2l1r1(k)=sw_trial;
        else
            trial2sw2l1r1(k)=l1r1_swi(k+1)-l1r1_swi(k);
        end
    end

    trial2sw2lnrn=(1:length(lnrn_swi)-1)*nan;
    for k=1:length(trial2sw2lnrn)
        sw_trial=find(floor(naResult(lnrn_swi(k)+1:lnrn_swi(k+1)))==1 &...
            [0==1 diff(floor(naResult(lnrn_swi(k)+1:lnrn_swi(k+1)))==1)==0],1,'first');
        if ~isempty(sw_trial)
            trial2sw2lnrn(k)=sw_trial;
        else
            trial2sw2lnrn(k)=lnrn_swi(k+1)-lnrn_swi(k);
        end
    end
    text(2,.2,['trials to L1R1 sw:' num2str(mean(trial2sw2l1r1)-1) sprintf('\n') 'trials to LnRn sw:' num2str(mean(trial2sw2lnrn)-1)]);
    hold(ax3h,'off');

elseif CountedTrial
    Waiting_Time=zeros(size(Ratio));
    correct_trial_Waiting_Time=Waiting_Time;
    error_trial_Waiting_Time=Waiting_Time;
    for i=1:length(Ratio)
        correct_trial_idx=(Schedule(1:CountedTrial)==i & Result(1:CountedTrial)==1.0);
        if sum(correct_trial_idx)==0
            correct_trial_Waiting_Time(i)=NaN;
        else
            correct_trial_Waiting_Time(i)=mean(WPort_in2LastOut(correct_trial_idx));
        end
        error_trial_idx=(Schedule(1:CountedTrial)==i & Result(1:CountedTrial)==2);
        if sum(error_trial_idx)==0
            error_trial_Waiting_Time(i)=NaN;
        else
            error_trial_Waiting_Time(i)=mean(WPort_in2LastOut(error_trial_idx));
        end
    end
    x=1:length(Ratio);
    plot(ax3h,x,error_trial_Waiting_Time,'r.',x,correct_trial_Waiting_Time,'b*');
    ax=axis;
    axis(ax3h,[0.5 length(Ratio)+.5 0 max(ax(4),0.1)]);
    set(ax3h,'XTick',[1:1:length(Ratio)],'XTickLabel',[1:1:length(Ratio)]);
    % xlabel([ sprintf('%6.2g',Valid_Performance)  sprintf('\n') 'Stim #']);
    ylabel(ax3h,['reward waiting time' sprintf('\n') '(sec)']);
end

set(ax3h,'tag','plot_jnt_performance');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function change_schedule(varargin)
a=clock;
rand('state', ceil(a(end)));

CountedTrial = GetParam(me,'CountedTrial')+GetParam('rpbox','run');
Schedule     = GetParam(me,'Schedule');
OdorSchedule = GetParam(me,'OdorSchedule');
Port_Side    = GetParam(me,'Port_Side');
Cue_Port_Side= GetParam(me,'Cue_Port_Side');
VP_LED       = GetParam(me,'VP_LED');
OdorChannel  = GetParam(me,'OdorChannel');
OdorName     = GetParam(me,'OdorName');

StimParam   = GetParam(me,'StimParam');
param_string=GetParam(me,'StimParam','user');
OdrCh_List=str2double(StimParam(:,strcmp(param_string,'Dout Channel')));
OdrNm_List=StimParam(:,strcmp(param_string,'Odor Name'));
vpled_List=StimParam(:,strcmp(param_string,'VP LED cue'));

Ratio_cell = StimParam(:,strcmp(param_string,'stimulus probability'));
Ratio=zeros(1,length(Ratio_cell));
for i=1:length(Ratio_cell)
    Ratio(i)=str2double(Ratio_cell{i});
end
Ratio=Ratio/sum(Ratio);

MaxTrial = GetParam(me,'MaxTrial');

Cum_Ratio =[0 cumsum(Ratio)/sum(Ratio)];

last_port_side=0;
same_side_cont=0;
same_side_limit=GetParam(me,'SameSideLimit');
LeftRewardP  =str2double(StimParam(:,strcmp(param_string,'left reward ratio')));
RightRewardP =str2double(StimParam(:,strcmp(param_string,'right reward ratio')));
Message(me,'');
if (2^(-same_side_limit))> min(Ratio*[RightRewardP LeftRewardP])
    Message(me,'same side limit will NOT be applied');
end
for i = CountedTrial+1 : MaxTrial
    random_num = rand;
    for j = 1:length(Ratio)
        if Cum_Ratio(j) <= random_num && random_num < Cum_Ratio(j+1)
            chan=j;
        break
        end
    end

    if sum([RightRewardP(chan) LeftRewardP(chan)])>1
        LeftRewardP(chan)  =LeftRewardP(chan)/sum([RightRewardP(chan) LeftRewardP(chan)]);
        RightRewardP(chan) =RightRewardP(chan)/sum([RightRewardP(chan) LeftRewardP(chan)]);
    end
    Cum_RewardP  =[0 cumsum([RightRewardP(chan) LeftRewardP(chan)])];
    random_num = rand;
    Port_Side(i)=0;
    Cue_Port_Side(i)=0;
    if random_num < Cum_RewardP(3)
        for j = 1:2
            if Cum_RewardP(j) <= random_num && random_num < Cum_RewardP(j+1)
                Port_Side(i)=j;
                Cue_Port_Side(i)=j;
            break
            end
        end
    else
        Cum_CueP = [Cum_RewardP(3) Cum_RewardP(3)+cumsum([RightRewardP(chan) LeftRewardP(chan)])/(RightRewardP(chan)+LeftRewardP(chan))*(1-Cum_RewardP(3))];
        for j = 1:2
            if Cum_CueP(j) <= random_num && random_num < Cum_CueP(j+1)
                Cue_Port_Side(i)=j;
            break
            end
        end
    end

    if last_port_side==Port_Side(i)
        same_side_cont=same_side_cont+1;
    else
        same_side_cont=0;
    end
    while (same_side_cont+1 > same_side_limit)&& (2^(-same_side_limit))<min(Ratio*[RightRewardP LeftRewardP])
        random_num = rand;
        for j = 1:length(Ratio)
            if Cum_Ratio(j) <= random_num && random_num < Cum_Ratio(j+1)
                chan=j;
            break
            end
        end
        if sum([RightRewardP(chan) LeftRewardP(chan)])>1
            LeftRewardP(chan)  =LeftRewardP(chan)/sum([RightRewardP(chan) LeftRewardP(chan)]);
            RightRewardP(chan) =RightRewardP(chan)/sum([RightRewardP(chan) LeftRewardP(chan)]);
        end
        Cum_RewardP  =[0 cumsum([RightRewardP(chan) LeftRewardP(chan)])];
        random_num = rand;
        Port_Side(i)=0;
        Cue_Port_Side(i)=0;
        if random_num < Cum_RewardP(3)
            for j = 1:2
                if Cum_RewardP(j) <= random_num && random_num < Cum_RewardP(j+1)
                    Port_Side(i)=j;
                    Cue_Port_Side(i)=j;
                break
                end
            end
        else
            Cum_CueP = [Cum_RewardP(3) Cum_RewardP(3)+cumsum([RightRewardP(chan) LeftRewardP(chan)])/(RightRewardP(chan)+LeftRewardP(chan))*(1-Cum_RewardP(3))];
            for j = 1:2
                if Cum_CueP(j) <= random_num && random_num < Cum_CueP(j+1)
                    Cue_Port_Side(i)=j;
                break
                end
            end
        end
        if last_port_side==Port_Side(i)
            same_side_cont=same_side_cont+1;
        else
            same_side_cont=0;
        end
        if same_side_cont > MaxTrial
            Message(me,'SameSideLimit ignored','error');
            SetParam(me,'SameSideLimit',inf);
            same_side_limit=inf;
        end
    end
    Schedule(i)=chan;
    OdorSchedule(i)=chan;
    OdorChannel(i)=OdrCh_List(chan);
    OdorName(i) =double(OdrNm_List{chan});
    VP_LED(i)=str2double(vpled_List{chan});
    last_port_side=Port_Side(i);
end

SetParam(me,'Schedule','value',Schedule);
SetParam(me,'Port_Side','value',Port_Side);
SetParam(me,'Cue_Port_Side','value',Cue_Port_Side);
SetParam(me,'VP_LED','value',VP_LED);
SetParam(me,'OdorChannel','value',OdorChannel);
SetParam(me,'OdorName','value',OdorName);
SetParam(me,'OdorSchedule','value',OdorSchedule);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% StimEditGUI
%		Select characteristics of the filter to create.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function StimEditGUI
toclose=findobj('Type','figure','Name','Stimulus Parameters','NumberTitle','off','Menu','None','File',me);
close(toclose);
stimparamfig=figure('Name','Stimulus Parameters','NumberTitle','off','Menu','None','File',me);
mainfig=findobj('Type','figure','tag',me);
main_fig_pos=get(mainfig,'pos');
thispos=get(stimparamfig,'Position');
gui_h=thispos(4)*3/4;
gui_w=thispos(3);
set(stimparamfig,'Position',[main_fig_pos(1)-30 sum(main_fig_pos([2 4]))-20-gui_h gui_w gui_h]);
StimParam=GetParam(me,'StimParam');
param_string=GetParam(me,'StimParam','user');
winstep=1/(size(StimParam,1)+4);
winheight=1/(size(StimParam,1)+5);

uicontrol(stimparamfig,'Style','text','Units','normal','FontWeight','bold','tag','stimulus param description','Position',[0 (size(StimParam,1)+3)*winstep 1 winstep], ...
    'String','Stimulus Parameters');

% Stimulus Number
uicontrol(stimparamfig,'Style','text','Units','normal','tag','stimulus number', ...
    'Position',[0 (size(StimParam,1)+1)*winstep .03 winheight],'String','#');
% Odor Name
uicontrol(stimparamfig,'Style','text','Units','normal','tag','Odor Name',...
    'Position',[.03 (size(StimParam,1)+1)*winstep .08 winheight*2], ...
    'String','Odor Name');
% Digital Out channel
uicontrol(stimparamfig,'Style','text','Units','normal','tag','Dout Channel',...
    'Position',[.115 (size(StimParam,1)+1)*winstep .08 winheight*2], ...
    'String','Dout Channel');
% Reward ratio
uicontrol(stimparamfig,'Style','text','Units','normal','tag','reward ratio',...
    'Position',[.2 (size(StimParam,1)+1)*winstep .17 winheight*2.4], ...
    'String','L-reward ratio-R (Between 0~1, No-Go: both= -1)');
% VP LED cue
uicontrol(stimparamfig,'Style','text','Units','normal','tag','VP LED cue',...
    'Position',[.37 (size(StimParam,1)+1)*winstep .125 winheight*2.4], ...
    'String','VP LED cue (0:no cue, 1:cue, 2:both)');
% Stimulus Duration
uicontrol(stimparamfig,'Style','text','Units','normal','tag','stimulus duration',...
    'Position',[.495 (size(StimParam,1)+1)*winstep .1 winheight*2], ...
    'String','    Duration    (sec)');
% Stimulus Probability
uicontrol(stimparamfig,'Style','text','Units','normal','tag','stimulus probability',...
    'Position',[.6 (size(StimParam,1)+1)*winstep .13 winheight*2], ...
    'String','    Probability    (% of trials)');
% Stimulus Name
uicontrol(stimparamfig,'Style','text','Units','normal','tag','stimulus name',...
    'Position',[.73 (size(StimParam,1)+1)*winstep .26 winheight], ...
    'String','stimulus name');

for i=1:size(StimParam,1)
    i_str=num2str(i);
    uicontrol(stimparamfig,'Style','text','Units','normal','tag',['stimulus number' i_str ' '], ...
        'Position',[0 (size(StimParam,1)+1-i)*winstep .03 winheight],'String',i_str);
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{1} i_str ' '], ...
        'Position',[.03 (size(StimParam,1)+1-i)*winstep .085 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,1},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{2} i_str ' '],...
        'Position',[.115 (size(StimParam,1)+1-i)*winstep .08 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,2},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{3} i_str ' '],...
        'Position',[.2 (size(StimParam,1)+1-i)*winstep .082 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,3},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{4} i_str ' '],...
        'Position',[.283 (size(StimParam,1)+1-i)*winstep .082 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,4},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{5} i_str ' '],...
        'Position',[.37 (size(StimParam,1)+1-i)*winstep .13 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,5},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{6} i_str ' '],...
        'Position',[.5 (size(StimParam,1)+1-i)*winstep .1 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,6},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{7} i_str ' '],...
        'Position',[.6 (size(StimParam,1)+1-i)*winstep .13 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,7},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
    uicontrol(stimparamfig,'Style','edit','Units','normal','tag',[param_string{8} i_str ' '],...
        'Position',[.73 (size(StimParam,1)+1-i)*winstep .30 winheight],'BackgroundColor',[1 1 1],...
        'String',StimParam{i,8},'callback','set(gcbo,''BackgroundColor'',[1 0 0]);');
end

uicontrol(stimparamfig,'Style','push','Units','normal','tag','Apply_stimparam_change',...
    'Position',[.04 (size(StimParam,1)-i)*winstep .15 winheight], ...
    'String','Apply change','Callback',[me ' SetStimParams;']);


% Close button
uicontrol(stimparamfig,'Style','push','Units','normal','tag','close button', ...
    'Position',[.24 (size(StimParam,1)-i)*winstep .15 winheight], ...
    'String','Close','Callback','close(gcf);');
%
% % Make the background agree.
% h=findobj(stimparamfig, 'Tag','close button');
% bgcol=get(h, 'BackgroundColor');
% set(stimparamfig,'Color',bgcol);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SetStimParams
%		Set filter parameters from the GUI values entered.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function StimParam=SetStimParams
% [ud mainfig]=getUserData;
paramfig=findobj('Type','figure','Name','Stimulus Parameters','NumberTitle','off','Menu','None','File',me);

StimParam=GetParam(me,'StimParam');
param_string=GetParam(me,'StimParam','user');
Ratio=[];
stim_prob_h=[];
for i=1:size(StimParam,1)
    i_str=num2str(i);
    lNoGo_flag=0;
    rNoGo_flag=0;
    for j=1:length(param_string)
        h=findobj(paramfig,'tag',[param_string{j} i_str ' ']);
        set(h,'BackgroundColor',[1 1 1]);
        if strcmp(param_string{j},'left reward ratio')
            lh=h;
            lij=[i j];
            if str2double(get(h,'String'))==-1
                lNoGo_flag=1;
                left_reward_ratio=str2double(StimParam{i,j});
                left_reward_ratio=left_reward_ratio*(left_reward_ratio>0);
            elseif str2double(get(h,'String'))>=0
                left_reward_ratio=str2double(get(h,'String'));
            else
                left_reward_ratio=str2double(StimParam{i,j});
            end
        end
        if strcmp(param_string{j},'right reward ratio')
            rh=h;
            rij=[i j];
            if str2double(get(h,'String'))==-1
                rNoGo_flag=1;
                right_reward_ratio=str2double(StimParam{i,j});
                right_reward_ratio=right_reward_ratio*(right_reward_ratio>0);
            elseif str2double(get(h,'String'))>=0
                right_reward_ratio=str2double(get(h,'String'));
            else
                right_reward_ratio=str2double(StimParam{i,j});
            end
        end
        if strcmp(param_string{j},'stimulus probability')
            jj=j;
            stim_prob_h(i)=h;
            Ratio(i)=str2num(get(h,'String'));
        end
        StimParam{i,j}=get(h,'String');
    end
    if lNoGo_flag==1 && rNoGo_flag==1
        reward_ratio= [-1 -1];
    elseif (left_reward_ratio + right_reward_ratio)>1
        reward_ratio= round([left_reward_ratio right_reward_ratio]/sum([left_reward_ratio right_reward_ratio])*100)/100;
    else
        reward_ratio= round([left_reward_ratio right_reward_ratio]*100)/100;
    end
    StimParam{lij(1),lij(2)}=num2str(reward_ratio(1));
    StimParam{rij(1),rij(2)}=num2str(reward_ratio(2));
    set(lh,'string',num2str(reward_ratio(1)));
    set(rh,'string',num2str(reward_ratio(2)));
end
Ratio=round(Ratio/sum(Ratio)*100)/100;
for i=1:size(StimParam,1)
    StimParam(i,jj)={num2str(Ratio(i))};
    set(stim_prob_h(i),'string',num2str(Ratio(i)));
end
SetParam(me,'StimParam',StimParam);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=str_sep(varargin)
% out=str_sep(input,separation_sign)
% str_sep is a function that separate the input string into several
% strings based on the separation sign(s) (in a single string), the
% default separation sign is space.
% 08/25/2002 Lung-Hao Tai
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out = '';
if nargin > 1
    sep_sign=varargin{2};
    str=varargin{1};
elseif nargin > 0
    sep_sign=char(32);
    str=varargin{1};
else
    disp('Please specify input string');
    eval('help str_sep');
end

if (iscell(str) && ~isa(str{1},'char')) || (~isa(str,'char')&& ~iscell(str))
    disp('Input should be string or cell array of string');
    eval('help str_sep');
    return;
elseif ~isa(sep_sign,'char')
    disp('Separartion sign should be string');
    eval('help str_sep');
    return
end

if iscell(str)
    out=cell(length(str),1);
elseif isa(str,'char')
    str={str};
    out={''};
end

for cnt=1:length(out)
    str_sz=length(str{cnt});
    starts=zeros(1,str_sz);
    ends=zeros(1,str_sz);
    strs=ones(1,str_sz);
    for i=1:length(sep_sign)
        strs(find(str{cnt}==sep_sign(i)))=0;
    end

    %   find string starting point
    for i=1:str_sz
        if i==1 && strs(i)==1
            starts(i)=1;
        elseif i>1 && strs(i-1)==0 && strs(i)==1
            starts(i)=1;
        end
    end

    %   find string end point
    for i=str_sz:-1:1
        if i==str_sz && strs(i)==1
            ends(i)=1;
        elseif i<str_sz && strs(i+1)==0 && strs(i)==1
            ends(i)=1;
        end
    end

    start_ind=find(starts);
    end_ind=find(ends);
    str_ind=[start_ind; end_ind];

    % update index list
    n_ind=length(start_ind);
    for i=1:n_ind
        out{cnt,i}=str{cnt}(start_ind(i):end_ind(i));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Seting_Str=Load_Seting_Strs
datapath=GetParam(me,'datapath');
setting_files=dir([datapath filesep me '_load_*_settings.mat']);
Seting_Str{1}='none';
if ~isempty(setting_files)
    for i=1:length(setting_files)
        str=str_sep(setting_files(i).name(strfind(setting_files(i).name,'load'):end),'_');
        Seting_Str{i+1}=str{2};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=me
% Simple function for getting the name of this m-file.
out=lower(mfilename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = callback
out = [lower(mfilename) ';'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ud mainfig ]=getUserData
mainfig=findobj('Type','figure','tag',me);
ud=get(mainfig,'UserData');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=InitVP_Sound
RPFs = get_generic('sampling_rate');
[s, Fs]=wavread('XPding88200.wav');
%normalize s
s=s-mean(s);
% s=s/max(abs(s))/2;
[nrat,drat]=rat( RPFs / Fs );
sRPS=resample( s, nrat, drat );
vp_sound{1}=sRPS'/20;
out=vp_sound;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function restore_event(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin
    trial_events=varargin{1};
else
    trial_events=GetParam('rpbox','trial_events');
end

CountedTrial    =GetParam(me,'CountedTrial');
% dd              =GetParam(me,'DirectDelivery');
% Trial_Events    =GetParam(me,'Trial_Events','value');
% nTonePoke       =GetParam(me,'nTonePoke');
% Port_Side       =GetParam(me,'Port_Side');
Result          =GetParam(me,'Result');
missing_result_idx=find(Result(1:GetParam(me,'countedtrial'))==0);

end_of_trial_ind=trial_events(:,3)==512;
end_of_trial_ind=find([diff(end_of_trial_ind);0]& end_of_trial_ind);

for i=1:length(end_of_trial_ind)
    if i==1
        begin_of_trial_ind=1;
    else
        begin_of_trial_ind=end_of_trial_ind(i-1)+1;
    end
    Event=trial_events(begin_of_trial_ind:end_of_trial_ind(i),[3 4 2 5]); % [state,chan,event time, new state]
    SetParam(me,'CountedTrial',i-1);
    Setparam('rpbox','event','user',Event);
    update_event;
end
SetParam(me,'CountedTrial',CountedTrial);
% double check if anything goes wrong
Result2=GetParam(me,'Result');
% check if all non-processed trial is now processed
if sum(Result2((Result(1:CountedTrial)==0))==0)
    disp('not all non-processed trial is processed');
    SetParam(me,'Result',Result);
    % check if all processed trial is processed the same way
elseif sum((Result2((Result(1:CountedTrial)~=0))==Result((Result(1:CountedTrial)~=0)))~=1)
    disp('not all processed trial is processed the same way');
    SetParam(me,'Result',Result);
end