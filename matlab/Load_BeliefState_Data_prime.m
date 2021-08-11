function [Animal_Behavior_Data,celltype,FP_Data,data_notes,hemispheres,filenames,AnimalList, sessions, modeling_var,modeling_note,GLM]=Load_BeliefState_Data_prime(root)

include_model = 0; %decide whether or not to include modeling data
include_GLM = 0; %decide whether or not to include GLM modeled signal

%select_animals = [2,3,4,5]; %include integers that correspond to the AnimalList names of the animals you want to analyze

%AnimalList = {'BSD001','BSD002','BSD003','BSD004','BSD005', 'BSD006','BSD007','BSD008','BSD009'};
%AnimalNames = {'A2A-15B-B_LT','A2A-15B-B_RT','A2A-16B-1_RT','A2A-16B-1_TT','D1-27H_LT',...
%               'A2A-19B_LT','A2A-19B_RT','A2A-19B_RV','D1-28B_LT'};
%celltype = {'D2-MSN', 'D2-MSN','D2-MSN','D2-MSN','D1-MSN','D2-MSN','D2-MSN','D2-MSN','D1-MSN'};

AnimalList = {'BSD002','BSD003','BSD004','BSD005',...
              'BSD006','BSD007','BSD008','BSD009'};
AnimalNames = {'A2A-15B-B_RT','A2A-16B-1_RT','A2A-16B-1_TT','D1-27H_LT',...
               'A2A-19B_LT','A2A-19B_RT','A2A-19B_RV','D1-28B_LT'};

celltype = {'D2-MSN','D2-MSN','D2-MSN','D1-MSN','D2-MSN','D2-MSN','D2-MSN','D1-MSN'};

% FP information
data_notes = {{'top_top'},... 
              {'top_top','top_top','top_top','top_top','bottom_bottom',...
               'top_top','top_top','top_top','top_top','top_top',...
               'top_top'},...
              {'top_top','bottom_top','top_top','top_top','top_top',...
               'top_top','top_top','top_top','top_top','top_top'},...
              {'top_top','top_top','bottom_top','top_top','top_top',...
               'top_top','top_top','top_top','top_top','top_top',...
               'top_top','top_top'},...
              {'bottom_top'},...
              {'top_top','top_top','top_top'},...
              {'bottom_top','top_top','top_top'},...
              {'bottom_top','top_top','top_top','top_top'}};      

hemispheres = {{'NAc','DMS'},{'DMS','DMS'},{'DMS','DMS'},{'DMS','NAc'},...
               {'NAc','NAc'},{'NAc','NAc'},{'NAc','NAc'},{'NAc','NAc'}};
           
% sessions = {{},...
%             {},...
%             {},...
%             {},...
%             {},...
%             {},...
%             {},...
%             {'p135_FP_LH', 'p135_session2_FP_LH'}};
sessions = get_all_sessions(root, AnimalNames); 

% % properly handle filenames, maybe delete later
% filenames = {{'A2A-15B-B_RT_20200314_ProbSwitch_FP_LH_p153'},...
%              {'A2A-16B-1_RT_20200317_ChR2_FP_LH_p147','A2A-16B-1_RT_20200317_ChR2_FP_RH_p147','A2A-16B-1_RT_20200530_ProbSwitch_FP_RH_p221',...
%              'A2A-16B-1_RT_20200602_ProbSwitch_FP_LH_p224','A2A-16B-1_RT_20200605_ProbSwitch_FP_RH_p227','A2A-16B-1_RT_20200609_ProbSwitch_FP_LH_p231',...
%              'A2A-16B_RT_20200616_ProbSwitch_p238_FP_RH','A2A-16B_RT_20200620_ProbSwitch_p242_FP_LH','A2A-16B_RT_20200624_ProbSwitch_p246_FP_RH_lots_miss_after_trial1100',...
%              'A2A-16B_RT_20200627_ProbSwitch_p249_FP_LH','A2A-16B_RT_20200704_ProbSwitch_p256_FP_RH'},...
%              {'A2A-16B-1_TT_20200315_ChR2-Switch-no-cue_FP_RH_p145','A2A-16B-1_TT_20200316_ChR2-Switch-no-cue_FP_LH_p146','A2A-16B-1_TT_20200529_ProbSwitch_FP_RH_p220',...
%              'A2A-16B-1_TT_20200531_ProbSwitch_FP_LH_p222','A2A-16B_TT_20200611_ProbSwitch_p233_FP_LH','A2A-16B_TT_20200615_ProbSwitch_p237_FP_RH',...
%              'A2A-16B_TT_20200619_ProbSwitch_p241_FP_LH','A2A-16B_TT_20200623_ProbSwitch_p245_FP_RH','A2A-16B_TT_20200626_ProbSwitch_p248_FP_LH',...
%              'A2A-16B_TT_20200703_ProbSwitch_p255_FP_LH'},...
%              {'D1-27H_LT_20200314_ProbSwitch_FP_RH_p103'},...
%              {'A2A-19B_LT_20200709_ProbSwitch_p140_FP_LH'},...
%              {'A2A-19B_RT_20200708_ProbSwitch_p139_FP_LH','A2A-19B_RT_20200712_ProbSwitch_p143_FP_RH_no_signal','A2A-19B_RT_20200717_ProbSwitch_p148_FP_LH'},...
%              {'A2A-19B_RV_20200707_ProbSwitch_p138_FP_LH_no470','A2A-19B_RV_20200711_ProbSwitch_p142_FP_RH','A2A-19B_RV_20200716_ProbSwitch_p147_FP_LH'},...
%              {'D1-28B_LT_20200706_ProbSwitch_p135_FP_LH','D1-28B_LT_20200706_ProbSwitch_p135_FP_LH_session2','D1-28B_LT_20200710_ProbSwitch_p140_560nm_only_470forSync_FP_RH',...
%               'D1-28B_LT_20200715_ProbSwitch_p144_FP_LH'}}; %'D1-28B_LT_20200706_ProbSwitch_p135_FP_LH_session2' valve problem


filenames = cell(length(AnimalList), 1);
Animal_Behavior_Data = cell(length(AnimalList), 1);
FP_Data = cell(length(AnimalList), 1);
% filenames = {};
% Animal_Behavior_Data = {};
% FP_Data = {};
for i = 1:length(AnimalList)
    Animal_Behavior_Data{i} = cell(length(sessions{i}), 1);
    FP_Data{i} = cell(length(sessions{i}), 1);
%     Animal_Behavior_Data{i} = {};
%     FP_Data{i} = {};
    animal = AnimalNames{i};
    for j = 1:length(sessions{i})
        session = sessions{i}{j};
        matfile = get_session_files(root, animal, session, 'exper');
        if ~isempty(matfile)
            [~, f, ~] = fileparts(string(matfile));
            Animal_Behavior_Data{i}{j} = matfile;
            filenames{i}{j} = f;
            FP_Data{i}{j} = get_session_files(root, animal, session, {'green', 'red', 'Binary_Matrix', 'timestamp', 'MetaData'});
        end
    end
end
        

% 1=green, 2=red, 3=NIDAQ_Ai0_Binary_Matrix, 4=NIDAQ_Ai0_timestamp

modeling_var={};
modeling_note={};
%modeling_var is a triply indexed cell. 1st = model variables; 2nd =
%animal; 3rd = session
if include_model
    modeling_note={{'4_Trial_Back_Regress'},{'Qs','Ps'}};
    load('cohort_modeling_analysis.mat','models');
    FP_sessions = {{11,12,13,16,21,24,27,32,33,37,42},...%[11,12,13,16,21,24,27,32,33,37,42]
                   {3,4,7,10,13,17,24,28,32,35,42},...
                   {7,8,15,25,26,30,34,38,41,48},...
                   {9,10,11,12,23,29,33,37,40,43,47,50},...[9,10,11,12,23,29,33,37,40,43,47,50]
                   {26},...
                   {25,29,34},...
                   {24,28,33},...
                   {24,25,29,34}}; %[24,25,29,34]
    for p = 1:length(FP_Data)
        for g = 1:length(FP_sessions{p})
            modeling_var{p}(g) = models.regression_4_trials{p}(FP_sessions{p}{g});
        end
    end    
end

if include_GLM
    load('glm_test_xval_full')
    GLM{1}{1} = glm_test.BRL.FP_time; % {1} = green; {1}{1} = green time;
    GLM{1}{2}{1} = glm_test.BRL.FP_signal; % {1}{1-3} = green signals
    GLM{1}{2}{2} = glm_test.BRL.vb.signal_fit;
    GLM{1}{2}{3} = glm_test.SRL.vb.signal_fit;
else
    GLM = {};
end
end

function [filenames] = get_session_files(folder, animal, session, ftypes)
    if ~iscell(ftypes)
        ftypes = {ftypes};
    end
    nf = length(ftypes);
    filenames = cell(nf, 1);
    for k=1:nf
        filenames{k} = '';
    end
    files = listdir(fullfile(folder, animal));
    tokenNames = regexp(session, '(?<day>p\d+)(?<S>_session\d+|_?)_FP_(?<H>(L|R)H)', 'names');
    found = 0;
    for f=files
        if found == nf
            break
        end
        for k=1:nf
            keyword = ftypes{k};
            if strcmp(keyword, 'exper')
                keyword = '.mat';
            end
            if strcmp(tokenNames.S, '')
                criterion = contains(f, animal) && contains(f, tokenNames.day) && (~contains(f, 'session'))...
                    && contains(f, tokenNames.H) && contains(f, keyword) && ~contains(f, 'analyzed');
            else
                criterion = contains(f, animal) && contains(f, tokenNames.day) && contains(f, tokenNames.S)...
                    && contains(f, tokenNames.H) && contains(f, keyword) && ~contains(f, 'analyzed');
            end
                
            if criterion
                filenames{k} = string(fullfile(animal, f));
                found = found + 1;
            end
        end
    end
    if length(filenames) == 1
        filenames = filenames{1};
    end
end
    
function sessions = get_all_sessions(folder, animals)
    sessions = cell(length(animals), 1);
    for i=1:length(animals)
        sessions{i} = {};
        animal = animals{i};
        animal_folder = fullfile(folder, animal);
        if ~exist(animal_folder,'dir')
            continue
        end
        files = listdir(animal_folder);
        asession = files(contains(files, animal) & contains(files, 'timestamp'));
        regstr = '.*_(?<H>[LR]H)_(?<A>p\d+)[-&\w]*_timestamp(?<S>_session\d+_|_?)[-\w]*.csv';
        for s = asession
            tokens = regexp(string(s), regstr, 'names');
            if isempty(tokens)
                continue;
            else
                if contains(tokens.S, 'session')
                    session = strcat(tokens.A, tokens.S, 'FP_', tokens.H);
                else
                    session = strcat(tokens.A, '_FP_', tokens.H);
                end
                sessions{i}{end+1} = string(session);
            end
        end
    end
end
        
    
function contents = listdir(root)
    %helper function to return a list of directory contents (cell array),
    %without ./../.DS_Store
    Ds = dir(root);
    contents = {Ds.name};
    contents = contents(~ismember(contents, {'.', '..', '.DS_Store'}));
end

        