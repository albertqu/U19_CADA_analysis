root = 'D:\U19\datacamp\Summer_Coding_FP_Analysis\';
save_out = 'D:\U19\ProbSwitch_FP_data\';
save_out2 = '/media/data/U19/ProbSwitch_FP_data_test/';
matfile = [root 'probswitch_modeling_output.mat'];
iter_results = load(matfile);
[modeling_map, animals] = get_modeling_vars();
for i=1:length(modeling_map)
    mcode = modeling_map{i}{1};
    ages = modeling_map{i}{2};
    animal = animals{i};
    for j=1:length(mcode)
        ss = mcode(j);
        aa = ages(j);
        iden = [char(animal) '_' char(aa)];
        session_folder = find_folder_or_create(save_out, iden, 'XH');
        %session_folder2 = find_folder_or_create(save_out2, iden, 'XH');
        disp([ss; aa; session_folder]);
        h5fname = fullfile(session_folder, [iden '_modeling.hdf5']);
        if isfile(h5fname)
            disp([h5fname ' already exists, overwrite']);
            delete(h5fname);
        end
        fns = fieldnames(iter_results);
        for imdl=1:numel(fns)
            mdl = fns{imdl};
            %allfields = fieldnames(iter_results.(mdl).latents);
            %sz = size(iter_results.(mdl).latents.(allfields(1)));
            all_fields = fieldnames(iter_results.(mdl){i}(ss));
            for ilt=1:numel(all_fields)
                lt = all_fields{ilt};
                pname = ['/' char(mdl) '/' char(lt)];
                sz = size(iter_results.(mdl){i}(ss).(lt));
                h5create(h5fname, pname, sz);
                %h5write(h5fname, pname, iter_results.(mdl).latents.(allfields(1)));
                h5write(h5fname, pname, iter_results.(mdl){i}(ss).(lt));
            end
        end
    end
end
    

function out = generate_modeling_all()
    % all data
    root = 'D:\U19\datacamp\Summer_Coding_FP_Analysis\';
    save_out = 'D:\U19\ProbSwitch_FP_data\';
    save_out2 = '/media/data/U19/ProbSwitch_FP_data_test/';
    matfile = [root 'probswitch_modeling_output.mat'];
    iter_results = load(matfile);
    [modeling_map, animals] = get_modeling_vars();
    for i=1:length(modeling_map)
        animal = animals{i};
        fns = fieldnames(iter_results);
        nsessions = iter_results.(fns{1}){i};
        for j=1:nsessions
            iden = [char(animal) '_' char(string(j))];
            session_folder = find_folder_or_create(save_out, iden, '');
            disp(session_folder);
            h5fname = fullfile(session_folder, [iden '_modeling.hdf5']);
            fns = fieldnames(iter_results);
            for imdl=1:numel(fns)
                mdl = fns{imdl};
                %allfields = fieldnames(iter_results.(mdl).latents);
                %sz = size(iter_results.(mdl).latents.(allfields(1)));
                all_fields = fieldnames(iter_results.(mdl){i}(j));
                for ilt=1:numel(all_fields)
                    lt = all_fields{ilt};
                    pname = ['/' char(mdl) '/' char(lt)];
                    sz = size(iter_results.(mdl){i}(j).(lt));
                    h5create(h5fname, pname, sz);
                    %h5write(h5fname, pname, iter_results.(mdl).latents.(allfields(1)));
                    h5write(h5fname, pname, iter_results.(mdl){i}(j).(lt));
                end
            end
        end
    end
end

function out = generate_modeling_FP()
    root = 'D:\U19\datacamp\Summer_Coding_FP_Analysis\';
    save_out = 'D:\U19\ProbSwitch_FP_data\';
    save_out2 = '/media/data/U19/ProbSwitch_FP_data_test/';
    matfile = [root 'probswitch_modeling_output.mat'];
    iter_results = load(matfile);
    [modeling_map, animals] = get_modeling_vars();
    for i=1:length(modeling_map)
        mcode = modeling_map{i}{1};
        ages = modeling_map{i}{2};
        animal = animals{i};
        for j=1:length(mcode)
            ss = mcode(j);
            aa = ages(j);
            iden = [char(animal) '_' char(aa)];
            session_folder = find_folder_or_create(save_out, iden, 'XH');
            %session_folder2 = find_folder_or_create(save_out2, iden, 'XH');
            disp([ss; aa; session_folder]);
            h5fname = fullfile(session_folder, [iden '_modeling.hdf5']);
            fns = fieldnames(iter_results);
            for imdl=1:numel(fns)
                mdl = fns{imdl};
                %allfields = fieldnames(iter_results.(mdl).latents);
                %sz = size(iter_results.(mdl).latents.(allfields(1)));
                all_fields = fieldnames(iter_results.(mdl){i}(ss));
                for ilt=1:numel(all_fields)
                    lt = all_fields{ilt};
                    pname = ['/' char(mdl) '/' char(lt)];
                    sz = size(iter_results.(mdl){i}(ss).(lt));
                    h5create(h5fname, pname, sz);
                    %h5write(h5fname, pname, iter_results.(mdl).latents.(allfields(1)));
                    h5write(h5fname, pname, iter_results.(mdl){i}(ss).(lt));
                end
            end
        end
    end
end

function [out, animals] = get_modeling_vars()
    % Takes in animal and session, returning the modeling variables
%   i.e. the first day of FP is the 11th session in modeling code, 4th FP day is 16th, etc.
    BSD002_modeling_code = [11,12,13,16,21,24,27,32,33,37,42]; 
    BSD002_ages = {'p151_session1','p151_session2','p153','p156','p232','p235','p238','p243','p244','p248','p252'};
    BSD003_modeling_code = [3,4,7,10,13,17,24,28,32,35,42];
    BSD003_ages = {'p147_FP_LH','p147_FP_RH','p221','p224','p227','p231','p238','p242','p246','p249','p256'};
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
    animals = {'A2A-15B-B_RT','A2A-16B-1_RT','A2A-16B-1_TT','D1-27H_LT',...
               'A2A-19B_LT','A2A-19B_RT','A2A-19B_RV','D1-28B_LT'};
    out = maps;
end

function out = record_chris_sam_sessions()
    [maps, animals] = get_modeling_vars();
    jj=1;
    tosave={};
    for i=1:length(animals)
        ith_anm_indices = maps{i}{1};
        ith_anm_ages = maps{i}{2};
        for j=1:length(ith_anm_indices)
            row = {animals{i} ith_anm_indices(j) ith_anm_ages(j)};
            tosave(jj, :) = row;
            jj = jj + 1;
        end
    end
    tb = cell2table(tosave, 'VariableNames', {'animal', 'FP_number', 'session'});
    writetable(tb,'D:\U19\ProbSwitch_FP_data\samchris_modeling_sessions.csv');
end

function out = find_folder_or_create(root, key, attr)
    files = listdir(root);
    out = [];
    for f=files
        if contains(f, key)
            out = fullfile(root, f);
            if ~isfolder(out)
                out = [];
                break;
            end
        end
    end
    if isempty(out)
        out = fullfile(root, [key '_' attr]);
        mkdir(out)
    end
    out = char(out);
end

function contents = listdir(root)
    %helper function to return a list of directory contents (cell array),
    %without ./../.DS_Store
    Ds = dir(root);
    contents = {Ds.name};
    contents = contents(~ismember(contents, {'.', '..', '.DS_Store'}));
end