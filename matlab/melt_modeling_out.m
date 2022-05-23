function [out] = melt_modeling_out(outpath, results, data, catalog, modeling_id)
    %% split results and organize by animal session, and 
    % append modeling evaluation results to table <category>
    % outpath: path to save modeling data
    % for each animal session, create hfile, create 1 entry for each model
    % for each model, result(imdl).latents(a_i).(lt)(data(a_i).s == s_j)
    n_animal = length(unique(catalog.a_i));
    alevel_fields = {'K', 'x', 'logpost', 'loglik', 'bic', 'aic'}; %x diff for each model
    models = {'HMM', 'SRL', 'BRL'};
    modelx = {{'invtemp', 'tr', 'st'}, {'invtemp', 'lr_pos', 'lr_neg', 'st', 'q0'}, ...
        {'invtemp', 'tr', 'lr', 'st', 'd'}};
    
    for a_i = 1:n_animal
        % save animal level modeling summary to h5file
        animal = catalog((catalog.a_i == a_i), :).animal{1};
        ai_folder = fullfile(outpath, animal);
        if ~isfolder(ai_folder)
            mkdir(ai_folder);
        end
        ai_fname = fullfile(ai_folder, [animal '_meval_' modeling_id '.hdf5']);
        for ialf=length(alevel_fields)
            for imdl=1:length(models)
                alf = alevel_fields{ialf};
                pname = ['/' models{imdl} '/' alf];
                if strcmp(alf, 'x')
                    xfs = modelx{imdl};
                    for ixfs=1:length(xfs)
                        pname_n = [pname '/' xfs{ixfs}];
                        h5create(ai_fname, pname_n, 1);
                        h5write(ai_fname, pname_n, results(imdl).(alf)(1, ixfs));
                    end
                else
                    h5create(ai_fname, pname, 1);
                    h5write(ai_fname, pname, results(imdl).(alf)(1));
                end
            end
        end
        for s_j=1:length(unique(data(a_i).s))
            session = catalog((catalog.a_i == a_i) & (catalog.s_j == s_j), :).session{1};
            ij_folder = fullfile(outpath, animal, session);
            if ~isfolder(ij_folder)
                mkdir(ij_folder);
            end
            ij_fname = fullfile(ij_folder, [animal '_' session '_modeling_' modeling_id '.hdf5']);
            for imdl=1:length(models)
                latents = fieldnames(results(imdl).latents(a_i));
                for ilt=1:length(latents)
                    lt = latents{ilt};
                    d2save = results(imdl).latents(a_i).(lt)(data(a_i).s == s_j);
                    pname = ['/' models{imdl} '/' lt];
                    h5create(ij_fname, pname, size(d2save));
                    h5write(ij_fname, pname, d2save)
                end
            end
            % save raw data
        end
    end
    out=[];
end