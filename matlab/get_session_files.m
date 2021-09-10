function [filenames] = get_session_files(folder, animal, session, ftypes, fs_opt)
    if ~iscell(ftypes)
        ftypes = {ftypes};
    end
    nf = length(ftypes);
    filenames = cell(nf, 1);
    for k=1:nf
        filenames{k} = '';
    end
    if strcmp(fs_opt, 'root')
        files = listdir(folder);
        prefix = folder;
    elseif strcmp(fs_opt, 'animal')
        files = listdir(fullfile(folder, animal));
        prefix = fullfile(folder, animal);
    elseif strcmp(fs_opt, 'session')
        files = listdir(fullfile(folder, animal, session));
        prefix = fullfile(folder, animal, session);
    else
        disp('wrong option')
        return
    end
    tokenNames = regexp(session, '(?<day>p\d+)(?<S>_session\d+|_?)', 'names');
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
                    && contains(f, keyword) && ~contains(f, 'analyzed');
            else
                criterion = contains(f, animal) && contains(f, tokenNames.day) && contains(f, tokenNames.S)...
                    && contains(f, keyword) && ~contains(f, 'analyzed');
            end
                
            if criterion
                filenames{k} = string(fullfile(prefix, f));
                found = found + 1;
            end
        end
    end
    if length(filenames) == 1
        filenames = filenames{1};
    end
end

function contents = listdir(root)
    %helper function to return a list of directory contents (cell array),
    %without ./../.DS_Store
    Ds = dir(root);
    contents = {Ds.name};
    contents = contents(~ismember(contents, {'.', '..', '.DS_Store'}));
end