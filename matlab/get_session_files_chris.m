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