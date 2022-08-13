root = 'D:\U19\test';
Ds = dir(root);
contents = {Ds.name};
contents = contents(~ismember(contents, {'.', '..', '.DS_Store'}));
for f=contents
    f=char(f);
    if contains(f, 'BSD019')
        f_splits = split(f(1:end-4), '_');
        p_age = str2num(replace(f_splits{end}, 'p', ''));
        new_age = p_age+2;
        new_fname = replace(f, f_splits{end}, sprintf('p%.2d', new_age));
        movefile(fullfile(root, f), fullfile(root, new_fname));
    end
end
