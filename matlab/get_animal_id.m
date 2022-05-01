function [a_id] = get_animal_id(animal_ID)
    split_parts = split(animal_ID, '-');
    if length(split_parts) == 3
        first = join(split_parts(1:2, :), '-');
        second_parts = split(split_parts{3}, '_');
        first{2} = second_parts{2};
        a_id = join(first, '_');
        a_id = a_id{1};
    else
        a_id = animal_ID;
    end
end