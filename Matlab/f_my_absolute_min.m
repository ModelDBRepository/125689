function temp_min = my_absolute_min(input_matrix)

temp_min = [0 0];

while find(size(temp_min) > 1)

    temp_min = min(input_matrix);
    input_matrix = temp_min;
end

return;

end