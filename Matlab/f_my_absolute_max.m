function temp_max = my_absolute_max(input_matrix)

temp_max = [0 0];

while find(size(temp_max) > 1)

    temp_max = max(input_matrix);
    input_matrix = temp_max;
end

return;

end