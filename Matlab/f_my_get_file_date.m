function file_date = my_get_file_date(filename)

% check if file exists
file_struct = dir(filename);
file_date = file_struct.date;

end