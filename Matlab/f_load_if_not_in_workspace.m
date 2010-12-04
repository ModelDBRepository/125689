function loaded_var = load_if_not_in_workspace(variable_name, path_and_filename_to_load)
% Will see if the file is in the workspace, if so 
% will NOT load the variable and will call the variable directly from the workspace
%
% path and filename HAS TO BE A STRING!

%[pathstr, filename, ext, versn] = fileparts('path_and_filename');

% Goes to workspace and see if variable is there, returns the variable name
file_in_workspace_flag = evalin('base',['who(''' variable_name ''')']);


% If the above flag is empty will have to load the file
if isempty(file_in_workspace_flag)                   
    fprintf('Loading: %s... ', path_and_filename_to_load);
    loaded_var = load(path_and_filename_to_load);
    fprintf('Done!\n');
else                                        % Means that var exists in the workspace
    loaded_var = evalin('base', variable_name);  % Gets var from the workspace
    fprintf(['NOTE: Using var ''' variable_name ''' directly from the workspace!\nNOT loading new file!\n\n']); beep;
end

end % End main function
