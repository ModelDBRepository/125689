function p_txt = my_format_parameters_to_txt(p, char_lim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Takes as input the parameters 'p' STRUCTURE and returns
% a nicely formated 'p_txt' CELL ARRAY
%
% 16 Jan 2008 - If 'exclude_for_print' is field will remove it
%
% 06 Out 2008 - If one of the fields is a matrix or column will just print first row
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if isfield(p,'exclude_for_print'); p = rmfield(p, 'exclude_for_print'); end

    % Sets default length for char_lim
    if nargin < 2; char_lim = 50; end;

    fields = fieldnames(p);
    
    for i = 1:length(fields)
        % If the field is a CELL array, will just use the FIRST cell 
        if iscell(p.(fields{i}))
            p.(fields{i}) = p.(fields{i}){1};
		% 06 Out 2008 - If one of the fields is a matrix or column will just print first row
		elseif isnumeric( p.(fields{i}) )
			if size(p.(fields{i}),1) > 1
				p.(fields{i}) = p.(fields{i})(1,:);
			end
			
        end
        
       p_txt{i} = [inputname(1) '.' fields{i} ' = ' num2str(p.(fields{i}))];    % Input name is the variable names, in this case just the first variable
       if length(p_txt{i}) > char_lim;
           p_txt{i} = p_txt{i}(1:char_lim-1);
           %fprintf('%s\n', p_txt{i});
       end
    end

end % End Main function