function my_print_txted_parameters(txted_parameters, subplot_row, subplot_col, number_last_rows, columns_to_use)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Writes into the current figure the texted parameters
%
%   - subplot_row and subplot_col are the values for the general subplot matrix
%   - They will always be printed in the LAST ROW of the subplot matrix  06/21/06 IF number_last_rows IS NOT PROVIDED
%   - 06/21/06 Added extra parameter: number_last_rows, which says into how many LAST rows to print the text
%   - 07/17/06 Re-sets gca, to make better use of page width
%   - 12/04/06 Added extra parameter: in which columns to print the parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (nargin < 3)
    error('### Please provide AT LEAST 3 arguments to this function');
% If number_last_rows is NOT provided will print in the LAST row
elseif (nargin < 4)
    number_last_rows = 1;
    columns_to_use = 1:subplot_col;
elseif (nargin < 5)
    columns_to_use = 1:subplot_col;
end

% Rounds the size of the text array to the subplot and number of lines that will be printed per column
for i = length(txted_parameters)+1:length(columns_to_use)*ceil(length(txted_parameters)/length(columns_to_use))
    txted_parameters{i} = '';
end

for i = 1:length(columns_to_use)
    
    column = columns_to_use(i);
    
    subplot(subplot_row,subplot_col, fliplr((subplot_row-1)*subplot_col+1 + (column-1) - ((0:number_last_rows-1)*subplot_col)));
    pos = get(gca, 'Position');
    %set(gca ,'Position', [0.05+(1/subplot_col)*(i-1) pos(2) (1/subplot_col)*0.95 pos(4)]);
    set(gca ,'Position', [0.05+(1/subplot_col)*(column-1) pos(2) (1/subplot_col)*0.5 pos(4)]);
    
    set(gca, 'Visible','off', 'color','none');
    text('Units','normalized', 'Position',[0 0.5], 'String',txted_parameters(1+(i-1)*ceil(end/length(columns_to_use)):ceil(end/length(columns_to_use))*i), 'FontSize',8, 'Interpreter','none');

end

end % End main function




% 
% if (nargin < 3)
%     error('### Please provide AT LEAST 3 arguments to this function');
% % If number_last_rows is NOT provided will print in the LAST row
% elseif (nargin < 4)
%     number_last_rows = 1;
% elseif (nargin < 5)
%     columns_to_use = 1:subplot_col;
% end
% 
% % Rounds the size of the text array to the subplot and number of lines that will be printed per column
% for i = length(txted_parameters)+1:subplot_col*ceil(length(txted_parameters)/subplot_col)
%     txted_parameters{i} = '';
% end
% 
% for i = 1:subplot_col
%     
%     subplot(subplot_row,subplot_col, fliplr((subplot_row-1)*subplot_col+1 + (i-1) - ((0:number_last_rows-1)*subplot_col)));
%     pos = get(gca, 'Position');
%     %set(gca ,'Position', [0.05+(1/subplot_col)*(i-1) pos(2) (1/subplot_col)*0.95 pos(4)]);
%     set(gca ,'Position', [0.05+(1/subplot_col)*(i-1) pos(2) (1/subplot_col)*0.5 pos(4)]);
%     
%     set(gca, 'Visible','off', 'color','none');
%     text('Units','normalized', 'Position',[0 0.5], 'String',txted_parameters(1+(i-1)*ceil(end/subplot_col):ceil(end/subplot_col)*i), 'FontSize',8, 'Interpreter','none');
% 
% end