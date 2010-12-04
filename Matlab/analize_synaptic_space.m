%% CARVALHO & BUONOMANO 
%% NEURON 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function analize_synaptic_space_multiple_trials_and_inputs
% For Gulbenkian PPT

% SELECT THIS IF WANT TO PRINT INPUTS or EPSPs
p.print_EPSPs = 1;



run_label = ''; % This has been fixed for the Units of Inh->Ex

p.s1 = [mfilename 'v12.05'];


NEURON_output_data_filename = 'SynSpace_Out';
NEURON_output_parameters_filename = 'SynSpace_Param';
IW_var_handle = ['IW' run_label]; % Main name of the voltage file

%tstop = 40;
process_IW = 0;

if (~strcmp(run_label, ''))
    NEURON_output_data_filename = [NEURON_output_data_filename '_'];
    NEURON_output_parameters_filename = [NEURON_output_parameters_filename '_'];
end

data_filename = [NEURON_output_data_filename run_label '.dat'];
parameters_filename = [NEURON_output_parameters_filename run_label '.dat'];

% Format of the synaptic space output file
if (0)
Ex_i_col = 1;
Ex_W_col = 2;
Ex_Ca_col = 3;
Ex_max_col = 4;
Inh_i_col = 5;
Inh_W_col = 6;
Inh_Ca_col = 7;
Inh_max_col = 8;
total_inh_fired = 9;
nr_inputs = 10;
EPSP_slope = 11;
else
Ex_i_col = 1;
Ex_Ca_col = 2;
Inh_i_col = 3;
total_inh_fired = 4;
nr_inputs = 5;
EPSP_slope = 6;
end

p.colormap_total_colors = 250;
p.minY_thr_sigmoid = 0.1;
p.maxY_thr_sigmoid = 0.8;
p.title = data_filename;
p.linear_bound = [0.25 0.75]; 


Neuron_Output_Matrix = f_load_if_not_in_workspace('Neuron_Output_Matrix', data_filename);
assignin('base', 'Neuron_Output_Matrix',Neuron_Output_Matrix);


%%% Reads parameters from Parameter Scan %%%
param_fid = fopen(parameters_filename,'r');
param_txt = {}; % Will store txt with parameter values, that will be printed to figure
while (1)
    param_line = fgetl(param_fid);
    if param_line == -1 | strcmp(param_line,'STOP'); fclose(param_fid); break; end;
    %fprintf('%s\n',param_line);
    A = find(param_line == '=');
    eval([param_line(1:A-1) '= str2num(param_line(A+1:length(param_line)));']);
    param_txt{end+1} = sprintf('%s',param_line); % Store parameter txt line to be printed into figure
end


if rem(ExINPUT_end, ExINPUT_block_size) ~= 0; error('ExINPUT_end should be multiple of ExINPUT_block_size! Code halted'); end;
inpts_idx_array = ExINPUT_block_size:ExINPUT_block_size:ExINPUT_end;


tic

% Builds 3D matrix to represent the probability of firing with varying Inp->Ex and Inh->Ex and varying input strength
fprintf('If Ex fires more than once, I''ll make it as if it fired only once!\n');
Neuron_Output_Matrix(Neuron_Output_Matrix(:, Ex_Ca_col) > 1, Ex_Ca_col) = 1; % If Ex cell fired more than once make it as if was only once!
fprintf('Neuron output file has to be:\nP1\n\tP2\n\t\tInputs\n\t\t\trepetitions\n');

% fprintf('Problem in assignment matrix here if P1_steps is 1 or P2 and P1_steps are 1!!'); beep;
Ex_w_inputs_space_mat = zeros(nr_repetitions, ExINPUT/ExINPUT_block_size, P2_steps, P1_steps); % This HAS TO MATCH the order in which Neuron file is written: repetitions->inputs->Inh->Ex
Ex_w_inputs_space_mat(:) = Neuron_Output_Matrix(:,Ex_Ca_col); 
Ex_w_inputs_space_mat = squeeze(sum(Ex_w_inputs_space_mat, 1)); % Sums Ex firing for all repetitions and reduces dimension
Ex_w_inputs_space_mat = shiftdim(Ex_w_inputs_space_mat, 1); % Just to put in the old format (Inh, Ex, Inputs)
Ex_w_inputs_space_mat = Ex_w_inputs_space_mat/(nr_repetitions); if find(Ex_w_inputs_space_mat > 1); error('How come probabilities acn be bigger than 1??'); end;
assignin('base','Ex_w_inputs_space_mat',Ex_w_inputs_space_mat); % (Inh i, Ex i, ipt_idx) = probability of firing for these conditions

% Builds Avg Inh firing matrix, to represent the probability of firing with varying Inp->Ex and Inh->Ex and varying input strength
if (~isempty(find(Neuron_Output_Matrix(:, total_inh_fired) > nInh))); error('How come is detecting firing of Inh > nInh?!!!\nThis should have been fixed in NEURON!!\n'); end;
Avg_Inh_firing_space_mat = zeros(nr_repetitions, ExINPUT/ExINPUT_block_size, P2_steps, P1_steps);
Avg_Inh_firing_space_mat(:) = Neuron_Output_Matrix(:,total_inh_fired); 
Avg_Inh_firing_space_mat = squeeze(sum(Avg_Inh_firing_space_mat, 1)); % Sums Ex firing for all repetitions and reduces dimension
Avg_Inh_firing_space_mat = shiftdim(Avg_Inh_firing_space_mat, 1); % Just to put in the old format (Inh, Ex, Inputs)
Avg_Inh_firing_space_mat = Avg_Inh_firing_space_mat/(nr_repetitions*nInh); if find(Avg_Inh_firing_space_mat > nInh); error('How come percentage Inh firing be bigger than 1??'); end;
assignin('base','Avg_Inh_firing_space_mat',Avg_Inh_firing_space_mat);


% IO slopes is a cell array, for each {P1, P2} has a matrix with 2 columns, 1st column is EPSP slope, 2nd column is whether there was a spike or not
IO_EPSP_slopes = zeros(nr_repetitions*ExINPUT/ExINPUT_block_size, P2_steps, P1_steps);
IO_EPSP_slopes(:) = Neuron_Output_Matrix(:,EPSP_slope);
IO_fired = zeros(nr_repetitions*ExINPUT/ExINPUT_block_size, P2_steps, P1_steps);
IO_fired(:) = Neuron_Output_Matrix(:,Ex_Ca_col);
for i = 1:P2_steps
    for ii = 1:P1_steps
        IO_cell_matrix{i, ii}(:,:) = [squeeze(IO_EPSP_slopes(:,i,ii)) squeeze(IO_fired(:,i,ii))];
    end
end
fprintf('Time parsing data: %.2f sec\n', toc);
assignin('base', 'IO_cell_matrix', IO_cell_matrix);

tic
    

%% Uses Inputs instead of EPSP slopes as X axis to compute sigmoids
if ~(p.print_EPSPs)
%sig_coef_mat_INPUTS = compute_sigmoids_with_inputs_in_X(Ex_w_inputs_space_mat, inpts_idx_array);
sig_coef_mat= compute_sigmoids_with_inputs_in_X(Ex_w_inputs_space_mat, inpts_idx_array);
end


%% Uses EPSP slope instead of inputs
if (p.print_EPSPs)
    %IO_cell_matrix;

    p.nr_bins = ExINPUT;
    
    
    % Makes min number of points in bins 10% of repetitions, but no less than 5!
    if ceil(nr_repetitions*0.1) > 5; p.min_bin_count = ceil(nr_repetitions*0.1);
    else p.min_bin_count = 5; end

     
    t_mat = cell2mat(IO_cell_matrix);
    all_EPSPs = t_mat(:,1:2:size(t_mat,2));
      
    %EPSP_range = max(current_IO_mat(:,1)) - min(current_IO_mat(:,1));
    
    p.bin_start = f_my_absolute_min(all_EPSPs);
    p.bin_end = f_my_absolute_max(all_EPSPs);
    EPSP_range = p.bin_end - p.bin_start;
    p.bin_size = EPSP_range/p.nr_bins;
    p.hist_x_axis = p.bin_start:p.bin_size:p.bin_end; p.hist_x_axis(end) = p.hist_x_axis(end)*1.00001; % Multiplication is because of EDGES value, see help in histc

    
    clear('t_mat','all_EPSPs');


    warning off
    for Inh_i = 1:size(IO_cell_matrix,1)
        for Ex_i = 1:size(IO_cell_matrix,2)
            hist_prob_data = [];
            current_IO_mat = IO_cell_matrix{Inh_i,Ex_i};   % current_IO_mat holds 2 columns, first column is EPSP slope, 2nd column is firing or not
             
            [n_counts, bin_idxs] = histc(current_IO_mat(:,1), p.hist_x_axis); % Notice that n_counts will contain 1 more bin, that corresponds exactly to the value of last EDGE, but this was already accounted for above            
            for i = 1:p.nr_bins
                if any(bin_idxs==i) % Only if there are EPSPs in the current bin
                    hist_prob_data(i,1) = p.hist_x_axis(i) + p.bin_size/2;        % Histogram centers
                    hist_prob_data(i,2) = sum(current_IO_mat(bin_idxs==i,2));       % Counts the spikes in for this bin
                    hist_prob_data(i,3) = numel(current_IO_mat(bin_idxs==i,2));     % Counts the total number of events in this bin
                end
            end
            % Gets rid of bins with low count
            hist_prob_data(hist_prob_data(:,3) < p.min_bin_count,:) = [];
            
            % Makes the SIGMOID FITTINGS
            xx = hist_prob_data(:,1);
            yy = hist_prob_data(:,2)./hist_prob_data(:,3);

            % If never fired for this W1.W2
            if ~any(hist_prob_data(:,2))
                sig_coef_mat(Inh_i, Ex_i, 1) = 0;
                sig_coef_mat(Inh_i, Ex_i, 2) = 0;   % SLOPE
                sig_coef_mat(Inh_i, Ex_i, 3) = 0;   % sigmoid Y value at last point
                sig_coef_mat(Inh_i, Ex_i, 4) = 0; % sigmoid Y value at first point
                beta = [0 0];
            % If fired for this W1.W2 compute sigmoid
            else
                b0 = 0;
%% May 8th 2007 better estimates of parameters, fits linear bins ]0 1[ and then estimates sigmoidal k
                dummy_idxs = find(yy > 0 & yy < 1);
                if length(dummy_idxs) > 1   % If there are at least 2 bins between 0 and 1
                    dummy_xx = xx(dummy_idxs); dummy_yy = yy(dummy_idxs);
                    dummy = polyfit(dummy_xx,dummy_yy,1); % Will fit linearly the bins, and use that slope as estimate.
                    if (dummy(1) > 0)                   
                        b0 = [mean(xx) 0.236*dummy(1)^-1.0078]; % Convoluted relationship between the linear slope determined above and k
                    end
                end

                % This means b0 hasn't been set yet
                if b0 == 0;
                    dummy = polyfit(dummy_xx,dummy_yy,1); % Will fit linearly ALL the bins, and use that slope as estimate.
                    b0=[mean(xx) 0.236*dummy(1)^-1.0078];
                end   % Asymptote, E50 (x axis), slope

                [beta,r,J]=nlinfit(xx,yy,@f_SIGMOID_C,b0);    % SIGMOID_C - constrained, ONLY TAKES 2 PARAMETERS!
                %fprintf('Estimated k: %.2f    fit k: %.2f\n\n', b0(2), beta(2));

                sig_coef_mat(Inh_i, Ex_i, 1) = beta(1);
                sig_coef_mat(Inh_i, Ex_i, 2) = beta(2); % Sigmoid SLOPE
                sig_coef_mat(Inh_i, Ex_i, 2) = 0.1; % Sigmoid SLOPE
                
                
                
                sig_coef_mat(Inh_i, Ex_i, 3) = f_SIGMOID_C(beta, max(xx)); % sigmoid Y value at last point
                sig_coef_mat(Inh_i, Ex_i, 4) = f_SIGMOID_C(beta, min(xx)); % sigmoid Y value at first point
            
                 if any(sig_coef_mat(Inh_i, Ex_i, 4) <= p.linear_bound(1) && sig_coef_mat(Inh_i, Ex_i, 3) >= p.linear_bound(2))
                     xxx = [min(xx):(max(xx)-min(xx))/500:max(xx)];  % plots 500 points
                     coeffs = f_get_linear_coefs_from_sigmoid(xxx, f_SIGMOID_C(beta, xxx), p); 
                     sig_coef_mat(Inh_i, Ex_i, 2) = 1/coeffs.a; % Sigmoid SLOPE 
					 %sig_coef_mat(Inh_i, Ex_i, 2) = coeffs.dr;		% REAL Dynamic Range
                 end
             
				
            end
            
            plot_data{Inh_i,Ex_i,1} = xx;
            plot_data{Inh_i,Ex_i,2} = yy;
            plot_data{Inh_i,Ex_i,3} = beta;
        end
    end
    assignin('base','plot_data',plot_data);
end
fprintf('Time parsing sigmoids: %.2f sec\n', toc);
tic

% "Fixes" the coefficients
%%%%%%%%%%%%%%%
% Makes 2D only copies of slopes and maxY value of sigmoid
temp_mat_E50 = sig_coef_mat(:,:,1);
temp_mat_slope = sig_coef_mat(:,:,2);
temp_mat_maxY = sig_coef_mat(:,:,3);
temp_mat_minY = sig_coef_mat(:,:,4);

% Gets rid of negative values
temp_mat_slope(temp_mat_slope < 0) = -1;
%temp_mat_E50(temp_mat_slope < 0) = -1;

% Gets rid of HUGE slopes, either the ones at the bottom, low prob of firing, as well as the ones with high prob of firing
%temp_mat_E50((temp_mat_maxY < p.maxY_thr_sigmoid & temp_mat_slope > 0) | (temp_mat_minY > p.minY_thr_sigmoid & temp_mat_slope > 0)) = -1; %Labels HUGE SLOPES -1 is just to label
temp_mat_slope((temp_mat_maxY < p.maxY_thr_sigmoid & temp_mat_slope > 0) | (temp_mat_minY > p.minY_thr_sigmoid & temp_mat_slope > 0)) = -1; %Labels HUGE SLOPES -1 is just to label

% Computes my maximal slope (after getting rid of HUGE outliers)
max_sig_slope = max(max(temp_mat_slope));   % Computes the maximum slope (now without, those outliers)

% Gets rid of very FAST slopes (horizontal lines)
temp_mat_slope(temp_mat_slope < (max_sig_slope/p.colormap_total_colors)*2 & temp_mat_slope > 0) = -2; % max_sig_slope/(p.colormap_total_colors-4);
%temp_mat_E50(temp_mat_slope < (max_sig_slope/p.colormap_total_colors)*2 & temp_mat_slope > 0) = -2; % max_sig_slope/(p.colormap_total_colors-4);

% Makes gray and black in E50, using slope as template
temp_mat_E50(temp_mat_slope == -1) = -1;
temp_mat_E50(temp_mat_slope == -2) = -2;

% Gives appropriate values, huge slopes will be second in colorbar, quick slopes will be third in colorbar
temp_mat_slope(temp_mat_slope == -1) = max_sig_slope/(p.colormap_total_colors-2);   % Huje slopes will be before last color
temp_mat_slope(temp_mat_slope == -2) = (max_sig_slope/(p.colormap_total_colors-2))*2;   % quick slopes will be before last color


% Gives appropriate values to E50, so that it has appropriate colors
max_E50 = f_my_absolute_max(temp_mat_E50);   % Computes the maximum E50, in the cleaned up sigmoids
min_E50 = f_my_absolute_min(temp_mat_E50(temp_mat_E50 > 0));
delta_E50 = max_E50 - min_E50;
if (delta_E50)
temp_mat_E50(temp_mat_E50 == 0) = min_E50 - (delta_E50/(p.colormap_total_colors-2))*2;   % E50 w/ NO spikes will be last color
temp_mat_E50(temp_mat_E50 == -1) = min_E50 - (delta_E50/(p.colormap_total_colors-2))*1;  % Mot usefull E50 will be before last color
temp_mat_E50(temp_mat_E50 == -2) = min_E50 - (delta_E50/(p.colormap_total_colors-2))*1;  % Mot usefull E50 will be before last color
end
sig_coef_mat(:,:,2) = temp_mat_slope;
sig_coef_mat(:,:,1) = temp_mat_E50;

fprintf('Time fixing coefficients: %.2f sec\n', toc);
assignin('base','sig_coef_mat',sig_coef_mat);


load_fig(parameters_filename);
subplot(2,6, 1:3);
pscan_title = {p.title; ['Ex(ALL)->Inh(ALL)  | totEx_Inh = ' num2str(totEx_Inh) ' | SLOPE']};
plot_param_scan(temp_mat_slope, p, pscan_title, P1_start, P1_step_size, P1_steps, P2_start, P2_step_size, P2_steps, data_filename, parameters_filename, nr_repetitions, 0)

subplot(2,6, 4:6);
elapsed_time = etime(datevec(f_my_get_file_date(data_filename), 0), datevec(f_my_get_file_date(parameters_filename), 0))/3600;
pscan_title = {sprintf('pseudo-run-time:  %.1f hrs (%.2e hr/rep)', elapsed_time, elapsed_time/(P1_steps*P2_steps*nr_repetitions)); ' | E50'};
plot_param_scan(temp_mat_E50, p, pscan_title, P1_start, P1_step_size, P1_steps, P2_start, P2_step_size, P2_steps, data_filename, parameters_filename, nr_repetitions, f_my_absolute_min(temp_mat_E50))


% Appends this code parameters
param_txt{end+1} =  [];
p_txt = f_my_format_parameters_to_txt(p, 50);
param_txt = [param_txt, p_txt];

assignin('base','param_txt',param_txt);
f_my_print_txted_parameters(param_txt, 2, 3, 1, 1:3);


%%
if (process_IW)

    %nr_repetitions = 20; % # repetitions with same parameters
    %nr_inputs = ExINPUT;
    %nr_P1_steps = 5;
    %nr_P2_steps = 5;

    IW_file = load('-ascii', [IW_var_handle '.dat']);
    IW_file = single(IW_file);       % Makes it single
    %assignin('base','IW_file', IW_file);  % Puts loaded var in the workspace


    %expected_rows = nr_repetitions * ExINPUT * P1_steps * P2_steps;
    expected_rows = 1 * ExINPUT * P1_steps * P2_steps;  % Because I'm computing the average now
    if length(IW_file) ~= expected_rows; beep; beep; fprintf('THIS IS AN ERROR: Check my inputs! They don''t match length of IW file!!'); end;

    % Subtracts the columns. In this way negative values correspond to IW bigger than tstop and 0 corresponds to never above threshold
    IW_size_matrix = IW_file(:,2) - IW_file(:,1);
    %IW_matrix = reshape(IW_size_matrix, [nr_repetitions ExINPUT P2_steps P1_steps]);
    IW_matrix = reshape(IW_size_matrix, [1 ExINPUT P2_steps P1_steps]); % Because I'm processing the average now!
    assignin('base', 'IW_matrix', IW_matrix);

end


% Plots the PROBABILITIES OF FIRING plots
if (0)
    nr_plots = 3;
    for i = 1:nr_plots
        if length(inpts_idx_array) >= i
            nr_fibers_idx = inpts_idx_array(end)/ExINPUT_block_size - i + 1;
            plot_probabilities_of_firing(nr_fibers_idx, xscale_array, yscale_array, Ex_w_inputs_space_mat, ExINPUT_block_size, data_filename)
        end
    end
end


end % End Main function

%% 
function plot_probabilities_of_firing(nr_fibers_idx, xscale_array, yscale_array, Ex_w_inputs_space_mat, ExINPUT_block_size, data_filename)
%%% Plot PROBABILITIES OF FIRING
figure('Units','normalized', 'Position',[0.3 0.2 0.5 0.3], 'PaperPosition',[0.15 0.34 0.7 0.32], 'PaperUnits','normalized');
subplot(1,4,[1:3]);
imagesc(xscale_array, yscale_array, Ex_w_inputs_space_mat(:,:,nr_fibers_idx))
colorbar; set(gca,'ydir','normal');
title({[data_filename ' | Probabilities of firing']; ['nr inputs: ' num2str(nr_fibers_idx * ExINPUT_block_size)]}, 'Interpreter','none');
xlabel('Input->Ex'); ylabel('Inh->Ex');
end


%% Just puts empty fig on screen
function fig_h = load_fig(parameters_filename);
scrsz = get(0,'ScreenSize'); bdwidth = 4/scrsz(3); topbdwidth = 75/scrsz(4); w1 = 0.70; h1 = 0.7; x1 = bdwidth+0; y1 = scrsz(2)-h1-topbdwidth;
fig_h = figure('Units','normalized', 'Position',[x1 y1 w1 h1], 'PaperPosition',[0.08 0.1 0.8 0.80], 'PaperUnits','normalized');
% I'll be using parameters file date because it is more important the date the simulation started than when it ended!
annotation('textbox', 'String',{['Run date: ' f_my_get_file_date(parameters_filename)]}, 'Units','normalized', 'Position', [0.6, 0.97, 0.395 0.04], 'HorizontalAlignment','right', 'VerticalAlignment','middle', 'LineStyle','none', 'BackgroundColor',[0.9 0.9 0.9], 'Interpreter','none', 'FontSize',11);
end

%% Function to plot the parameter scan
function plot_param_scan(datamat_2D, p, pscan_title, P1_start, P1_step_size, P1_steps, P2_start, P2_step_size, P2_steps, data_filename, parameters_filename, nr_repetitions, clim_min)

% Builds scale vectors
xscale_array = P1_start:P1_step_size:P1_start+P1_step_size*(P1_steps-1);
yscale_array = P2_start:P2_step_size:P2_start+P2_step_size*(P2_steps-1);

% Plots the color matrix
imagesc(xscale_array, yscale_array, datamat_2D);

cmap = colormap(jet(p.colormap_total_colors));
cmap = [0 0 0; 0.3 0.3 0.3; cmap];
colormap(cmap); 
if (f_my_absolute_max(datamat_2D) > 0); set(gca, 'Clim', [clim_min f_my_absolute_max(datamat_2D)]); end;
colorbar('Location','SouthOutside');
set(gca, 'YDir','normal');
elapsed_time = etime(datevec(f_my_get_file_date(data_filename), 0), datevec(f_my_get_file_date(parameters_filename), 0))/3600;
xlabel({'Input -> Ex'; sprintf('pseudo-run-time:  %.1f hrs (%.2e hr/rep)', elapsed_time, elapsed_time/(P1_steps*P2_steps*nr_repetitions))});
ylabel('Inh -> Ex')
title(pscan_title, 'Interpreter','none')
end

%%
function sig_coef_mat = compute_sigmoids_with_inputs_in_X(Ex_w_inputs_space_mat, inpts_idx_array)

    % Computes sigmoid coefficients
    warning off
    fprintf('Fiting each point in matrix to sigmoid...\n');
    sig_coef_mat = zeros(size(Ex_w_inputs_space_mat,1), size(Ex_w_inputs_space_mat,2), 2);
    for Inh_i = 1:size(Ex_w_inputs_space_mat,1)
        for Ex_i = 1:size(Ex_w_inputs_space_mat,2)
            probs_firing = squeeze(Ex_w_inputs_space_mat(Inh_i,Ex_i,:));    % Array with probabilities of firing for different inputs

            yy = probs_firing;
            xx = inpts_idx_array';

            % If never fired for this W1.W2
            if length(find(yy == 0)) == length(probs_firing)
                sig_coef_mat(Inh_i, Ex_i, 1) = 0;
                sig_coef_mat(Inh_i, Ex_i, 2) = 0;   % SLOPE
                sig_coef_mat(Inh_i, Ex_i, 3) = 0;   % sigmoid Y value at last point
                sig_coef_mat(Inh_i, Ex_i, 4) = 0; % sigmoid Y value at first point
                % If fired for this W1.W2 compute sigmoid
            else
                b0=[mean(xx) 0.5];
                [beta,r,J]=nlinfit(xx,yy,@SIGMOID_C,b0); % Beta(2) is E50, beta(3) is sigmoid slope
                %beta
                sig_coef_mat(Inh_i, Ex_i, 1) = beta(1);
                sig_coef_mat(Inh_i, Ex_i, 2) = beta(2); % Sigmoid SLOPE
                sig_coef_mat(Inh_i, Ex_i, 3) = SIGMOID_C(beta, max(xx)); % sigmoid Y value at last point
                sig_coef_mat(Inh_i, Ex_i, 4) = SIGMOID_C(beta, min(xx)); % sigmoid Y value at first point
            end
        end
    end
end     % End Function



