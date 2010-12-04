%% CARVALHO & BUONOMANO 
%% NEURON 2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run 'analize_synaptic_space' FIRST, and then run this


% Info to draw the little squares
coord = [34 10;               % Coordinates    [X Y]
         26 10;
         18 10;
         32 21;
         17 5;
		 26 1;
        ]; 


run_label = '';
p = f_load_parameters_Dean_way(['SynSpace_Param' run_label '.dat']);


p.script_dir = '';
p.print_filename = 'F1_ParamScan.tif';

p.colormap_total_colors = 250;

p.AxesTitleFontSize = 11;
p.AxesLabelsFontSize = 8;
p.AxesNumbersFontSize = 6;
p.markersize = 4;
p.LineWidth = 1; % Param Scan 
p.SqLineWidth = 1.5; % Square Line Width
p.IOLineWidth = 2;
p.IOAxisLineWidth = 1;


% Builds scale vectors
xscale_array = p.P1_start:p.P1_step_size:p.P1_start+p.P1_step_size*(p.P1_steps-1);
yscale_array = p.P2_start:p.P2_step_size:p.P2_start+p.P2_step_size*(p.P2_steps-1);



p.coord_colors = [	0.98 0.98 0;
					0 0 1;
					0 0.5 0
					0.64 0.43 0.94;
					1 0.6 0;
					1 0 0;
					];


% [1 0.6 0] - Orange
% [0.98 0.98 0] - Yellow
% [0.64 0.43 0.94] - Purple
% [0.6 0.4 0] - brown
p.marker = ['o','o','o','o','o', 'o'];
%p.number = ['3','2','1','5','4', '6'];
p.pscanMarkerSize = 5;
p.pscanMarkerWidth = 1.3;

p.p1L = 0.12;
p.p2L = 0.57;
p.p1Y = 0.44;
p.p2Y = 0.11;
p.p1H = 0.46;
p.p2H = 0.18;
p.p1W = 0.30;
p.p2W = p.p1W;

p.cbL = 0.01;
p.cbW = 0.02;

															   %0.45
h_fig_plots = figure('Visible','on', 'PaperPosition',[0.20 0.15 0.55 0.28], 'PaperUnits','normalized', 'Color','w');

sp_gain = subplot('Position', [p.p1L p.p1Y p.p1W p.p1H]); hold on;			
sp_thr = subplot('Position', [p.p2L p.p1Y p.p2W p.p1H]); hold on;
sp_IO_left = subplot('Position', [p.p1L p.p2Y p.p1W p.p2H]); hold all;
sp_IO_right =subplot('Position', [p.p2L p.p2Y p.p2W p.p2H]); hold all;


if (1) % Draw the mini sigmoids, next to the colorbar
dm.L = p.p1L+p.p1W+p.cbL+p.cbW+0.01;
dm.W = 0.05;
dm.H = 0.05;
dm.Y1 = p.p1Y + p.p1H-dm.H;
dm.Y2 = p.p1Y + (p.p1H-dm.H)/2;
dm.Y3 = p.p1Y;

xxLim = [5 20];
xx = xxLim(1):(xxLim(2)-xxLim(1))/1000:xxLim(2);
beta(1) = 12;  % SP50
sp_sig(1) = subplot('Position', [dm.L dm.Y1 dm.W dm.H]); hold on;
beta(2) = 3;     % Dynamic Range, k
yy = f_SIGMOID_C(beta, xx);
h(1) = plot(xx,yy, 'Color',[0.8 0 0]);

sp_sig(2) = subplot('Position', [dm.L dm.Y2 dm.W dm.H]); hold on;
beta(2) = 1.2;     % Dynamic Range, k
yy = f_SIGMOID_C(beta, xx);
h(2) = plot(xx,yy,'g');

sp_sig(3) = subplot('Position', [dm.L dm.Y3 dm.W dm.H]); hold on;
beta(2) = 0.4;     % Dynamic Range, k
yy = f_SIGMOID_C(beta, xx);
h(3) = plot(xx,yy, 'Color',[0 0 0.8]);


beta(2) = 0.6;     % Dynamic Range, k
dm.L = p.p2L+p.p2W+p.cbL+p.cbW+0.01;
sp_sig(4) = subplot('Position', [dm.L dm.Y1 dm.W dm.H]); hold on;
beta(1) = 15.5;  % SP50
yy = f_SIGMOID_C(beta, xx);
h(4) = plot(xx,yy, 'Color',[0.8 0 0]);
f_my_arrow([8,0.5],[13,0.5],'FaceColor',[0.8 0 0], 'EdgeColor',[0.8 0 0], 'Length',4, 'TipAngle',40, 'Width',2);

sp_sig(5) = subplot('Position', [dm.L dm.Y2 dm.W dm.H]); hold on;
beta(1) = 13;  % SP50
yy = f_SIGMOID_C(beta, xx);
h(5) = plot(xx,yy,'g');

sp_sig(6) = subplot('Position', [dm.L dm.Y3 dm.W dm.H]); hold on;
beta(1) = 9.5;  % SP50
yy = f_SIGMOID_C(beta, xx);
h(6) = plot(xx,yy, 'Color',[0 0 0.8]);
f_my_arrow([17,0.5],[12,0.5],'FaceColor',[0 0 0.8], 'EdgeColor',[0 0 0.8], 'Length',4, 'TipAngle',40, 'Width',2)

set(sp_sig, 'XTick',[],'YTick',[],'Visible','off','YLim',[-0.05 1.05],'XLim',[xxLim(1) xxLim(2)]);
set(h, 'LineWidth',1);
clear('sp_sig'); clear('dm'); clear('h');
end



%% Plots Gain color matrix



subplot(sp_gain);
imagesc(xscale_array, yscale_array, sig_coef_mat(:,:,2));

cmap = colormap(jet(p.colormap_total_colors));
cmap = [0 0 0; 0.3 0.3 0.3; cmap];
colormap(cmap); %if (max_sig_slope > 0); set(gca, 'Clim', [0 max_sig_slope]); end;
h_cbar = colorbar; 
set(h_cbar, 'FontSize',p.AxesLabelsFontSize, 'fontweight','bold', 'Position', [p.p1L+p.p1W+p.cbL p.p1Y p.cbW p.p1H], 'YTick',[]); % 1 10


% Makes the little squares, can specify in coordinates or in strength values
dm.circleX = p.P1_step_size*1.5;
dm.circleY = p.P2_step_size*1.5;
for i = 1:size(coord,1)
%  	rectangle('Position',[xscale_array(coord(i,1))-dm.circleX/2, yscale_array(coord(i,2))-dm.circleY/2, dm.circleX, dm.circleY], 'Curvature',[1 1], 'LineWidth',p.SqLineWidth, 'EdgeColor',p.coord_colors(i,:));
  	plot(xscale_array(coord(i,1)), yscale_array(coord(i,2)), 'LineStyle','none','LineWidth',p.pscanMarkerWidth, 'MarkerEdgeColor',p.coord_colors(i,:), 'MarkerSize',p.pscanMarkerSize, 'Marker', p.marker(i));
% 	text(xscale_array(coord(i,1))-p.P1_step_size, yscale_array(coord(i,2))+0.01, p.number(i), 'FontWeight','bold', 'FontSize',8, 'Color',p.coord_colors(i,:), 'FontName','ArialBlack');

end

set(gca, 'YDir','normal', 'YTick',[0 0.2 0.4 0.6], 'XTick',[0.02 0.03 0.04], 'XLim', [xscale_array(1)-p.P1_step_size/2 xscale_array(end)+p.P1_step_size/2], 'YLim', [yscale_array(1)-p.P2_step_size/2 yscale_array(end)+p.P2_step_size/2], 'LineWidth',p.LineWidth);
set(gca, 'FontSize',p.AxesNumbersFontSize, 'FontWeight','bold');

title('Gain^{-1}', 'FontSize',p.AxesTitleFontSize, 'fontweight','bold');
xlabel({['Input \rightarrow Ex (\muS)']}, 'FontSize',p.AxesLabelsFontSize);
ylabel('Inh \rightarrow Ex (\muS)', 'FontSize',p.AxesLabelsFontSize)



%% Prints the Threshold Parameter Scan Plot

subplot(sp_thr);
imagesc(xscale_array, yscale_array, sig_coef_mat(:,:,1));


cmap = colormap(jet(p.colormap_total_colors));
cmap = [0 0 0; 0.3 0.3 0.3; cmap];
colormap(cmap); %if (max_sig_slope > 0); set(gca, 'Clim', [0 max_sig_slope]); end;
h_cbar = colorbar; 
set(h_cbar, 'Position', [p.p2L+p.p2W+p.cbL p.p1Y p.cbW p.p1H], 'FontSize',p.AxesLabelsFontSize, 'fontweight','bold', 'YTick',[]); % 11 19

% Makes the little squares, can specify in coordinates or in strength values
for i = 1:size(coord,1)
% 	rectangle('Position',[xscale_array(coord(i,1))-dm.circleX/2, yscale_array(coord(i,2))-dm.circleY/2, dm.circleX, dm.circleY], 'Curvature',[1 1], 'LineWidth',p.SqLineWidth, 'EdgeColor',p.coord_colors(i,:));
 	plot(xscale_array(coord(i,1)), yscale_array(coord(i,2)), 'LineStyle','none','LineWidth',p.pscanMarkerWidth, 'MarkerEdgeColor',p.coord_colors(i,:), 'MarkerSize',p.pscanMarkerSize, 'Marker', p.marker(i));
% 	text(xscale_array(coord(i,1))-p.P1_step_size, yscale_array(coord(i,2))+0.01, p.number(i), 'FontWeight','bold', 'FontSize',8, 'Color',p.coord_colors(i,:), 'FontName','ArialBlack');

end


set(gca, 'YTick',[0 0.2 0.4 0.6], 'YDir','normal',  'XTick',[0.02 0.03 0.04],'XLim', [xscale_array(1)-p.P1_step_size/2 xscale_array(end)+p.P1_step_size/2], 'YLim', [yscale_array(1)-p.P2_step_size/2 yscale_array(end)+p.P2_step_size/2], 'LineWidth',p.LineWidth);
set(gca, 'FontSize',p.AxesNumbersFontSize, 'FontWeight','bold');

title('Threshold', 'FontSize',p.AxesTitleFontSize, 'fontweight','bold');
xlabel({'Input \rightarrow Ex (\muS)'}, 'FontSize',p.AxesLabelsFontSize); 




%% PLOT THE IOs

                 
nr_inputs = 20;
subplot(sp_IO_left);


conditions_to_plot_here = [1 2 3];
conditions_to_plot = coord(conditions_to_plot_here,:);
colororder = p.coord_colors;


for i = size(conditions_to_plot_here,2):-1:1
    plot_color = colororder(conditions_to_plot_here(i),:);
    Ex = conditions_to_plot(i,1);
    Inh = conditions_to_plot(i,2);
    line_handle = f_plot_the_fit_subfunc(IO_cell_matrix, sig_coef_mat, plot_data, Inh, Ex, plot_color, p.IOLineWidth, p.marker(conditions_to_plot_here(i)), p.markersize);
end

h_IOaxes(1) = gca;
set(gca,'YTick',[0 1]);
xlabel({'EPSP slope (mV/ms)'}, 'FontSize',p.AxesLabelsFontSize, 'FontWeight','bold');
ylabel({'Probability'; 'Ex Firing'}, 'FontSize',p.AxesLabelsFontSize, 'FontWeight','bold');


%%

subplot(sp_IO_right);

conditions_to_plot_here = [4 2 5];
conditions_to_plot = coord(conditions_to_plot_here,:);
colororder = p.coord_colors;


for i = 1:size(conditions_to_plot_here,2)
    plot_color = colororder(conditions_to_plot_here(i),:);
    Ex = conditions_to_plot(i,1);
    Inh = conditions_to_plot(i,2);
    line_handle = f_plot_the_fit_subfunc(IO_cell_matrix, sig_coef_mat, plot_data, Inh, Ex, plot_color, p.IOLineWidth, p.marker(conditions_to_plot_here(i)), p.markersize);
end
h_IOaxes(2) = gca;
set(gca,'YTick',[0 1]);
xlabel({'EPSP slope (mV/ms)'}, 'FontSize',p.AxesLabelsFontSize, 'FontWeight','bold'); 
set(h_IOaxes, 'XLim',[8 21], 'YAxisLocation','right', 'YLim',[-0.04 1.04], 'FontSize',p.AxesNumbersFontSize, 'FontWeight','bold', 'LineWidth',p.IOAxisLineWidth, 'Box','off');





print('-dtiff', '-r300', [p.script_dir p.print_filename]);
winopen([p.script_dir p.print_filename]);






return

