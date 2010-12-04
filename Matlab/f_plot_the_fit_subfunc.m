function line_handle = plot_the_fit_subfunc(IO_cell_matrix, sig_coef_mat, plot_data, Inh, Ex, plot_color, linewidth, marker, markersize)
   
    xx = plot_data{Inh,Ex,1};
    yy = plot_data{Inh,Ex,2};
    beta = plot_data{Inh,Ex,3};
    
    xxx = [min(xx):0.002:max(xx)];
    zz = f_SIGMOID_C(beta, xxx);
    %plot(xxx, zz)
    %colororder = get(gca, 'ColorOrder');
    %plot_color = colororder(rem(Inh+Ex, size(colororder,1))+1,:);
    plot(xx, yy, 'LineStyle','none', 'Marker',marker, 'MarkerSize',markersize, 'MarkerEdgeColor',plot_color, 'MarkerFaceColor',plot_color); % 'LineWidth',1, 
	line_handle = plot(xxx, zz, 'Color',plot_color, 'LineWidth',linewidth);

 
    fprintf(['     in matrix: ' num2str(sig_coef_mat(Inh,Ex,2)) '\n']);
    fprintf(['beta(2)(slope): ' num2str(beta(2)) '     beta(1): ' num2str(beta(1)) '\n']);
end % End plot_the_fit_subfunc