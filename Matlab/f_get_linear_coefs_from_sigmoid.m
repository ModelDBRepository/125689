function coeffs = get_linear_coefs_from_sigmoid(xx, yy, p)
% Gets the 'a' and 'b' coefficients from the linear portion of a sigmoid function
% 
% 14th May 2007 - Returns also the *REAL* Dynamic Range (between 0.25 and 0.75, for instance)
% 30th May 2007 - If input sigmoid does not contain full range will return zeros (as opposed to erroring as before)
% 16th Jul 2008 - Makes sure there are enough points in the linear range to compute 'dr','a','b'

if ~isfield(p, 'linear_bound'); error('### Please specify Y limits withing which sigmoid is considered linear!'); end;

% Makes sure that the sigmoid has a complete linear portion, otherwise return with 0s
if ~any(yy <= p.linear_bound(1)) || ~any(yy >= p.linear_bound(2))
    %error('### Sigmoid does NOT contain a full linear range!');
    coeffs.dr = 0;
    coeffs.a = 0;
    coeffs.b = 0;
    return;
end

% Finds the linear portion of the sigmoid
linear_idxs = yy > p.linear_bound(1) & yy < p.linear_bound(2);

% Makes sure the linear portion of the sigmoid has enought points to compute 'dr','a','b'
if nnz(linear_idxs) < 2
	coeffs.dr = 0.000000000001;
    coeffs.a = 100000000000000;
    coeffs.b = -10000000000000;
    return;
end
linear_XX = xx(linear_idxs);
linear_YY = yy(linear_idxs);

% Gets Dynamic Range right here
coeffs.dr = linear_XX(end) - linear_XX(1);% + (linear_XX(2) - linear_XX(1));  % Corrects for the extra point

% Makes the linear fit
pfit = polyfit(linear_XX, linear_YY, 1);
coeffs.a = pfit(1);
coeffs.b = pfit(2);


end