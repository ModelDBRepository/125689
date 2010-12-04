% sigmoidal fit function for iES curves-- CONSTRAINED for b1=1

%b1 is asymptote (max spike prob)
%b2 is E50 (inflection pt or threshold)--epsp at which SpikeProb is 50% 
%b3 reflects slope (span or horizontal stretch)

function S = SIGMOID_C(b,x);
	%b1 = b(1); % Assymptote
	b2 = b(1);  % E50
    b3 = b(2);  % Slope
	S = 1 ./ (1 + exp((b2 - x)/b3)); 