function p = load_parameters_Dean_way(filename)

%%% GET HEADER VALUES %%%   Loads the parameters using Dean's W file
fprintf('Loading Parameter File: ''%s''\n', filename);
fid = fopen(filename,'r');
linecount = 0;
for i=1:99
   a=fgetl(fid);
   %fprintf('%s\n',a);
   if strcmp(a,'STOP') || ~ischar(a)       % EOF is not a string
       break;
   end
   
   % Strips white spaces
   a = a(~isspace(a));
   
   A = find(a=='=');
   %eval(['p.' a(1:A-1) '= str2num(a(A+1:length(a)));']);
   p.(a(1:A-1)) = str2double(a(A+1:length(a)));
   linecount = linecount+1;
end
fclose(fid);

end % End load_parameters_Dean_way