function writeHMMparams(T,E,filename)

fid = fopen(filename,'w');

nrStates     = size(E,1);
nrEmissions  = size(E,2);

fprintf(fid,[repmat('%1.2f ',1,nrStates) '\n'],T');
fprintf(fid,'\n');
fprintf(fid,[repmat('%1.2f ',1,nrEmissions) '\n'],E');   
    
fclose(fid);

end