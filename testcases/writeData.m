function writeData(seq,states,filename)

fid = fopen(filename,'w');



for i=1:length(seq)
    seqLength = length(seq);
    fprintf(fid,[repmat('%d ',1,seqLength) '\n'], seq{i});
    fprintf(fid,'\n');
    fprintf(fid,[repmat('%d ',1,seqLength) '\n'], states{i});
    fprintf(fid,'\n\n');
end


fclose(fid);
end