%% Params
nrExamples  = 100;
sequenceLen = [5 20];


%% Example 1: 2 states
paramsFile   = 'Example_1_gt.txt'; % The parameters file
dataFile     = 'Example_1.txt';

T = [0.27, 0.73;...
     0.42 0.58];
E = [ones(1,10)/10;...
    1/5 1/5 1/10 1/10 1/20 1/20 1/20 1/10 1/10  1/20];
seq      = cell(1,nrExamples);
states   = cell(1,nrExamples);
for i=1:nrExamples
    [seq{i},states{i}] = hmmgenerate(randi(sequenceLen),T,E);
end


writeHMMparams(T,E,paramsFile); % Write the parameters used to generate the sequences
writeData(seq,states,dataFile); % Write the data generated



