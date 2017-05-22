% compute features with kaldi functions, create hdf5
clc
clear all
close all

load('labels_num.mat')

mode = 'test';
dset_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits_iso/';
temp_dir = [dset_path, 'temp/'];
try
    mkdir(temp_dir);
end


file_name = [mode,'_fbank.h5'];
s=dir([dset_path, mode, '/*.wav']);

% Load labels
if strcmp(mode, 'train') == 1
    data = load([dset_path, 'tidigits_mfccs_train.mat']);
    labels = data.tidigits_mfccs_train.idx_labels;
elseif strcmp(mode, 'test') == 1
    data = load([dset_path, 'tidigits_mfccs_test.mat']);
    labels = data.tidigits_mfccs_test.idx_labels;
end
% Convert Labels from Digits to Chars

samples = length(labels);
labels_char =[];
for i=1:1:samples
    curr_label_char = labels_num{labels(i)+1};
    label_lens_char(i) = length(curr_label_char);
    labels_char = horzcat(labels_char, curr_label_char);
end

keys= 1:1:samples;
% label_lens = ones(1, samples);


%% Create hdf5
if exist(file_name, 'file')==2
  delete(file_name);
end
h5create(file_name, '/features', [123 Inf], 'ChunkSize', [123 8192], 'Datatype', 'single');
h5create(file_name, '/feature_lens', samples)
h5create(file_name, '/labels', length(labels_char));
h5create(file_name, '/label_lens', samples);
% h5create(file_name, '/keys', samples);
h5create(file_name, '/keys', [samples 1]);


h5write(file_name, '/labels', labels_char);
h5write(file_name, '/label_lens', label_lens_char);
% h5write(file_name, '/keys', keys);

start=1;
for i=1:1:samples
    curr_key = double(keys(i));
    
    % print control file for compute-mfcc-feats
    fileID = fopen([temp_dir,'wav.scp'],'w');
    fprintf(fileID, ['1 ', dset_path, mode, '/', mode, '_',num2str(i), '.wav']);
    fclose(fileID);  
    
    % compute filter bank features
    %mfcc
%     command = sprintf('compute-mfcc-feats --sample-frequency=20000 scp,p:%swav.scp ark:%sfeats', temp_dir, temp_dir);
    %fbank
    command = sprintf('compute-fbank-feats --num-mel-bins=40 --use-energy=True --sample-frequency=20000 --high-freq=8000 scp,p:%swav.scp ark:%sfeats', temp_dir, temp_dir);

    [~,~] = system(command, '-echo');
    
    % add deltas
    command = sprintf('add-deltas ark:%sfeats ark:%sfeats_deltas', temp_dir, temp_dir);
    [~,~] = system(command);
    
    % read ark file, convert to single precision (save space) 
    [~, FEATURE_MAT]=arkread([temp_dir,'feats_deltas']);
    logf_mat = single(FEATURE_MAT');
    
    h5write(file_name, '/features', logf_mat, [1 start], size(logf_mat)) % write features
    h5write(file_name, '/feature_lens', length(logf_mat(1,:)), i,1) % write lenghts of samples
    % update hdf5 indices for features
    start = start+length(logf_mat(1,:));
    h5write(file_name, '/keys', curr_key, [i 1], size(curr_key))

    
end
