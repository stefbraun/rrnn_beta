% converts tidigits mat files to wav files
clc
clear all
close all

dset_path = '/media/stefbraun/ext4/Dropbox/dataset/tidigits/';
try
    mkdir([dset_path,'train'])
    mkdir([dset_path,'test'])
end

train = load([dset_path, 'tidigits_mfccs_train.mat']);
test = load([dset_path, 'tidigits_mfccs_test.mat']);

train_samples = length(train.tidigits_mfccs_train.wavs)
test_samples = length(test.tidigits_mfccs_test.wavs)

for i=1:1:train_samples
    file_name = [dset_path,'train/train_', num2str(i), '.wav']
    wav = double(train.tidigits_mfccs_train.wavs{i});
    wav = wav./max(abs(wav));
    audiowrite(file_name, wav, 20000);
end

for i=1:1:test_samples
    file_name = [dset_path,'test/test_', num2str(i), '.wav']
    wav = double(test.tidigits_mfccs_test.wavs{i});
    wav = wav./max(abs(wav));
    audiowrite(file_name, wav, 20000);
end
