
%%% Generate the training data.

clear;close all;

addpath(genpath('./.'));
%addpath utilities;

batchSize      = 128;        %%% batch size
max_numPatches = batchSize*800; 
modelName      = 'diff_net\model_64_25_Res_Bnorm_Adam';


%%% training and testing
folder_train  = 'F:\DnCNN-master15\DnCNN-master\TrainingCodes\DnCNN_TrainingCodes_v1.0\diff_net_clips\Train';  %%% training
folder_test   = 'F:\DnCNN-master15\DnCNN-master\TrainingCodes\DnCNN_TrainingCodes_v1.0\diff_net_clips\Test';%%% testing
size_input    = 40;          %%% training
size_label    = 40;          %%% testing
stride_train  = 20;          %%% training
stride_test   = 40;          %%% testing
val_train     = 0;           %%% training % default
val_test      = 1;           %%% testing  % default

%%% training patches
[inputs, labels, set]  = patches_generation_for_diff_net(size_input,size_label,stride_train,folder_train,val_train,max_numPatches,batchSize);
%%% testing  patches
[inputs2,labels2,set2] = patches_generation_for_diff_net(size_input,size_label,stride_test,folder_test,val_test,max_numPatches,batchSize);

inputs   = cat(4,inputs,inputs2);      clear inputs2;
labels   = cat(4,labels,labels2);      clear labels2;
set      = cat(2,set,set2);            clear set2;

if ~exist(modelName,'file')
    mkdir(modelName);
end

%%% save data
save(fullfile(modelName,'imdb'), 'inputs','labels','set','-v7.3')

