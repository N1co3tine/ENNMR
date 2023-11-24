clear all
clc
close all

load AR_120_50_40.mat

addpath NMR_toolbox ;

EachClassNum = 26;
ClassNum = length(unique(sample_label)); %% 120 faces, 每张脸有26个图
p = 50; %% width
q = 40; %% height
occlusion_type = 'sunglasses'; %% occluison type

%% select training and test samples
% sample_data = sample_data ./ repmat(sqrt(sum(sample_data .* sample_data)), [size(sample_data, 1) 1]); %normalize
[train_data, train_label, test_data, test_label] = AR_sample_select(sample_data, sample_label, occlusion_type, EachClassNum, ClassNum);
%save('AR_sunglasses_50_40.mat','Train_DAT','Test_DAT')
%for i = [1:720]
%    pic = train_data(:,i);
%    pic = reshape(pic, p, q);
%    imwrite(uint8(pic), ['./img3/', num2str(i), '.png'])
%end
%% NMR
lambda = 1000;
beta = 0.001;
mu = 0.01;
% each Ztr: 960 * n, Etr: 2000 * n, n为样本数
tic;
%[Ztt, Ett] = ADMM_NMR(train_data, test_data, 120, lambda, p, q, mu);
[Ztt, Ett] = ENNMR(train_data, test_data, 120, lambda, beta, p, q, mu, 0);
toc
%% test
%test_result(Ztt, test_label)
