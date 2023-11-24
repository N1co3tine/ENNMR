clear all
clc
close all

load yaleB.mat

addpath NMR_toolbox ;

EachClassNum = 26;
ClassNum = 38; %% 120 faces, 每张脸有26个图
p = 192/3; %% width
q = 168/3; %% height

%% NMR
lambda = 1000;
beta = 0.001;
mu = 0.01;

% each Ztr: 960 * n, Etr: 2000 * n, n为样本数
tic;
%[Ztt, Ett] = ADMM_NMR(train_data, test_data, 120, lambda, p, q, mu);
[Ztt, Ett] = ENNMR(X1, X4, 31, lambda, beta, p, q, mu, 1);
toc
%% test
%test_result(Ztt, test_label)
