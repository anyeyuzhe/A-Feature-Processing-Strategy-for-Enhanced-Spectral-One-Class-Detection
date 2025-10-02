function [net] = aec2(xtrain,ecochs)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
inputSize =size(xtrain,2); % 输入层大小
numSamples = size(xtrain,1); % 样本数量

% 步骤 2：定义网络架构
layers = [
    sequenceInputLayer(inputSize, 'Name', 'input') % 输入层
    fullyConnectedLayer(50, 'Name', 'fc1') % 第一个隐藏层
    % dropoutLayer(0.5)
    fullyConnectedLayer(20, 'Name', 'fc1') % 第一个隐藏层
    % fullyConnectedLayer(10, 'Name', 'latent') % 瓦力层
    % dropoutLayer(0.5)
    % reluLayer('Name', 'latent_relu') % ReLU激活函数
    % fullyConnectedLayer(20, 'Name', 'fc1') % 第一个隐藏层
    % dropoutLayer(0.5)
    fullyConnectedLayer(50, 'Name', 'fc1') % 第一个隐藏层
    fullyConnectedLayer(inputSize, 'Name', 'output') % 输出层
    regressionLayer
];

% 步骤 3：定义训练参数
options = trainingOptions('adam', ...
    'MaxEpochs', ecochs, ... % 最大训练轮数
    'InitialLearnRate', 0.001, ... % 初始学习率
    'Verbose', false);

% 步骤 4：训练自编码器
net = trainNetwork(xtrain', xtrain', layers, options);
% xtestc = predict(net, xtest');

end