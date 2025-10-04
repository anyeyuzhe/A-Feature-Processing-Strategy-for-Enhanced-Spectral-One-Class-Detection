function [net] = aec2(xtrain,ecochs)
inputSize =size(xtrain,2); 
numSamples = size(xtrain,1); 
layers = [
    sequenceInputLayer(inputSize, 'Name', 'input')
    fullyConnectedLayer(50, 'Name', 'fc1') 
    % dropoutLayer(0.5)
    fullyConnectedLayer(20, 'Name', 'fc1')
    % fullyConnectedLayer(10, 'Name', 'latent')
    % dropoutLayer(0.5)
    % reluLayer('Name', 'latent_relu')
    % fullyConnectedLayer(20, 'Name', 'fc1')
    % dropoutLayer(0.5)
    fullyConnectedLayer(50, 'Name', 'fc1')
    fullyConnectedLayer(inputSize, 'Name', 'output')
    regressionLayer
];
options = trainingOptions('adam', ...
    'MaxEpochs', ecochs, ... 
    'InitialLearnRate', 0.001, ...
    'Verbose', false);
net = trainNetwork(xtrain', xtrain', layers, options);
% xtestc = predict(net, xtest');
end
