

clear

%% 

net = newrb(IrisData',Label');
%训练网络
%% 
Y_predicted = net(IrisData')

