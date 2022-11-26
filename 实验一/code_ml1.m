%----------------code by Xiaoxingyingyingying--------------%
%-----------------注意本代码一定要分节运行，不懂的可以问作者---------
clear
train = importdata("trainData.txt");%加载数据集,如果数据和代码不在一个文件夹，需要修改一下，建议代码和数据放在一个文件夹
X = train(:,1:4);%特征
Y = train(:,5);%标签
test = importdata("testData.txt");%测试集
Y_temp = zeros(75,3);
for i=1:length(Y)
    if Y(i)==1
        Y_temp(i,:) = [1,0,0];
    elseif Y(i)==2
        Y_temp(i,:) = [0,1,0];
    elseif Y(i)==3
        Y_temp(i,:) = [0,0,1];
    end
end
Y = Y_temp;
%----------------------------------------数据预处理
%% 
test_data = load("testData.txt");
test_x = test_data(:,1:4);
test_y = test_data(:,5);
y_predict = results.Network(test_x');
y_predict_temp = zeros(75,1);
for j = 1:length(y_predict)
   [~,y_predict_temp(j)]  = max(y_predict(:,j));
end
y_predict = y_predict_temp;
%--------------------------------------预测，使用预测脚本前，先用神经网络模式识别工具箱训练网络
figure(1)
plot(linspace(1,75,75),test_y,'r-.',linspace(1,75,75),y_predict,'b-*')
legend('真实值','预测值')

