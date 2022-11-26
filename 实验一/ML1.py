import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
def getdata():
    train = np.loadtxt('trainData.txt')
    test = np.loadtxt('testData.txt')
    train_x = torch.tensor(train[...,0:4],dtype=torch.float32)
    train_y = torch.tensor(train[...,4],dtype=torch.float32)
    train_y = train_y.view(75,1)
    test_x = torch.tensor(test[...,0:4],dtype=torch.float32)
    test_y = torch.tensor(test[...,4],dtype=torch.float32)
    test_y = test_y.view(75,1)
    return train_x, train_y,test_x,test_y

train_x, train_y, test_x, test_y = getdata()
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out =self.predict(out)
        return out
net = Net(4,10,1)


optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
loss_func = torch.nn.MSELoss()

loss_log = np.zeros((1000,1))
for epoch in range(1000):
    loss_mean = 0
    for i in range(75):
        prediction = net(train_x[i,...])
        loss = loss_func(prediction,train_y[i,...])
        loss_mean = loss_mean+loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_log[epoch] = loss_mean.detach().numpy()
    if epoch % 5 ==0:
        print(f'epoch ============{epoch}\t\tLoss = %.4f'% loss_mean)

prediced_y = net(test_x)
plt.figure(1)
plt.plot(prediced_y.detach().numpy())
plt.plot(test_y)
plt.figure(2)
plt.plot(loss_log)
plt.show()


