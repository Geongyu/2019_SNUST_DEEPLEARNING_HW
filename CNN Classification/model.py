
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=0)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2,2))
       # import pdb; pdb.set_trace()
        x = F.max_pool2d(self.conv2(x), (2,2))
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = x
        return output

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1   = nn.Linear(1*32*32, 56)
        self.fc2   = nn.Linear(56, 44)
        self.fc3   = nn.Linear(44, 32)
        self.fc4   = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        output = x
        return output

if __name__ == "__main__" :
    from torchsummary import summary
    net = LeNet5()
    summary(net.cuda(), input_size=(1, 32, 32), batch_size=1)
    net = CustomMLP()
    summary(net.cuda(), input_size=(1, 32, 32), batch_size=1)
