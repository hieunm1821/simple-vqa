import torch
import torch.nn as nn
import torch.nn.functional as F




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.fc = nn.Linear(16 * 16 * 16, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.tanh(self.fc(x))
        return x

net = ConvNet()
x = torch.rand(1,3,64,64)
print(net(x))