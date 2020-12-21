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
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.tanh(self.fc(x))
        return x

class QNet(nn.Module):
    def __init__(self, vocab_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(vocab_size, 32) # vec_size
        self.fc2 = nn.Linear(32, 32)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class MergeNet(nn.Module):
    def __init__(self, vocab_size, num_ans):
        super(MergeNet, self).__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, num_ans)
        self.Conv = ConvNet()
        self.Q = QNet(vocab_size)
    def forward(self, image, question):
        x = torch.mul(self.Conv(image), self.Q(question))
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# net = MergeNet(26, 13)
# qn = torch.rand(1,26)
# im = torch.rand(1,3,64,64)
# print(net(im,qn))