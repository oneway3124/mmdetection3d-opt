import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net = Net()
x = torch.randn(1, 10)
output = net(x)
print(output.shape)
