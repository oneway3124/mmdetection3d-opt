import torch
import torch.nn as nn


# 定义全连接神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建一个简单的训练数据集
x_train = torch.randn(10, 5)
y_train = torch.randn(10, 2)

# 初始化模型和优化器
model = SimpleNet(input_size=5, hidden_size=10, output_size=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 正向传递和反向传递的训练过程
for i in range(1000):
    # 正向传递
    y_pred = model(x_train)

    # 计算损失函数和误差
    loss = nn.MSELoss()(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()

    # 反向传递
    optimizer.step()

    # 打印损失值
    print('Epoch: {}, Loss: {:.4f}'.format(i + 1, loss.item()))
