import torch
import torch.nn as nn


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义数据集
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# 初始化模型、损失函数和优化器
model = MLP(input_size=10, hidden_size=20, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10000):
    # 前向传播
    y_pred = model(x)

    # 计算损失函数
    loss = criterion(y_pred, y)

    # 反向传播并更新梯度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失函数值
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, loss.item()))
