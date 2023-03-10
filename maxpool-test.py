import torch
import torch.nn as nn

# 定义最大池化层
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 输入特征图大小为 4x4x3
x = torch.randn(1, 3, 4, 4)

# 进行最大池化操作
out = max_pool(x)

# 输出特征图大小为 2x2x3
print(out.shape)
