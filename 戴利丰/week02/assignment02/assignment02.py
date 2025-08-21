import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
x = torch.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
y = torch.sin(x) + torch.randn_like(torch.sin(x)) * 0.05

print("数据生成完成。")
print("---" * 10)

class SinNet(nn.Module):
    def __init__(self, hidden_size=100):
        super(SinNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 120),  # 输入层→隐藏层
            nn.ReLU(),  # 非线性激活
            nn.Linear(120, 120),  # 输入层→隐藏层
            nn.ReLU(),
            nn.Linear(120, 60),  # 输入层→隐藏层
            nn.ReLU(),
            nn.Linear(60, 30),  # 隐藏层→隐藏层
            nn.ReLU(),
            nn.Linear(30, 1)  # 隐藏层→输出层
        )

    def forward(self, x):
        return self.net(x)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
model = SinNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 4. 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    # losses.append(loss.item())

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 5. 打印最终学到的参数
print("\n训练完成！")

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值

model.eval()

with torch.no_grad():
    y_predicted = model(x).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y.numpy(), label='Raw data', color='blue', alpha=0.6)
plt.plot(x.numpy(), y_predicted, label=f'Model Predicted', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
