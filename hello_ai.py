import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 准备训练数据 (AI的课本)
# 隐藏规律是 y = 2x + 1
# ==========================================
# X 是输入特征 (比如 1, 2, 3...)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
# y 是真实答案 (比如 3, 5, 7...)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

# ==========================================
# 2. 搭建极简神经网络 (AI的大脑)
# ==========================================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个最基础的线性层，内部包含两个需要学习的参数：权重(w)和偏置(b)
        # 它天生就会做运算：输出 = w * 输入 + b
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 实例化这个模型
model = SimpleModel()

# ==========================================
# 3. 设置“考核标准”和“学习方法”
# ==========================================
# 损失函数 (Loss)：用均方误差 (MSE) 来计算AI的预测值和真实答案差了多少
criterion = nn.MSELoss() 
# 优化器 (Optimizer)：使用随机梯度下降 (SGD)，学习率(步长)设为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==========================================
# 4. 开始闭环训练！(核心引擎)
# ==========================================
epochs = 2000 # 让AI把这套数据反复学 200 遍
print("🚀 AI 开始训练，正在寻找数据背后的规律...")

for epoch in range(epochs):
    # 步骤A (前向传播)：让模型根据当前的参数猜一猜输出是什么
    predictions = model(X)
    
    # 步骤B (计算误差)：对比模型的猜测和真实答案，得到误差信号
    loss = criterion(predictions, y)
    
    # 步骤C (反向传播与优化)：
    optimizer.zero_grad() # 1. 清空上一步残余的梯度
    loss.backward()       # 2. 根据误差信号，反向求导计算出每个参数该往哪个方向调
    optimizer.step()      # 3. 真正去修改模型内部的权重(w)和偏置(b)
    
    # 每 40 遍打印一次学习进度
    if (epoch + 1) % 40 == 0:
        print(f"第 {epoch+1:3d} 轮 | 当前误差 (Loss): {loss.item():.4f}")

# ==========================================
# 5. 最终考试！
# ==========================================
print("\n🎉 训练结束！我们来考考它：")
# 我们给它一个从来没见过的输入: 6.0
test_X = torch.tensor([[100.0]])
predicted_y = model(test_X)

print(f"题目：当输入 x=6 时，答案是多少？")
print(f"绝对真理：13.0000 (因为 2*6+1 = 13)")
print(f"AI的回答：{predicted_y.item():.4f}")