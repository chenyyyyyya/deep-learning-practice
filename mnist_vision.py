import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 数据流水线 (Data Pipeline)
# ==========================================
print("📥 正在下载/加载 MNIST 数据集...")
# 将图片转换为 PyTorch 能看懂的 Tensor，并将像素值归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载训练集和测试集（会自动保存在当前目录的 data 文件夹下）
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 负责把成千上万张图片打包成批次 (Batch) 喂给模型
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 2. 搭建视觉大脑 (Neural Network)
# ==========================================
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一步：把 28x28 的二维矩阵展平成 784 的一维向量
        self.flatten = nn.Flatten()
        
        # 第二步：核心网络架构 (包含两个隐藏层，类似于复杂的状态空间变换)
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128), # 输入 784，映射到 128 维的高维空间
            nn.ReLU(),               # 非线性激活函数 (系统的非线性环节，让模型具备学习复杂特征的能力)
            nn.Linear(128, 64),      # 128 维降到 64 维
            nn.ReLU(),
            nn.Linear(64, 10)        # 输出 10 维，对应 0-9 十个数字
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 实例化大脑
model = DigitRecognizer()

# ==========================================
# 3. 设置“考核标准”和“学习方法”
# ==========================================
# 交叉熵损失函数：分类任务的绝对标配，计算概率分布的差异
criterion = nn.CrossEntropyLoss()
# Adam 优化器：比之前用的 SGD 更聪明的自适应控制器，能自动调节学习步长
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. 开启闭环训练 (核心引擎)
# ==========================================
epochs = 3 # 为了快速验证，我们只让它过 3 遍完整的数据集
print("\n🚀 视觉大脑开始训练！")

for epoch in range(epochs):
    model.train() # 切换到训练模式
    running_loss = 0.0
    
    # 从流水线中不断拿取 Batch 数据 (images 是图片，labels 是真实数字)
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 步骤A：前向预测
        predictions = model(images)
        # 步骤B：计算误差
        loss = criterion(predictions, labels)
        
        # 步骤C：反向传播与参数优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # 每处理 300 个批次打印一次进度
        if batch_idx % 300 == 299:
            print(f"第 {epoch+1} 轮 | 批次 {batch_idx+1:4d} | 当前误差 (Loss): {running_loss/300:.4f}")
            running_loss = 0.0

# ==========================================
# 5. 最终期末考试！(在测试集上评估准确率)
# ==========================================
print("\n🎓 训练结束，正在进行全国统考 (测试集评估)...")
model.eval() # 切换到评估模式 (关闭梯度计算，省内存)
correct = 0
total = 0

with torch.no_grad(): # 考试时不需要再学习(反向传播)了
    for images, labels in test_loader:
        outputs = model(images)
        # 找出 10 个输出概率中最大的那个，作为预测答案
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎉 最终成绩：AI 识别手写数字的准确率为 {accuracy:.2f}%！")
# ==========================================
# 6. 封印大脑记忆 (保存模型权重)
# ==========================================
print("\n🧠 正在将经过统考验证的记忆封印到文件中...")
# state_dict() 就是模型所有权重参数的字典
torch.save(model.state_dict(), "mnist_brain.pth")
print("✅ 封印成功！记忆已永久保存至 mnist_brain.pth")