import os

import torch
from torch.utils.data import DataLoader
from dataset import UNetDataset
from transforms import train_transform
from model import UNet
import torch.nn as nn
import torch.optim as optim

# 创建训练数据集实例，指定图像和掩码目录，并应用训练时的数据增强变换
train_dataset = UNetDataset(
    image_dir="Data/train/image",
    mask_dir="Data/train/mask",
    transform=train_transform
)
batch_size = 4
# 创建数据加载器，设置批次大小为4并打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 设置设备：如果CUDA可用则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化UNet模型并将其移动到指定设备上
model = UNet().to(device)
# 定义损失函数为二分类交叉熵损失（适用于分割任务）
criterion = nn.BCEWithLogitsLoss()
# 定义优化器为Adam，学习率设置为1e-4
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 主要进行测试 20轮快一点
# 训练循环：共进行20个epoch
epochs = 20
print("start---------------------------------------")
for epoch in range(epochs):
    # 将模型设置为训练模式
    model.train()
    all_loss  = 0
    # 遍历训练数据加载器中的每个批次
    for imgs, masks in train_loader:
        # 将图像和掩码数据移动到指定设备上，并将掩码转换为浮点型
        imgs = imgs.to(device)
        masks = masks.unsqueeze(1)
        masks = masks.to(device).float() / 255.0

        # 前向传播：通过模型获取预测结果
        preds = model(imgs)
        # 计算预测结果与真实标签之间的损失
        loss = criterion(preds, masks)
        all_loss += loss.item()

        # 清零优化器的梯度缓存
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        print(f"batch loss:{loss.item()}")

    # 打印当前epoch的损失值
    print(f"Epoch {epoch+1}, Loss = {(all_loss / 20):.4f}")
# 创建保存结果的目录，如果目录已存在则不会报错
os.makedirs("model_save", exist_ok=True)
# 保存训练好的模型参数到文件
torch.save(model.state_dict(), "model_save/unet.pth")
