import os
import torch
from torch.utils.data import DataLoader
from dataset import UNetDataset
from transforms import test_transform
from model import UNet
import cv2
import numpy as np

# 创建测试数据集实例，用于加载测试图像和对应的掩码
# image_dir: 测试图像文件夹路径
# mask_dir: 测试掩码文件夹路径
# transform: 应用于测试数据的变换操作
test_dataset = UNetDataset(
    image_dir="Data/test/image",
    mask_dir="Data/test/mask",
    transform=test_transform
)

# 创建测试数据加载器，用于批量加载测试数据
# test_dataset: 测试数据集对象
# batch_size: 批次大小，设置为1表示每次处理一张图像
# shuffle: 是否打乱数据顺序，测试时设为False保持数据顺序一致
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化UNet模型并加载预训练权重
# UNet(): 创建UNet模型实例
# load_state_dict(): 加载训练好的模型权重参数
# map_location="cpu": 将模型加载到CPU上运行
# eval(): 设置模型为评估模式，关闭dropout和batch normalization的训练行为
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()
model.load_state_dict(torch.load("model_save/unet.pth", map_location="cpu"))
model.eval()

# 创建保存结果的目录，如果目录已存在则不会报错
os.makedirs("results", exist_ok=True)

# 遍历测试数据集，对每张图像进行预测并保存结果
# idx: 图像索引
# img: 输入的测试图像数据
# _: 图像对应的标签数据（此处未使用）
for idx, (img, _) in enumerate(test_loader):
    # 禁用梯度计算以节省内存和提高推理速度
    with torch.no_grad():
        # 模型前向传播得到预测结果，并通过sigmoid激活函数转换为概率值
        pred = model(img).sigmoid()
        # 将概率值二值化处理，大于0.5的置为1，小于等于0.5的置为0
        pred = (pred > 0.5).float()

    # 将预测结果从tensor转换为numpy数组，并调整数值范围到0-255
    pred_np = pred.squeeze().cpu().numpy() * 255

    # 将预测结果保存为PNG格式图像文件
    cv2.imwrite(f"results/{idx}.png", pred_np)


# 可视化预测结果
# from utils.visualize import visualize_triplet
#
# for idx, (img, mask) in enumerate(test_loader):
#     img = img.to(device)
#     mask = mask.to(device)
#
#     with torch.no_grad():
#         pred = model(img).sigmoid()
#         pred_mask = (pred > 0.5).float()
#
#     visualize_triplet(
#         image=img[0],
#         gt_mask=mask[0],
#         pred_mask=pred_mask[0],
#         save_path=f"results/vis/{idx}.png"
#     )

