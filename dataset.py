import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class UNetDataset(Dataset):
    """
    UNet数据集类，用于加载图像和对应的掩码文件

    参数:
        image_dir (str): 图像文件所在的目录路径
        mask_dir (str): 掩码文件所在的目录路径
        transform (callable, optional): 应用于图像和掩码的变换函数，默认为None
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取并排序图像文件列表
        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):
        """
        返回数据集中样本的数量

        返回:
            int: 数据集中图像文件的数量
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取指定索引位置的图像和对应掩码

        参数:
            idx (int): 样本的索引位置

        返回:
            tuple: 包含图像数据和掩码数据的元组
        """
        image_name = self.images[idx]

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        # 加载图像并转换为RGB格式
        img = Image.open(img_path).convert("RGB")
        # 加载掩码并转换为灰度格式
        mask = Image.open(mask_path).convert("L")

        # 转换为numpy数组
        img = np.array(img)
        mask = np.array(mask)

        # 如果有变换函数，则应用变换
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask

