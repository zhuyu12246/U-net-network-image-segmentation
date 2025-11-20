import albumentations as A
from albumentations.pytorch import ToTensorV2

# 训练数据增强变换管道
# 包含多种数据增强操作用于提高模型泛化能力：
# - 随机水平翻转(50%概率)
# - 随机垂直翻转(50%概率)
# - 随机旋转90度倍数(50%概率)
# - 随机亮度对比度调整(30%概率)
# - 数据标准化处理
# - 转换为PyTorch张量格式
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

# 测试数据预处理变换管道
# 仅包含必要的预处理操作：
# - 数据标准化处理(与训练集保持一致)
# - 转换为PyTorch张量格式
test_transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

