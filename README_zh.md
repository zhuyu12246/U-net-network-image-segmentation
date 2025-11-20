# U-Net 医学图像分割

![Python](https://badgen.net/badge/Python/3.x/blue)
![PyTorch](https://badgen.net/badge/PyTorch/1.x/red)
![许可证](https://badgen.net/badge/license/MIT/green)
![状态](https://badgen.net/badge/status/stable/green)

[English](README.md) | [中文](README_zh.md)

此存储库包含了使用 PyTorch 实现的 U-Net 医学图像分割模型。该模型旨在准确分割医学图像中的感兴趣区域，这对诊断和治疗计划至关重要。

## 目录
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [训练](#训练)
- [测试](#测试)
- [评估指标](#评估指标)
- [可视化](#可视化)
- [依赖](#依赖)
- [使用方法](#使用方法)
- [结果](#结果)

## 项目结构

```
.
├── model.py              # U-Net 模型实现
├── dataset.py            # 自定义数据集类，用于加载图像和掩码
├── transforms.py         # 数据增强和预处理转换
├── train.py              # 训练脚本
├── test.py               # 测试和推理脚本
├── utils/
│   ├── metrics.py        # 评估指标 (例如 Dice 系数)
│   └── visualize.py      # 可视化工具
├── README.md             # 英文项目文档
├── README_zh.md          # 中文项目文档
└── train_log.txt         # 训练日志
```

## 模型架构

该实现遵循经典的 U-Net 架构，具有编码器-解码器结构：

- **编码器路径**: 一系列带有下采样操作的卷积块
- **瓶颈层**: 中心特征提取层
- **解码器路径**: 带有来自编码器的跳跃连接的上采样层
- **输出层**: 最终的分割图

主要组件：
- 带有批归一化和 Dropout (0.3) 的卷积块
- LeakyReLU 激活函数
- 下采样和上采样模块
- 跳跃连接以保留空间信息

## 数据集

`UNetDataset` 类处理配对图像和分割掩码的加载：

- 图像以 RGB 格式加载
- 掩码以灰度格式加载
- 支持自定义数据增强转换

## 训练

训练流程包括：

- 数据增强：水平/垂直翻转、旋转和亮度调整
- 带有 logits 的二元交叉熵损失函数
- Adam 优化器，学习率为 1e-4
- 批大小为 4
- 可配置的训练周期数（默认 20）

训练模型：
```bash
python train.py
```

模型检查点保存在 `model_save/` 目录中。

## 测试

测试脚本对测试数据执行推理：

- 加载训练好的模型权重
- 不使用增强处理图像
- 生成二值分割掩码
- 将预测结果保存为 `results/` 目录中的 PNG 图像

运行推理：
```bash
python test.py
```

## 评估指标

我们使用 Dice 系数来评估分割性能：

- 范围：0（无重叠）到 1（完美重叠）
- 对医学图像中的类别不平衡具有鲁棒性
- 计算公式：2 * |X ∩ Y| / (|X| + |Y|)

实现在 [utils/metrics.py](utils/metrics.py) 中。

## 可视化

可视化工具创建以下内容的并排比较：
- 原始输入图像
- 真实分割掩码
- 预测分割掩码

这有助于定性评估模型性能。实现在 [utils/visualize.py](utils/visualize.py) 中。

## 依赖

- Python 3.x
- PyTorch
- Torchvision
- Albumentations
- OpenCV
- NumPy
- Matplotlib

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集，将图像和相应的掩码放在单独的文件夹中
2. 根据需要更新 `train.py` 和 `test.py` 中的数据路径
3. 运行训练：`python train.py`
4. 运行推理：`python test.py`
5. 在 `results/` 目录中查看结果

## 结果

该模型在医学图像分割任务上实现了有竞争力的性能。运行测试脚本后，可以在 `results/` 目录中找到示例结果。

## 许可证

该项目采用 MIT 许可证发布。