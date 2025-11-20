import torch

def dice_score(pred, target, epsilon=1e-6):
    """
    计算预测结果与目标之间的Dice相似度分数

    Dice系数是用于评估两个样本相似度的统计学指标，常用于图像分割任务中评估预测结果的准确性。
    Dice分数范围在[0,1]之间，1表示完全匹配，0表示无重叠。

    参数:
        pred (torch.Tensor): 预测结果张量，通常包含概率值
        target (torch.Tensor): 目标标签张量，通常为二值化的ground truth
        epsilon (float): 防止除零错误的小常数，默认值为1e-6

    返回:
        torch.Tensor: Dice相似度分数
    """
    # 将预测值和目标值二值化处理
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    # 计算预测值与目标值的交集
    intersection = (pred * target).sum()

    # 根据Dice系数公式计算相似度分数
    return (2 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

