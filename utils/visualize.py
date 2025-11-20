import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_triplet(image, gt_mask, pred_mask, save_path=None, title=None):
    """
    可视化 原图 / 真值mask / 预测mask
    image: Tensor or np.array, shape: (3,H,W) or (H,W,3)
    gt_mask: Tensor or np.array, shape: (H,W)
    pred_mask: Tensor or np.array, shape: (H,W)
    save_path: 保存路径（可选）
    """

    # 转为 numpy
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
    if hasattr(gt_mask, 'cpu'):
        gt_mask = gt_mask.cpu().numpy()
    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()

    # (3,H,W) → (H,W,3)
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    # 归一化到 0~1（避免显示过暗/过亮）
    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
