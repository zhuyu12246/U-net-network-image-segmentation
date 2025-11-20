import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入灰度图
img = cv2.imread("Data/test/mask/1.png", cv2.IMREAD_GRAYSCALE)

print("shape:", img.shape)
print("dtype:", img.dtype)
print("min,max:", img.min(), img.max())

# 绘制直方图
plt.figure(figsize=(6,4))
plt.hist(img.flatten(), bins=256, range=(0,255))
plt.title("Pixel Value Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.show()
