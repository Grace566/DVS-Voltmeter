import cv2
import numpy as np
import math

def calculate_psnr(img1, img2):
    """
    计算两张图片的 PSNR (峰值信噪比)

    参数:
        img1 (numpy.ndarray): 第一张图片，范围 [0, 255]。
        img2 (numpy.ndarray): 第二张图片，范围 [0, 255]。

    返回:
        float: PSNR 值。
    """
    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同的图像，PSNR是无穷大
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def adjust_global_brightness(image, brightness_factor):
    """
    调整图像的全局亮度。

    参数:
        image (numpy.ndarray): 输入图像（灰度图或彩色图）。
        brightness_factor (float): 全局亮度增强因子。

    返回:
        numpy.ndarray: 调整后的图像。
    """
    # 转换为浮点型，避免溢出
    image = image.astype(np.float32) / 255.0

    # 全局亮度调整
    adjusted_image = np.clip(image * brightness_factor, 0, 1)

    # 转换回 8 位整数图像
    return (adjusted_image * 255).astype(np.uint8)


# 文件路径
reference_image_path = r"H:\Dataset\EventNeRF\chair\train\grayscale_gt_mask\r_00473.png"
target_image_path = r"H:\Dataset\EventNeRF\chair\train\Exposure\OverExp\exposure_events_gray\r_00473.png"
output_image_path = r"H:\Dataset\EventNeRF\chair\train\Exposure\OverExp\exposure_events_gray\r_00473_adjust_brightness.png"

# 读取参考图像和待处理图像
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)  # 读取参考图像
target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)  # 读取待处理图像

# 初始化参数
best_psnr = -1
best_brightness_factor = 1.0
brightness_factors = np.arange(0.5, 2.0, 0.01)  # 测试亮度因子的范围

# 遍历不同的亮度因子，寻找最佳因子
# for factor in brightness_factors:
if 1:
    factor = 1
    # 调整亮度
    adjusted_image = adjust_global_brightness(target_image, factor)

    # 计算与参考图像的 PSNR
    current_psnr = calculate_psnr(reference_image, adjusted_image)

    # 更新最佳亮度因子
    if current_psnr > best_psnr:
        best_psnr = current_psnr
        best_brightness_factor = factor

print(f"Best Brightness Factor: {best_brightness_factor}")
print(f"Best PSNR: {best_psnr}")

# 使用最佳亮度因子调整图像
final_adjusted_image = adjust_global_brightness(target_image, best_brightness_factor)

# 保存结果
cv2.imwrite(output_image_path, final_adjusted_image)
print(f"Adjusted image saved at: {output_image_path}")
