# feather_rotate_utils.py
import numpy as np
import math
from skimage.transform import rotate
from scipy.ndimage import distance_transform_edt, gaussian_filter

def get_outer_ring_mean(image: np.ndarray, border: int) -> float:
    """
    计算图像最外面 border 圈（上下各border行 + 左右各border列）的平均值。
    用于在旋转时作为 cval 填充。
    如果图像大小比 2*border 小，会出现索引重叠，请根据需要处理。
    """
    if image.shape[0] < 2*border or image.shape[1] < 2*border:
        # 如果图像尺寸不足 2*border，简单返回整幅图的平均值
        return float(image.mean())

    top_rows    = image[:border, :]
    bottom_rows = image[-border:, :]
    left_cols   = image[:, :border]
    right_cols  = image[:, -border:]

    border_pixels = np.concatenate([
        top_rows.ravel(),
        bottom_rows.ravel(),
        left_cols.ravel(),
        right_cols.ravel()
    ])

    return float(border_pixels.mean())


def feather_rotate_cosine_with_blur(
    image: np.ndarray,
    angle: float,
    cval: float,
    fade_width: int = 1,
    gaussian_sigma: float = 2.0,
    valid_threshold: float = 1e-20
) -> np.ndarray:
    """
    1) 对 image 做 rotate，并使用外圈均值 cval 填充。
    2) 对(原图>threshold)构造掩码，并做相同旋转。
    3) 用高斯模糊 + 距离场 + 余弦函数做边缘过渡，以消除“硬”边。
       - fade_width: 控制淡出宽度
       - gaussian_sigma: 高斯模糊标准差
       - valid_threshold: 判断像素是否有效的阈值
    返回最终平滑过渡后的图像。
    """

    # ---------- 1) 旋转图像并用外圈均值填充 ----------
    rot_img = rotate(
        image,
        angle=angle,
        resize=False,
        mode='constant',
        cval=cval,
        preserve_range=True
    )

    # ---------- 2) 构造掩码并旋转 ----------
    mask = (image > valid_threshold).astype(np.float32)
    rot_mask = rotate(
        mask,
        angle=angle,
        resize=False,
        mode='constant',
        cval=0,
        preserve_range=True
    )

    # ---------- 3) 高斯模糊 + 二值化 ----------
    rot_mask_blurred = gaussian_filter(rot_mask, sigma=gaussian_sigma)
    rot_mask_bin = (rot_mask_blurred > 0.5).astype(np.uint8)

    # ---------- 4) 计算距离场并映射到 alpha ----------
    dist_map = distance_transform_edt(rot_mask_bin == 0)

    # t in [0,1], alpha = 0.5 * (1 + cos(pi * t))
    t = np.clip(dist_map / fade_width, 0, 1)
    alpha = 0.5 * (1 + np.cos(np.pi * t))

    # ---------- 5) 混合 ----------
    cval_img = np.full_like(rot_img, cval, dtype=rot_img.dtype)
    final_img = alpha * rot_img + (1 - alpha) * cval_img

    return final_img.astype(image.dtype)
