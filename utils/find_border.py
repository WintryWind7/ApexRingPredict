"""清理毒圈图片，移除干扰像素"""

from pathlib import Path
import cv2
import numpy as np
import sys


def clean_ring_image(input_path: str, output_path: str) -> None:
    """
    清理毒圈图片，移除 alpha > 247 的白色干扰像素
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
    """
    # 读取图片
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图片: {input_path}")
    
    if img.shape[2] != 4:
        raise ValueError("图片必须包含 alpha 通道")
    
    print(f"原始图片尺寸: {img.shape}")
    
    alpha = img[:, :, 3]
    
    # 统计原始像素
    total_pixels = img.shape[0] * img.shape[1]
    non_transparent = np.sum(alpha > 0)
    
    print(f"总像素数: {total_pixels}")
    print(f"非透明像素数: {non_transparent}")
    
    # 找出需要移除的像素：alpha > 247 且为纯白色 (255, 255, 255)
    high_alpha_mask = alpha > 247
    white_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
    remove_mask = high_alpha_mask & white_mask
    
    removed_count = np.sum(remove_mask)
    print(f"移除的像素数: {removed_count} (alpha > 247 的纯白色)")
    
    # 创建新图片
    result = img.copy()
    
    # 将需要移除的像素设为完全透明
    result[remove_mask] = [0, 0, 0, 0]
    
    # 保存结果
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path_obj), result)
    
    print(f"\n清理后的图片已保存到: {output_path}")
    print(f"保留的非透明像素数: {non_transparent - removed_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python find_border.py <输入图片> [输出图片]")
        print("示例: python find_border.py data/map1.png")
        print("      python find_border.py data/map1.png output/map1_border.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # 默认输出到 tests/temp 目录
        input_path = Path(input_file)
        output_file = f"tests/temp/{input_path.stem}_border{input_path.suffix}"
    
    clean_ring_image(input_file, output_file)
