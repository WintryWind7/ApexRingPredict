"""根据坐标数据绘制毒圈到地图上"""

from pathlib import Path
import cv2
import numpy as np


def draw_rings_on_map(
    map_path: str,
    rings: list[dict],
    grid_size: int = 16384,
    output_path: str = "tests/temp/rings_on_map.png"
) -> None:
    """
    在地图上绘制坐标数据的毒圈
    
    Args:
        map_path: 地图文件路径
        rings: 毒圈数据列表，格式 [{"x": int, "y": int, "r": int}, ...]
        grid_size: 坐标系格子数（默认 16384）
        output_path: 输出图片路径
    """
    # 读取地图
    map_img = cv2.imread(map_path)
    if map_img is None:
        raise ValueError(f"无法读取地图: {map_path}")
    
    print(f"地图尺寸: {map_img.shape}")
    print(f"坐标系格子数: {grid_size}")
    
    # 计算坐标转换比例
    scale = map_img.shape[0] / grid_size
    print(f"转换比例: {scale:.6f}\n")
    
    # 绘制每个毒圈（黄色）
    for i, ring in enumerate(rings, 1):
        x_grid = ring["x"]
        y_grid = ring["y"]
        r_grid = ring["r"]
        
        # 转换为像素坐标
        x_pixel = int(x_grid * scale)
        y_pixel = int(y_grid * scale)
        r_pixel = int(r_grid * scale)
        
        print(f"毒圈 {i}:")
        print(f"  坐标系: ({x_grid}, {y_grid}), 半径: {r_grid}")
        print(f"  像素: ({x_pixel}, {y_pixel}), 半径: {r_pixel}")
        
        # 绘制黄色圆圈
        cv2.circle(map_img, (x_pixel, y_pixel), r_pixel, (0, 255, 255), 2)
    
    # 保存结果
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path_obj), map_img)
    
    print(f"\n图片已保存到: {output_path}")


if __name__ == "__main__":
    # 测试数据
    rings_data = [
        {"x": 9449, "y": 8074, "r": 4894},
        {"x": 11356, "y": 9671, "r": 2407},
        {"x": 12258, "y": 10333, "r": 1284}
    ]
    
    draw_rings_on_map(
        map_path="data/mp_rr_tropic_island_mu2.png",
        rings=rings_data
    )
