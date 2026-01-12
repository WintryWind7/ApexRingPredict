"""测试地图和毒圈的叠加效果"""

from pathlib import Path
import cv2
import numpy as np


def overlay_map_and_rings(
    map_path: str,
    ring_path: str,
    output_path: str = "tests/temp/overlay_result.png"
) -> None:
    """
    将毒圈图片叠加到地图上
    
    Args:
        map_path: 地图文件路径
        ring_path: 毒圈文件路径
        output_path: 输出文件路径
    """
    # 读取地图（4096x4096）
    map_img = cv2.imread(map_path)
    if map_img is None:
        raise ValueError(f"无法读取地图: {map_path}")
    
    # 读取毒圈（2234x1436，带 alpha 通道）
    ring_img = cv2.imread(ring_path, cv2.IMREAD_UNCHANGED)
    if ring_img is None:
        raise ValueError(f"无法读取毒圈: {ring_path}")
    
    print(f"地图尺寸: {map_img.shape}")
    print(f"毒圈尺寸: {ring_img.shape}")
    
    # 毒圈放大 4 倍
    ring_scaled = cv2.resize(
        ring_img,
        (ring_img.shape[1] * 4, ring_img.shape[0] * 4),
        interpolation=cv2.INTER_LINEAR
    )
    print(f"毒圈放大后: {ring_scaled.shape}")
    
    # 计算有效区域
    map_h, map_w = map_img.shape[:2]
    ring_h, ring_w = ring_scaled.shape[:2]
    
    # 偏移量（居中对齐）
    offset_x = (map_w - ring_w) // 2
    offset_y = (map_h - ring_h) // 2
    
    print(f"偏移量: ({offset_x}, {offset_y})")
    
    # 毒圈在地图上的起始位置
    start_x = max(0, offset_x)
    start_y = max(0, offset_y)
    end_x = min(map_w, offset_x + ring_w)
    end_y = min(map_h, offset_y + ring_h)
    
    # 毒圈图片中对应的区域
    ring_start_x = max(0, -offset_x)
    ring_start_y = max(0, -offset_y)
    ring_end_x = ring_start_x + (end_x - start_x)
    ring_end_y = ring_start_y + (end_y - start_y)
    
    # 提取毒圈的有效部分
    ring_region = ring_scaled[ring_start_y:ring_end_y, ring_start_x:ring_end_x]
    
    # 分离 RGB 和 alpha 通道
    ring_rgb = ring_region[:, :, :3]
    ring_alpha = ring_region[:, :, 3:4] / 255.0
    
    # 叠加到地图上
    map_region = map_img[start_y:end_y, start_x:end_x]
    blended = (ring_rgb * ring_alpha + map_region * (1 - ring_alpha)).astype(np.uint8)
    map_img[start_y:end_y, start_x:end_x] = blended
    
    # 保存结果
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path_obj), map_img)
    
    print(f"\n叠加结果已保存到: {output_path}")


if __name__ == "__main__":
    overlay_map_and_rings(
        map_path="data/mp_rr_tropic_island_mu2.png",
        ring_path="data/map.png"
    )
