"""
视频预测脚本 — 逐帧检测，每12帧合并为一张3x4宫格图片
"""
from ultralytics import YOLO
import os
import cv2
import numpy as np

# ======================== 配置区 ========================
MODEL_PATH = ""          # 模型权重路径
SOURCE = ""              # 视频路径
OUTPUT_DIR = ""          # 输出文件夹路径
IMG_SIZE = 640           # 输入尺寸(与训练阶段一致)
FRAMES_PER_GRID = 12    # 每张宫格图包含的帧数
GRID_ROWS = 3           # 宫格行数
GRID_COLS = 4           # 宫格列数
# ======================================================


def build_grid(buffer, rows, cols):
    """将图片列表拼成 rows x cols 网格图"""
    h, w = buffer[0].shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for i, img in enumerate(buffer):
        row, col = i // cols, i % cols
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
    return grid


def main():
    # 检查路径
    if not os.path.exists(SOURCE):
        print(f"文件 {SOURCE} 不存在！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    model = YOLO(MODEL_PATH)

    print("开始逐帧检测...")
    frame_idx = 0
    batch_idx = 0
    buffer = []

    for result in model(SOURCE, imgsz=(IMG_SIZE, IMG_SIZE), stream=True, verbose=False):
        n_det = len(result.boxes) if result.boxes else 0
        buffer.append(result.plot())

        # 累积满一页则保存
        if len(buffer) == FRAMES_PER_GRID:
            grid = build_grid(buffer, GRID_ROWS, GRID_COLS)
            save_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}.jpg")
            cv2.imwrite(save_path, grid)
            print(f"  批次 {batch_idx} 已保存 (帧 {frame_idx - FRAMES_PER_GRID + 1} ~ {frame_idx})")
            batch_idx += 1
            buffer.clear()

        if frame_idx % 30 == 0:
            print(f"  处理第 {frame_idx} 帧，检测到 {n_det} 个目标")
        frame_idx += 1

    # 处理剩余不足一页的帧
    if buffer:
        grid = build_grid(buffer, GRID_ROWS, GRID_COLS)
        save_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}.jpg")
        cv2.imwrite(save_path, grid)
        batch_idx += 1

    total_batches = batch_idx
    print(f"\n预测完成！共处理 {frame_idx} 帧，生成 {total_batches} 张宫格图片")
    print(f"结果图片已保存至: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
