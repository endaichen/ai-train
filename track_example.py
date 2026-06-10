"""
跟踪示例脚本 — 使用YOLO模型对视频进行目标跟踪
"""
from ultralytics import YOLO

# ======================== 配置区 ========================
MODEL_PATH = ""    # 模型权重路径
VIDEO_PATH = ""    # 视频文件路径
# ======================================================

model = YOLO(MODEL_PATH)
results = model.track(VIDEO_PATH, save=True)
