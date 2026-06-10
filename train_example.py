"""
训练示例脚本 — 基于已有权重继续训练(微调)
"""
from ultralytics import YOLO
import datetime

# ======================== 配置区 ========================
MODEL_PATH = ""      # 预训练权重路径(best.pt)
DATA_YAML = ""       # 数据集配置文件路径(classes.yaml)
EPOCHS = 200         # 训练轮数
IMGSZ = 640          # 输入尺寸
BATCH = 32           # 批大小
PATIENCE = 50        # 早停耐心值
LR0 = 2e-5           # 初始学习率(微调时用小学习率)
LRF = 0.1            # 最终学习率系数
# ======================================================

if __name__ == '__main__':
    model = YOLO(MODEL_PATH)

    # 生成训练文件夹名称
    train_name = datetime.datetime.now().strftime("train-%Y-%m-%d-%H-from-best")

    results = model.train(
        name=train_name,
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        cache='disk',
        amp=True,
        patience=PATIENCE,
        # 数据增强参数
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        degrees=5.0,
        translate=0.3,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.01,
        hsv_s=0.1,
        hsv_v=0.1,
        fliplr=0.5,
        flipud=0.0,
        # 优化器参数
        optimizer='Adam',
        lr0=LR0,
        lrf=LRF,
        warmup_epochs=0,
        cos_lr=False,
        weight_decay=0.0,
        pretrained=True,
        resume=False,
    )
